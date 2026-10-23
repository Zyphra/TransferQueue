"""Offline evidence retains native bytes without granting episode or restore credit."""

import base64
import gzip
import hashlib
import io
import json
import pickle
from types import SimpleNamespace

import pytest
import torch

from transfer_queue.lean_archive import (
    audit_index,
    completion_plan,
    extract_unit,
    load_pickle,
    main,
    project_row,
    terminal_sessions,
)


def partition(tags):
    keys = {key: index for index, key in enumerate(tags)}
    return SimpleNamespace(
        keys_mapping=keys,
        revert_keys_mapping={index: key for key, index in keys.items()},
        custom_meta={keys[key]: value for key, value in tags.items()},
        field_name_mapping={"input_ids": 0, "response_mask": 1, "rm_scores": 2},
        production_status=torch.ones((len(keys), 3), dtype=torch.int8),
    )


@pytest.fixture
def row():
    return {
        "key": "prompt_0_0",
        "global_index": 0,
        "unit_position": 0,
        "produced_fields": ["input_ids", "response_mask", "rm_scores", "uid", "session_id", "extra_fields"],
        "signals": {"prompt_len": 2},
    }


@pytest.fixture
def fields():
    return {
        "input_ids": {0: torch.tensor([10, 11, 12, 13])},
        "response_mask": {0: torch.tensor([1, 0])},
        "rm_scores": {0: torch.tensor([0.0, float("nan")], dtype=torch.float32)},
        "uid": {0: "prompt"},
        "session_id": {0: 0},
        "routed_experts": {0: torch.ones((4, 2, 2), dtype=torch.int16)},
        "extra_fields": {
            0: {
                "infra_failure": "timeout",
                "huge_debug": "not archived",
                "episode_metrics": {"reward": 0.25, "scored": True, "huge_grader": "omitted"},
            }
        },
    }


def plan_for(row):
    return {
        "schema": "transfer_queue.lean_plan.v1",
        "units": 1,
        "rows": [row],
        "storage_units": [{"position": 0, "storage_unit_id": "unit"}],
        "controller": {"sha256": "c" * 64},
    }


def test_preserves_reward_bits_and_omits_full_payload(row, fields):
    record = project_row(fields, row)
    assert base64.b64decode(record["reward"]["data"]) == fields["rm_scores"][0].numpy().tobytes()
    assert record["reward"]["dtype"] == "torch.float32"
    assert record["signals"]["infra_failure"] == "timeout"
    assert record["signals"]["episode_metrics"] == {"reward": 0.25, "scored": True}
    assert "routed_experts" not in record and "huge_debug" not in record["signals"]


def test_missing_reward_and_zero_response_are_not_invented(row, fields):
    fields["input_ids"][0] = torch.tensor([10, 11])
    fields["response_mask"][0] = torch.empty(0, dtype=torch.int64)
    del fields["rm_scores"]
    row["produced_fields"].remove("rm_scores")
    record = project_row(fields, row)
    assert record["reward"] is None
    assert record["masks"]["response_mask"]["shape"] == [0]
    row["produced_fields"].append("rm_scores")
    with pytest.raises(ValueError, match="reward presence"):
        project_row(fields, row)


@pytest.mark.parametrize("change", ["mask", "float_tokens", "nested", "unaligned_reward"])
def test_rejects_unsupported_or_unaligned_rows(row, fields, change):
    if change == "mask":
        fields["loss_mask"] = {0: torch.tensor([1])}
        row["produced_fields"].append("loss_mask")
    elif change == "float_tokens":
        fields["input_ids"][0] = fields["input_ids"][0].float()
    elif change == "nested":
        fields["input_ids"][0] = torch.ones((2, 2), dtype=torch.int64)
    else:
        fields["rm_scores"][0] = torch.zeros(3)
    with pytest.raises(ValueError):
        project_row(fields, row)


def test_session_completion_requires_prompt_terminal_contiguous_rows_and_terminal_carrier():
    tags = {
        "uid_with_underscore": {
            "is_prompt": True,
            "expected_session_count": 2,
            "status": "finished",
            "failure_records": [
                {"session_id": 1, "category": "infrastructure", "error_message": "large secret traceback omitted"}
            ],
        },
        "uid_with_underscore_0_0": {"status": "success"},
        "uid_with_underscore_0_1": {"status": "success", "episode_metrics": {}},
    }
    result = terminal_sessions(partition(tags))
    assert result["all_episode_metadata_complete"] is True
    assert result["groups"][0]["failed_sessions"] == [{"session_id": 1, "category": "infrastructure"}]
    tags["uid_with_underscore"]["status"] = "running"
    assert terminal_sessions(partition(tags))["all_episode_metadata_complete"] is False
    tags["uid_with_underscore"]["status"] = "finished"
    del tags["uid_with_underscore_0_0"]
    assert terminal_sessions(partition(tags))["groups"][0]["unknown_sessions"] == [0]


def test_prompt_abort_and_overlapping_failure_keep_completion_unknown():
    tags = {
        "p": {
            "is_prompt": True,
            "expected_session_count": 1,
            "status": "failure",
            "failure_records": [{"session_id": None, "category": "abort"}, {"session_id": 0}],
        },
        "p_0_0": {"status": "success", "episode_metrics": {}},
    }
    result = terminal_sessions(partition(tags))
    assert result["all_episode_metadata_complete"] is False
    assert result["groups"][0]["unknown_sessions"] == [0]
    assert result["groups"][0]["prompt_failures"] == [{"session_id": None, "category": "abort"}]
    assert terminal_sessions(partition({}))["all_episode_metadata_complete"] is False


def test_publication_does_not_imply_episodes_and_reward_can_be_absent():
    p = partition({"p_0_0": {"status": "success", "prompt_len": 2}})
    p.production_status[0, 2] = 0
    plan = completion_plan({"partitions": {"train": p}}, 1, "status", "success", "train")
    assert "rm_scores" not in plan["rows"][0]["produced_fields"]
    assert plan["episode_completeness"].startswith("Unknown")
    p.revert_keys_mapping = {}
    with pytest.raises(ValueError, match="mapping"):
        completion_plan({"partitions": {"train": p}}, 1, "status", "success", "train")


def test_unproduced_stale_payload_is_not_archived(row, fields):
    row["produced_fields"].remove("rm_scores")
    assert project_row(fields, row)["reward"] is None
    fields["uid"][0] = "another-prompt"
    with pytest.raises(ValueError, match="session identity"):
        project_row(fields, row)


def test_unit_identity_and_modulo_are_bound(row, fields):
    state = {"field_data": fields, "storage_unit_id": "foreign-unit"}
    with pytest.raises(ValueError, match="identity"):
        extract_unit(state, plan_for(row), 0, io.BytesIO())
    state["storage_unit_id"] = "unit"
    plan = plan_for(row)
    plan["units"] = 2
    fields["input_ids"][1] = torch.ones(2, dtype=torch.int64)
    with pytest.raises(ValueError, match="another unit"):
        extract_unit(state, plan, 0, io.BytesIO())


def test_index_assertions_never_prove_episode_coverage(row):
    plan = plan_for(row)
    record = {
        "key": row["key"],
        "source_record_sha256": "a" * 64,
        "controller_sha256": "c" * 64,
        "token_ids_present": True,
        "masks_present": True,
        "reward_present": True,
    }
    assert audit_index(plan, [record])["all_expected_rows_attested"] is True
    assert audit_index(plan, [record])["episode_coverage_proved"] is False
    record["controller_sha256"] = "wrong"
    assert audit_index(plan, [record])["coverage_unknown_rows"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        audit_index(plan, [record, record])
    plan["rows"] = []
    assert audit_index(plan, [])["all_expected_rows_attested"] is False


def test_trusted_pickle_bound_hash_and_symlink(tmp_path):
    path = tmp_path / "input.pkl"
    path.write_bytes(pickle.dumps({"small": 1}))
    value, source = load_pickle(path, path.stat().st_size)
    assert value == {"small": 1} and source["ino"] == path.stat().st_ino
    with pytest.raises(ValueError, match="bound"):
        load_pickle(path, 1)
    with pytest.raises(ValueError, match="SHA-256"):
        load_pickle(path, 1000, "0" * 64)
    link = tmp_path / "link.pkl"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="regular"):
        load_pickle(link, 1000)


def test_extract_cli_manifest_is_last_and_no_overwrite(tmp_path, row, fields):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_for(row)))
    unit = tmp_path / "unit.pkl"
    unit.write_bytes(pickle.dumps({"field_data": fields, "storage_unit_id": "unit"}))
    output = tmp_path / "archive"
    argv = [
        "extract",
        "--plan",
        str(plan),
        "--plan-sha256",
        hashlib.sha256(plan.read_bytes()).hexdigest(),
        "--unit",
        str(unit),
        "--max-unit-bytes",
        str(unit.stat().st_size),
        "--unit-position",
        "0",
        "--output",
        str(output),
    ]
    main(argv)
    assert not (output / "INCOMPLETE").exists()
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["completed_rows"] == 1
    assert json.loads(gzip.decompress((output / "completed.jsonl.gz").read_bytes()))["key"] == row["key"]
    with pytest.raises(FileExistsError):
        main(argv)
    fields["rm_scores"].clear()
    unit.write_bytes(pickle.dumps({"field_data": fields, "storage_unit_id": "unit"}))
    failed = tmp_path / "failed"
    argv[-1] = str(failed)
    with pytest.raises(ValueError):
        main(argv)
    assert (failed / "INCOMPLETE").exists() and not (failed / "manifest.json").exists()


def test_plan_cli_reads_native_partition_snapshot_and_binds_unit_manifest(tmp_path):
    from transfer_queue.controller import DataPartitionStatus

    native = DataPartitionStatus(partition_id="train", global_indexes={0})
    sample = partition({"p_0_0": {"status": "success", "prompt_len": 2}})
    for name in ("keys_mapping", "revert_keys_mapping", "custom_meta", "field_name_mapping", "production_status"):
        setattr(native, name, getattr(sample, name))
    controller = tmp_path / "controller.pkl"
    controller.write_bytes(pickle.dumps({"partitions": {"train": native.to_snapshot()}}))
    units = tmp_path / "units.json"
    units.write_text(json.dumps([{"position": 0, "storage_unit_id": "unit"}]))
    output = tmp_path / "plan.json"
    main(
        [
            "plan",
            "--controller",
            str(controller),
            "--controller-sha256",
            hashlib.sha256(controller.read_bytes()).hexdigest(),
            "--max-controller-bytes",
            str(controller.stat().st_size),
            "--unit-manifest",
            str(units),
            "--unit-manifest-sha256",
            hashlib.sha256(units.read_bytes()).hexdigest(),
            "--completed-tag",
            "status",
            "--completed-value",
            "success",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_bytes())
    assert result["rows"][0]["key"] == "p_0_0"
    assert result["storage_units"] == [{"position": 0, "storage_unit_id": "unit"}]
    assert result["controller"]["sha256"] == hashlib.sha256(controller.read_bytes()).hexdigest()


@pytest.fixture
def extracted_archive(tmp_path, row, fields):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_for(row)))
    unit = tmp_path / "unit.pkl"
    unit.write_bytes(pickle.dumps({"field_data": fields, "storage_unit_id": "unit"}))
    archive = tmp_path / "archive"
    main(
        [
            "extract",
            "--plan",
            str(plan),
            "--plan-sha256",
            hashlib.sha256(plan.read_bytes()).hexdigest(),
            "--unit",
            str(unit),
            "--max-unit-bytes",
            str(unit.stat().st_size),
            "--unit-position",
            "0",
            "--output",
            str(archive),
        ]
    )
    return plan, archive


def verify_args(plan, archive, output):
    return [
        "verify",
        "--plan",
        str(plan),
        "--plan-sha256",
        hashlib.sha256(plan.read_bytes()).hexdigest(),
        "--archive",
        str(archive),
        "--manifest-sha256",
        hashlib.sha256((archive / "manifest.json").read_bytes()).hexdigest(),
        "--max-uncompressed-bytes",
        "1048576",
        "--output",
        str(output),
    ]


def test_verify_independently_decompresses_without_native_pickle_read(extracted_archive, tmp_path, monkeypatch):
    import transfer_queue.lean_archive as module

    plan, archive = extracted_archive
    monkeypatch.setattr(module, "load_pickle", lambda *args: pytest.fail("verification reopened native state"))
    output = tmp_path / "verified.json"
    main(verify_args(plan, archive, output))
    result = json.loads(output.read_bytes())
    assert result["status"] == "PASS_ARCHIVE_BYTES_AND_PLANNED_ROWS"
    assert result["verified_rows"] == 1 and result["episode_coverage_proved"] is False
    assert (
        result["uncompressed_records_sha256"]
        == hashlib.sha256(gzip.decompress((archive / "completed.jsonl.gz").read_bytes())).hexdigest()
    )


@pytest.mark.parametrize(
    "damage",
    [
        "crc",
        "truncated",
        "key",
        "records_digest",
        "keys_digest",
        "count",
        "extra",
        "missing",
        "incomplete",
        "plan_binding",
        "bounds",
    ],
)
def test_verify_refuses_corrupt_truncated_or_misbound_archives(extracted_archive, tmp_path, damage):
    plan, archive = extracted_archive
    path = archive / "completed.jsonl.gz"
    manifest = json.loads((archive / "manifest.json").read_bytes())
    payload = path.read_bytes()
    if damage == "crc":
        payload = payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:]
    elif damage == "truncated":
        payload = payload[:-4]
    elif damage == "key":
        record = json.loads(gzip.decompress(payload))
        record["key"] = "foreign_0_0"
        raw = json.dumps(record).encode() + b"\n"
        payload = gzip.compress(raw, mtime=0)
        manifest["uncompressed_records_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest["keys_sha256"] = hashlib.sha256(b'"foreign_0_0"\n').hexdigest()
    elif damage == "records_digest":
        manifest["uncompressed_records_sha256"] = "0" * 64
    elif damage == "keys_digest":
        manifest["keys_sha256"] = "0" * 64
    elif damage == "count":
        manifest["completed_rows"] += 1
    elif damage == "extra":
        payload = gzip.compress(gzip.decompress(payload) * 2, mtime=0)
    elif damage == "missing":
        payload = gzip.compress(b"", mtime=0)
    elif damage == "incomplete":
        (archive / "INCOMPLETE").touch()
    elif damage == "plan_binding":
        manifest["plan_sha256"] = "0" * 64
    path.write_bytes(payload)
    manifest.update(output_bytes=len(payload), output_sha256=hashlib.sha256(payload).hexdigest())
    (archive / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "should-not-pass.json"
    argv = verify_args(plan, archive, output)
    if damage == "bounds":
        argv[argv.index("--max-uncompressed-bytes") + 1] = "1"
    with pytest.raises((ValueError, OSError, EOFError)):
        main(argv)
    assert not output.exists()
