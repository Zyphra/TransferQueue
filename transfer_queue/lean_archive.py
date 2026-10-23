"""Offline, non-restorable archives of completed trajectory evidence."""

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import pickle
import stat
import sys
from collections import Counter
from pathlib import Path

import torch

PLAN_SCHEMA = "transfer_queue.lean_plan.v1"
RECORD_SCHEMA = "transfer_queue.lean_trajectory.v1"
LIGHT_KEYS = (
    "status",
    "data_source",
    "task_id",
    "global_steps",
    "min_global_steps",
    "max_global_steps",
    "prompt_len",
    "response_len",
    "seq_len",
    "infra_failure",
    "reward_status",
    "is_loss_masked",
    "verification_disposition",
    "training_disposition",
    "reward_gate_decisions",
)
MASK_FIELDS = ("response_mask", "loss_mask", "attention_mask", "generated_response_mask")
EPISODE_LIGHT_KEYS = (
    "done_reason",
    "generated_tokens",
    "prompt_tokens_end",
    "turns",
    "scored",
    "masked",
    "zero",
    "reward",
    "termination_evidence",
    "infra_stage",
    "infra_cause",
)


def file_identity(value):
    return {name: getattr(value, "st_" + name) for name in ("dev", "ino", "uid", "size", "mtime_ns", "ctime_ns")}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


class _HashingReader(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.stream.readinto(buffer)
        self.digest.update(memoryview(buffer)[:count])
        return count


def load_pickle(path, max_bytes, expected_sha256=None):
    """Read a trusted native pickle once; never call this on an untrusted file."""
    path = Path(path)
    before = path.lstat()
    if max_bytes <= 0 or not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
        raise ValueError("pickle must be a regular file within the declared input bound")
    if before.st_uid != os.getuid():
        raise ValueError("pickle must belong to the invoking user")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb", buffering=0) as raw:
        if file_identity(before) != file_identity(os.fstat(raw.fileno())):
            raise ValueError("input identity changed before opening")
        hashing = _HashingReader(raw)
        stream = io.BufferedReader(hashing)
        value = pickle.load(stream)
        if stream.read(1):
            raise ValueError("trailing bytes after native pickle")
        digest = hashing.digest.hexdigest()
        if file_identity(before) != file_identity(os.fstat(raw.fileno())):
            raise ValueError("input changed during extraction")
    if file_identity(before) != file_identity(path.lstat()):
        raise ValueError("input changed during extraction")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("input SHA-256 mismatch")
    return value, {"path": str(path.absolute()), **file_identity(before), "sha256": digest}


def light_values(values):
    selected = {key: values[key] for key in LIGHT_KEYS if key in values}
    episode = values.get("episode_metrics")
    if isinstance(episode, dict):
        selected["episode_metrics"] = {key: episode[key] for key in EPISODE_LIGHT_KEYS if key in episode}
    if len(canonical(selected)) > 16384:
        raise ValueError("selected light signals exceed 16 KiB; review the source contract")
    return selected


def completion_plan(state, units, completed_tag, completed_value, partition_id):
    if units < 1:
        raise ValueError("unit count must be positive")
    partition = state["partitions"][partition_id]
    if partition.revert_keys_mapping != {index: key for key, index in partition.keys_mapping.items()}:
        raise ValueError("controller key/index mapping is inconsistent")
    if len(set(partition.keys_mapping.values())) != len(partition.keys_mapping):
        raise ValueError("controller has duplicate row indexes")
    rows, statuses = [], Counter()
    for key, index in sorted(partition.keys_mapping.items(), key=lambda item: item[1]):
        tags = partition.custom_meta.get(index, {})
        statuses[str(tags.get(completed_tag))] += 1
        if tags.get(completed_tag) != completed_value:
            continue
        ready = [
            name
            for name, column in partition.field_name_mapping.items()
            if int(partition.production_status[index, column]) == 1
        ]
        if "response_mask" not in ready:
            raise ValueError(f"published row {key!r} lacks a produced response mask")
        if "input_ids" not in ready and not {"prompts", "responses"}.issubset(ready):
            raise ValueError(f"completed row {key!r} lacks produced token IDs")
        rows.append(
            {
                "key": key,
                "global_index": index,
                "unit_position": index % units,
                "produced_fields": ready,
                "signals": light_values(tags),
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "partition": partition_id,
        "units": units,
        "completion_contract": {"tag": completed_tag, "value": completed_value},
        "completed_published_trajectory_rows": len(rows),
        "status_counts": dict(statuses),
        "rows": rows,
        "episode_completeness": "Unknown: row publication is not an episode denominator",
        "not_a_restore_checkpoint": True,
    }


def terminal_sessions(partition):
    """Audit the explicit prompt/session contract; never infer completion from a highest row alone."""
    tags = {key: partition.custom_meta.get(index, {}) for key, index in partition.keys_mapping.items()}
    prompts = {key: value for key, value in tags.items() if value.get("is_prompt") is True}
    outputs = {}
    orphan_rows = []
    for key, value in tags.items():
        if key in prompts:
            continue
        parts = key.rsplit("_", 2)
        if len(parts) != 3 or parts[0] not in prompts:
            orphan_rows.append(key)
            continue
        uid, session, ordinal = parts
        if not session.isdecimal() or not ordinal.isdecimal() or key != f"{uid}_{int(session)}_{int(ordinal)}":
            raise ValueError("noncanonical session/output identity")
        outputs.setdefault(uid, {}).setdefault(int(session), {})[int(ordinal)] = key
    groups = []
    for uid, prompt in sorted(prompts.items()):
        expected = prompt.get("expected_session_count")
        if type(expected) is not int or expected <= 0:
            raise ValueError("prompt has no positive expected session count")
        failures = {}
        prompt_failures = []
        for failure in prompt.get("failure_records") or ():
            session = failure["session_id"]
            lean_failure = {
                key: failure[key] for key in ("session_id", "category", "stage", "error_class") if key in failure
            }
            lean_failure.update(light_values(failure))
            if session is None:
                prompt_failures.append(lean_failure)
                continue
            if type(session) is not int or not 0 <= session < expected or session in failures:
                raise ValueError("invalid or duplicate failed session identity")
            failures[session] = lean_failure
        sessions = outputs.get(uid, {})
        if any(session >= expected for session in sessions):
            raise ValueError("published session is outside enrollment")
        complete, unknown = [], []
        for session in range(expected):
            if session in failures and session not in sessions:
                continue
            rows = sessions.get(session, {})
            keys = [rows[index] for index in sorted(rows)]
            contiguous = bool(keys) and sorted(rows) == list(range(len(rows)))
            terminal = bool(keys) and isinstance(tags[keys[-1]].get("episode_metrics"), dict)
            if (
                session not in failures
                and contiguous
                and terminal
                and all(tags[key].get("status") == "success" for key in keys)
            ):
                complete.append({"session_id": session, "keys": keys})
            else:
                unknown.append(session)
        groups.append(
            {
                "uid": uid,
                "status": prompt.get("status"),
                "expected_sessions": expected,
                "published_sessions": complete,
                "failed_sessions": list(failures.values()),
                "prompt_failures": prompt_failures,
                "unknown_sessions": unknown,
                "metadata_complete": prompt.get("status") in ("finished", "failure")
                and not unknown
                and not prompt_failures,
            }
        )
    return {
        "contract": "prompt-session-terminal-v1",
        "groups": groups,
        "orphan_rows": sorted(orphan_rows),
        "all_episode_metadata_complete": bool(groups)
        and not orphan_rows
        and all(group["metadata_complete"] for group in groups),
        "limit": "Producer contract and storage readiness need qualification; metadata is not payload coverage.",
    }


def tensor_record(value):
    if type(value) is not torch.Tensor or value.device.type != "cpu" or value.is_nested:
        raise ValueError("archive token/mask/reward must be an ordinary CPU tensor")
    if value.layout != torch.strided or value.ndim != 1:
        raise ValueError("archive token/mask/reward must be a one-dimensional strided tensor")
    packed = value.detach().contiguous()
    raw = packed.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "encoding": "base64-native-bytes",
        "byteorder": sys.byteorder,
        "data": base64.b64encode(raw).decode(),
    }


def project_row(fields, planned):
    index = planned["global_index"]
    get = lambda name: fields.get(name, {}).get(index) if name in planned["produced_fields"] else None
    ids = get("input_ids")
    token_origin = "input_ids"
    if ids is None:
        prompts, responses = get("prompts"), get("responses")
        if type(prompts) is not torch.Tensor or type(responses) is not torch.Tensor:
            raise ValueError("planned token fields are absent from this storage unit")
        ids = torch.cat((prompts, responses))
        token_origin = "concatenated_native_prompts_responses"
    encoded_ids = tensor_record(ids)
    if ids.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        raise ValueError("native token IDs must have an integer dtype")
    response_mask, reward = get("response_mask"), get("rm_scores")
    if response_mask is None:
        raise ValueError("planned mask is absent from this storage unit")
    tensor_record(response_mask)
    if reward is not None:
        tensor_record(reward)
    if ("rm_scores" in planned["produced_fields"]) != (reward is not None):
        raise ValueError("reward presence differs from controller production metadata")
    prompt_length = planned["signals"].get("prompt_len")
    if prompt_length is None and type(get("prompts")) is torch.Tensor:
        prompt_length = get("prompts").numel()
    if type(prompt_length) is not int or prompt_length < 0:
        raise ValueError("native prompt length is missing or invalid")
    if ids.numel() != prompt_length + response_mask.numel() or (
        reward is not None and reward.numel() != response_mask.numel()
    ):
        raise ValueError("token/mask/reward alignment differs from native lengths")
    masks = {}
    for name in MASK_FIELDS:
        value = get(name)
        if name in planned["produced_fields"] and value is None:
            raise ValueError(f"produced {name} is absent from this storage unit")
        if value is not None:
            expected = ids.numel() if name == "attention_mask" else response_mask.numel()
            if value.numel() != expected:
                raise ValueError(f"{name} length differs from native tokens")
            masks[name] = tensor_record(value)
    identity = {name: get(name) for name in ("uid", "session_id", "task_id", "data_source")}
    if identity["uid"] is not None and identity["session_id"] is not None:
        prefix = f"{identity['uid']}_{identity['session_id']}_"
        if not planned["key"].startswith(prefix):
            raise ValueError("stored session identity differs from the controller key")
    extra = get("extra_fields")
    signals = dict(planned["signals"])
    if isinstance(extra, dict):
        signals.update(light_values(extra))
    return {
        "schema": RECORD_SCHEMA,
        "key": planned["key"],
        "global_index": index,
        "identity": identity,
        "prompt_length": prompt_length,
        "token_origin": token_origin,
        "input_ids": encoded_ids,
        "masks": masks,
        "reward": tensor_record(reward) if reward is not None else None,
        "reward_scope": "native_stored_rm_scores" if reward is not None else "absent_in_native_produced_fields",
        "signals": signals,
    }


def extract_unit(state, plan, position, stream):
    if plan["schema"] != PLAN_SCHEMA or not 0 <= position < plan["units"]:
        raise ValueError("invalid plan schema or unit position")
    selected = [row for row in plan["rows"] if row["unit_position"] == position]
    selected.sort(key=lambda row: row["global_index"])
    fields = state["field_data"]
    if state.get("storage_unit_id") != plan["storage_units"][position]["storage_unit_id"]:
        raise ValueError("storage unit identity differs from the controller plan manifest")
    if any(index % plan["units"] != position for values in fields.values() for index in values):
        raise ValueError("storage unit contains rows assigned to another unit position")
    keys_digest, records_digest = hashlib.sha256(), hashlib.sha256()
    for row in selected:
        record = canonical(project_row(fields, row)) + b"\n"
        stream.write(record)
        keys_digest.update(canonical(row["key"]) + b"\n")
        records_digest.update(record)
    return {
        "completed_rows": len(selected),
        "keys_sha256": keys_digest.hexdigest(),
        "uncompressed_records_sha256": records_digest.hexdigest(),
        "unit_position": position,
        "storage_unit_id": state.get("storage_unit_id"),
        "all_expected_unit_rows_exported": True,
    }


def read_json(path, expected_sha256):
    if Path(path).stat().st_size > 256 * 1024 * 1024:
        raise ValueError("JSON metadata exceeds 256 MiB; use a separately reviewed metadata plan")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("JSON input SHA-256 mismatch")
    return json.loads(raw)


def audit_index(plan, records):
    """An index must attest token/mask/reward presence; an identity alone is not coverage."""
    expected = {row["key"] for row in plan["rows"]}
    found, complete, unknown = set(), set(), set()
    for record in records:
        key = record["key"]
        if key in found:
            raise ValueError("duplicate source row key in export index")
        found.add(key)
        digest = record.get("source_record_sha256", "")
        if (
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest)
            and record.get("controller_sha256") == plan["controller"]["sha256"]
            and all(record.get(name) is True for name in ("token_ids_present", "masks_present", "reward_present"))
        ):
            complete.add(key)
        else:
            unknown.add(key)
    return {
        "expected_completed_rows": len(expected),
        "identity_matched_rows": len(expected & found),
        "index_attested_rows": len(expected & complete),
        "coverage_unknown_rows": len(expected & unknown),
        "missing_keys": sorted(expected - found),
        "extra_keys": sorted(found - expected),
        "all_expected_rows_attested": bool(expected) and expected.issubset(complete),
        "episode_coverage_proved": False,
        "limit": "Index assertions do not verify export bytes or completed episodes. No cleanup authorization.",
    }


def verify_archive(archive, plan, plan_sha256, manifest_sha256, max_bytes, max_record_bytes):
    """Independently decompress the finished artifact; never reopen its native pickle."""
    archive = Path(archive)
    if archive.is_symlink() or (archive / "INCOMPLETE").exists():
        raise ValueError("archive is a symlink or incomplete")
    manifest = read_json(archive / "manifest.json", manifest_sha256)
    if (
        plan["schema"] != PLAN_SCHEMA
        or manifest["schema"] != "transfer_queue.lean_unit_manifest.v1"
        or manifest["plan_sha256"] != plan_sha256
    ):
        raise ValueError("archive manifest/plan binding mismatch")
    position = manifest["unit_position"]
    if (
        type(position) is not int
        or not 0 <= position < plan["units"]
        or manifest["storage_unit_id"] != plan["storage_units"][position]["storage_unit_id"]
    ):
        raise ValueError("archive unit identity mismatch")
    if max_bytes <= 0 or max_record_bytes <= 0:
        raise ValueError("verification byte bounds must be positive")
    expected = sorted(
        (row for row in plan["rows"] if row["unit_position"] == position), key=lambda row: row["global_index"]
    )
    if len({row["key"] for row in expected}) != len(expected) or len({row["global_index"] for row in expected}) != len(
        expected
    ):
        raise ValueError("duplicate planned row identity")
    if manifest["completed_rows"] != len(expected):
        raise ValueError("manifest row count differs from plan")
    keys, records, count_bytes = hashlib.sha256(), hashlib.sha256(), 0
    path = archive / "completed.jsonl.gz"
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size != manifest["output_bytes"]:
        raise ValueError("archive is not a regular file with the declared size")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb", buffering=0) as raw:
        if file_identity(before) != file_identity(os.fstat(raw.fileno())):
            raise ValueError("archive changed before opening")
        hashing = _HashingReader(raw)
        with gzip.GzipFile(fileobj=io.BufferedReader(hashing), mode="rb") as zipped:
            for row in expected:
                line = zipped.readline(min(max_record_bytes, max_bytes - count_bytes) + 1)
                count_bytes += len(line)
                if len(line) > max_record_bytes or count_bytes > max_bytes or not line.endswith(b"\n"):
                    raise ValueError("archive row is missing, truncated or exceeds verification bounds")
                record = json.loads(line)
                if (
                    record["schema"] != RECORD_SCHEMA
                    or record["key"] != row["key"]
                    or record["global_index"] != row["global_index"]
                ):
                    raise ValueError("decompressed row identity differs from plan")
                keys.update(canonical(record["key"]) + b"\n")
                records.update(line)
            if zipped.read(1):
                raise ValueError("archive contains extra records")
        digest = hashing.digest.hexdigest()
        if file_identity(before) != file_identity(os.fstat(raw.fileno())):
            raise ValueError("archive changed during verification")
    if file_identity(before) != file_identity(path.lstat()) or (archive / "INCOMPLETE").exists():
        raise ValueError("archive changed during verification")
    if (
        digest != manifest["output_sha256"]
        or keys.hexdigest() != manifest["keys_sha256"]
        or records.hexdigest() != manifest["uncompressed_records_sha256"]
    ):
        raise ValueError("archive compressed/key/uncompressed-record digest mismatch")
    return {
        "status": "PASS_ARCHIVE_BYTES_AND_PLANNED_ROWS",
        "unit_position": position,
        "verified_rows": len(expected),
        "uncompressed_bytes": count_bytes,
        "output_sha256": digest,
        "keys_sha256": keys.hexdigest(),
        "uncompressed_records_sha256": records.hexdigest(),
        "plan_sha256": plan_sha256,
        "manifest_sha256": manifest_sha256,
        "episode_coverage_proved": False,
        "limits": "Verifies archived bytes and planned row identities, not original pickle equality, "
        "completed-episode denominator, numerical restore or cleanup authorization.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_cmd = commands.add_parser("plan", help="Project trusted controller metadata; reads no storage payload")
    plan_cmd.add_argument("--controller", required=True, type=Path)
    plan_cmd.add_argument("--controller-sha256", required=True)
    plan_cmd.add_argument("--max-controller-bytes", required=True, type=int)
    plan_cmd.add_argument("--unit-manifest", required=True, type=Path)
    plan_cmd.add_argument("--unit-manifest-sha256", required=True)
    plan_cmd.add_argument("--episode-contract", choices=["prompt-session-terminal-v1"])
    plan_cmd.add_argument("--partition", default="train")
    plan_cmd.add_argument("--completed-tag", required=True)
    plan_cmd.add_argument("--completed-value", required=True)
    plan_cmd.add_argument("--output", required=True, type=Path)
    extract = commands.add_parser(
        "extract", help="Load one trusted unit on allocated big-memory CPU; write lean evidence only"
    )
    extract.add_argument("--plan", required=True, type=Path)
    extract.add_argument("--plan-sha256", required=True)
    extract.add_argument("--unit", required=True, type=Path)
    extract.add_argument("--unit-sha256")
    extract.add_argument("--max-unit-bytes", required=True, type=int)
    extract.add_argument("--unit-position", required=True, type=int)
    extract.add_argument("--output", required=True, type=Path)
    audit = commands.add_parser(
        "audit-index", help="Join an authenticated lean export index to the completed-row denominator"
    )
    audit.add_argument("--plan", required=True, type=Path)
    audit.add_argument("--plan-sha256", required=True)
    audit.add_argument("--index", required=True, type=Path)
    audit.add_argument("--index-sha256", required=True)
    audit.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify", help="Independently decompress and verify a finished lean archive")
    verify.add_argument("--archive", required=True, type=Path)
    verify.add_argument("--manifest-sha256", required=True)
    verify.add_argument("--plan", required=True, type=Path)
    verify.add_argument("--plan-sha256", required=True)
    verify.add_argument("--max-uncompressed-bytes", required=True, type=int)
    verify.add_argument("--max-record-bytes", type=int, default=16 * 1024 * 1024)
    verify.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        state, source = load_pickle(args.controller, args.max_controller_bytes, args.controller_sha256)
        units = read_json(args.unit_manifest, args.unit_manifest_sha256)
        units.sort(key=lambda unit: unit["position"])
        if [unit["position"] for unit in units] != list(range(len(units))) or len(
            {unit["storage_unit_id"] for unit in units}
        ) != len(units):
            raise ValueError("unit manifest must contain unique IDs and contiguous positions")
        result = completion_plan(state, len(units), args.completed_tag, args.completed_value, args.partition)
        result["storage_units"] = units
        result["unit_manifest_sha256"] = args.unit_manifest_sha256
        result["controller"] = source
        if args.episode_contract:
            result["episodes"] = terminal_sessions(state["partitions"][args.partition])
        write_json(args.output, result)
    elif args.command == "audit-index":
        plan = read_json(args.plan, args.plan_sha256)
        raw = args.index.read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.index_sha256:
            raise ValueError("export index SHA-256 mismatch")
        result = audit_index(plan, (json.loads(line) for line in raw.splitlines()))
        write_json(args.output, result)
    elif args.command == "verify":
        plan = read_json(args.plan, args.plan_sha256)
        result = verify_archive(
            args.archive,
            plan,
            args.plan_sha256,
            args.manifest_sha256,
            args.max_uncompressed_bytes,
            args.max_record_bytes,
        )
        write_json(args.output, result)
    else:
        plan = read_json(args.plan, args.plan_sha256)
        args.output.mkdir(exist_ok=False)
        incomplete = args.output / "INCOMPLETE"
        incomplete.write_text("Only manifest.json with no INCOMPLETE marker denotes finished extraction.\n")
        state, source = load_pickle(args.unit, args.max_unit_bytes, args.unit_sha256)
        path = args.output / "completed.jsonl.gz"
        with path.open("xb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                result = extract_unit(state, plan, args.unit_position, zipped)
            raw.flush()
            os.fsync(raw.fileno())
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        result.update(
            schema="transfer_queue.lean_unit_manifest.v1",
            input=source,
            plan_sha256=args.plan_sha256,
            output_bytes=path.stat().st_size,
            output_sha256=digest.hexdigest(),
            not_a_restore_checkpoint=True,
            omitted=[
                "routed_experts",
                "full_metrics",
                "full_batch",
                "log_probs",
                "model_optimizer_RNG_loader_queue_state",
            ],
        )
        write_json(args.output / "manifest.json", result)
        incomplete.unlink()
        directory_fd = os.open(args.output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


if __name__ == "__main__":
    main()
