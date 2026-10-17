import multiprocessing
import os

import pytest
import torch

from transfer_queue.utils import common


@pytest.fixture(autouse=True)
def reset_thread_state(monkeypatch):
    before = torch.get_num_threads()
    monkeypatch.setattr(common, "_PHYSICAL_CPU_COUNT_CACHE", None)
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_repeated_contexts_cache_real_physical_detection(monkeypatch):
    detect = common.psutil.cpu_count
    calls = []

    def observed(*args, **kwargs):
        calls.append((os.getpid(), kwargs))
        return detect(*args, **kwargs)

    monkeypatch.setattr(common.psutil, "cpu_count", observed)
    torch.set_num_threads(2)
    for _ in range(5):
        with common.limit_pytorch_auto_parallel_threads(target_num_threads=1):
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == common.DEFAULT_TORCH_NUM_THREADS
    assert calls == [(os.getpid(), {"logical": False})]


@pytest.mark.parametrize("target", [None, 3])
def test_auto_and_clamped_thread_targets_are_preserved(monkeypatch, target):
    monkeypatch.setattr(common.psutil, "cpu_count", lambda logical: 2)
    with common.limit_pytorch_auto_parallel_threads(target_num_threads=target):
        assert torch.get_num_threads() == 2
    assert torch.get_num_threads() == common.DEFAULT_TORCH_NUM_THREADS


def test_body_exception_still_restores_default_threads():
    with pytest.raises(LookupError, match="body failed"):
        with common.limit_pytorch_auto_parallel_threads(target_num_threads=2):
            raise LookupError("body failed")
    assert torch.get_num_threads() == common.DEFAULT_TORCH_NUM_THREADS


def test_undetermined_count_is_not_cached(monkeypatch):
    detect = common.psutil.cpu_count
    calls = []

    def transient_none(*args, **kwargs):
        calls.append(1)
        return None if len(calls) == 1 else detect(*args, **kwargs)

    monkeypatch.setattr(common.psutil, "cpu_count", transient_none)
    with pytest.raises(TypeError):
        with common.limit_pytorch_auto_parallel_threads(target_num_threads=1):
            pass
    assert common._PHYSICAL_CPU_COUNT_CACHE is None
    with common.limit_pytorch_auto_parallel_threads(target_num_threads=1):
        assert torch.get_num_threads() == 1
    assert len(calls) == 2


def test_detector_exception_propagates_and_can_retry(monkeypatch):
    detect = common.psutil.cpu_count
    calls = []

    def transient_error(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("detector failed")
        return detect(*args, **kwargs)

    monkeypatch.setattr(common.psutil, "cpu_count", transient_error)
    before = torch.get_num_threads()
    with pytest.raises(OSError, match="detector failed"):
        with common.limit_pytorch_auto_parallel_threads(target_num_threads=1):
            pass
    assert torch.get_num_threads() == before
    assert common._PHYSICAL_CPU_COUNT_CACHE is None
    with common.limit_pytorch_auto_parallel_threads(target_num_threads=1):
        pass
    assert len(calls) == 2


def _fork_probe(connection, calls):
    with common.limit_pytorch_auto_parallel_threads(target_num_threads=1):
        inside = torch.get_num_threads()
    connection.send((os.getpid(), common._PHYSICAL_CPU_COUNT_CACHE[0], calls, inside, torch.get_num_threads()))
    connection.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="requires fork")
def test_forked_child_refreshes_parent_count(monkeypatch):
    detect = common.psutil.cpu_count
    calls = []

    def observed(*args, **kwargs):
        calls.append(os.getpid())
        return detect(*args, **kwargs)

    monkeypatch.setattr(common.psutil, "cpu_count", observed)
    with common.limit_pytorch_auto_parallel_threads(target_num_threads=1):
        pass
    parent = os.getpid()
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_fork_probe, args=(sender, calls))
    child.start()
    sender.close()
    try:
        assert receiver.poll(15), "forked helper did not finish"
        pid, cached_pid, child_calls, inside, restored = receiver.recv()
        child.join(15)
        assert child.exitcode == 0
        assert pid != parent and cached_pid == pid
        assert child_calls == [parent, pid]
        assert calls == [parent]
        assert inside == 1 and restored == common.DEFAULT_TORCH_NUM_THREADS
    finally:
        receiver.close()
        if child.is_alive():
            child.terminate()
            child.join(5)
