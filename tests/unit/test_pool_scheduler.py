import os
import signal
import sys
import threading
import time

import pytest

from grace_al.explore.backends import Job, Terminated, run_pool, slot_envs, visible_gpus

PY = sys.executable


def job(i, tmp_path, code):
    return Job(i, [PY, "-c", code], tmp_path / f"w{i}" / "walker.log")


# records start/end times and the slot env in the log
REC = ("import os, time; t = time.time(); time.sleep({s}); "
       "print(t, time.time(), os.environ.get('CUDA_VISIBLE_DEVICES', '-'), "
       "os.environ['TF_NUM_INTRAOP_THREADS'])")


def intervals(tmp_path, n):
    out = []
    for i in range(n):
        a, b, gpu, threads = (tmp_path / f"w{i}" / "walker.log").read_text().split()
        out.append((float(a), float(b), gpu, threads))
    return out


def test_concurrency_limit(tmp_path):
    jobs = [job(i, tmp_path, REC.format(s=0.3)) for i in range(5)]
    codes = run_pool(jobs, slot_envs([], 2, cpus=8), poll=0.02)
    assert codes == dict.fromkeys(range(5), 0)
    iv = intervals(tmp_path, 5)
    for t in [x[0] + 0.05 for x in iv]:
        assert sum(a <= t < b for a, b, *_ in iv) <= 2
    assert {x[3] for x in iv} == {"4"}                   # 8 cpus // 2 slots


def test_gpu_round_robin(tmp_path):
    envs = slot_envs(["0", "1"], 2, cpus=16)
    assert [e["CUDA_VISIBLE_DEVICES"] for e in envs] == ["0", "1", "0", "1"]
    assert all(e["TF_FORCE_GPU_ALLOW_GROWTH"] == "true" for e in envs)
    jobs = [job(i, tmp_path, REC.format(s=0.2)) for i in range(4)]
    run_pool(jobs, envs, poll=0.02)
    assert sorted(x[2] for x in intervals(tmp_path, 4)) == ["0", "0", "1", "1"]
    assert visible_gpus({"CUDA_VISIBLE_DEVICES": "2,3"}) == ["2", "3"]
    assert visible_gpus({"CUDA_VISIBLE_DEVICES": ""}) == []
    assert visible_gpus({}) == []


def test_failure_isolated(tmp_path):
    jobs = [job(0, tmp_path, "import sys; print('boom'); sys.exit(3)"),
            job(1, tmp_path, "print('ok')")]
    seen = {}
    codes = run_pool(jobs, slot_envs([], 1, 1), on_done=lambda j, rc: seen.update({j.id: rc}),
                     poll=0.02)
    assert codes == {0: 3, 1: 0} == seen
    assert "boom" in (tmp_path / "w0" / "walker.log").read_text()


def test_sigterm_terminates_children(tmp_path):
    jobs = [job(i, tmp_path, "import time; print('started', flush=True); time.sleep(30)")
            for i in range(3)]
    before = signal.getsignal(signal.SIGTERM)
    threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    t0 = time.time()
    with pytest.raises(Terminated, match="2 walkers stopped, 1 not started"):
        run_pool(jobs, slot_envs([], 2, 2), poll=0.02, grace=2)
    assert time.time() - t0 < 5
    assert not (tmp_path / "w2").exists()
    assert signal.getsignal(signal.SIGTERM) == before     # handler restored
