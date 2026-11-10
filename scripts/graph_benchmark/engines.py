"""Run-owned container lifecycle for the benchmark. Never touches resources it did not create."""

import json
import re
import secrets
import subprocess
import threading
import time
import uuid

ARCADE_IMAGES = {
    "26.12.1-SNAPSHOT": "arcadedata/arcadedb@sha256:586c622a0dbc48a50b30263b3255bd58e35cf4015727402c168f1d3263b932e4",
    "26.10.1": "arcadedata/arcadedb@sha256:9d2ddf7c42f52c54465edb2d4dbd73081fd8f13372e3fad12ffe97574b472980",
    "26.9.1": "arcadedata/arcadedb@sha256:a17d750ea214ae9dadf87d225e354b13f670475522128c21be55e94d6907fb97",
}
FALKOR_IMAGE = "falkordb/falkordb-server@sha256:b009c1df78ebc8acae7cdedeea635974fc83c3748f5e1591aecf7721ceb6f740"
LIMITS = ["--platform", "linux/amd64", "--cpus", "4", "--memory", "16g", "--memory-swap", "16g"]
READY_DEADLINE_S = 120


def sh(*args, check=True, timeout=300):
    return subprocess.run(args, check=check, capture_output=True, text=True, timeout=timeout)


_UNITS = {"B": 1, "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "KB": 1000, "MB": 1000**2, "GB": 1000**3}


def _parse_bytes(s):
    m = re.match(r"([\d.]+)\s*([A-Za-z]+)", s.strip())
    return int(float(m.group(1)) * _UNITS[m.group(2).upper()]) if m else None


class Engine:
    """One fresh container + Docker-managed volume. Credentials live only in memory."""

    def __init__(self, kind, version=None, tag="run"):
        self.kind = kind
        self.version = version
        self.run_id = uuid.uuid4().hex[:8]
        self.name = f"gb_{kind}_{tag}_{self.run_id}"
        self.volume = self.name + "_vol"
        self.password = secrets.token_urlsafe(18)
        self.port = None
        self.samples = []
        self._stop = threading.Event()
        self._thr = None
        self.image = ARCADE_IMAGES[version] if kind == "arcadedb" else FALKOR_IMAGE

    # lifecycle -------------------------------------------------------
    def start(self):
        sh("docker", "volume", "create", self.volume)
        cmd = ["docker", "run", "-d", "--name", self.name, *LIMITS, "-v", f"{self.volume}:{self._data_dir()}"]
        if self.kind == "arcadedb":
            cmd += ["-e", f"ARCADEDB_SETTINGS=-Darcadedb.server.rootPassword={self.password}", "-p", "127.0.0.1::2480"]
        else:
            cmd += ["-e", f"REDIS_ARGS=--requirepass {self.password}", "-e", "FALKORDB_ARGS=THREAD_COUNT 4",
                    "-p", "127.0.0.1::6379"]
        sh(*cmd, self.image)
        cport = "2480/tcp" if self.kind == "arcadedb" else "6379/tcp"
        self.port = int(sh("docker", "port", self.name, cport).stdout.splitlines()[0].rsplit(":", 1)[1])
        return self

    def _data_dir(self):
        return "/home/arcadedb/databases" if self.kind == "arcadedb" else "/var/lib/falkordb/data"

    def restart(self, kill=False):
        sh("docker", "kill" if kill else "stop", self.name)
        sh("docker", "start", self.name)
        cport = "2480/tcp" if self.kind == "arcadedb" else "6379/tcp"
        self.port = int(sh("docker", "port", self.name, cport).stdout.splitlines()[0].rsplit(":", 1)[1])

    def stop(self):
        self.stop_sampler()
        sh("docker", "rm", "-f", "-v", self.name, check=False)
        sh("docker", "volume", "rm", "-f", self.volume, check=False)

    def logs(self):
        r = sh("docker", "logs", self.name, check=False)
        return self._scrub((r.stdout or "") + (r.stderr or ""))

    def _scrub(self, text):
        return text.replace(self.password, "<redacted>")

    def inspect(self):
        return json.loads(sh("docker", "inspect", self.name).stdout)[0]

    # resource sampling (1 Hz, cgroup usage via docker stats) --------------
    def start_sampler(self):
        self._stop.clear()
        self.samples = []

        def loop():
            while not self._stop.is_set():
                t0 = time.time()
                r = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}|{{.CPUPerc}}", self.name, check=False)
                try:
                    mem, cpu = r.stdout.strip().split("|")
                    self.samples.append({"t": t0, "mem_bytes": _parse_bytes(mem.split("/")[0]),
                                         "cpu_pct": float(cpu.strip("%"))})
                except Exception:
                    pass
                self._stop.wait(max(0.0, 1.0 - (time.time() - t0)))

        self._thr = threading.Thread(target=loop, daemon=True)
        self._thr.start()

    def stop_sampler(self):
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=5)
            self._thr = None

    def resource_summary(self):
        mem = [s["mem_bytes"] for s in self.samples if s["mem_bytes"] is not None]
        return {"sample_interval_s": 1, "n_samples": len(mem), "peak_container_mem_bytes": max(mem) if mem else None,
                "note": "docker stats MemUsage = cgroup usage minus inactive file cache; not pure RSS",
                "volume_bytes": self.volume_bytes()}

    def volume_bytes(self):
        r = sh("docker", "exec", self.name, "du", "-sb", self._data_dir(), check=False)
        try:
            return int(r.stdout.split()[0])
        except Exception:
            return None

    def persistence_settings(self):
        if self.kind == "falkordb":
            r = sh("docker", "exec", self.name, "redis-cli", "-a", self.password, "--no-auth-warning", "CONFIG", "GET", "save", check=False)
            a = sh("docker", "exec", self.name, "redis-cli", "-a", self.password, "--no-auth-warning", "CONFIG", "GET", "appendonly", check=False)
            return {"save": r.stdout.split(), "appendonly": a.stdout.split()}
        return {"note": "ArcadeDB default WAL/flush policy (server defaults, not overridden)"}
