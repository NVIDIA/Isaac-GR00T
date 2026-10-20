"""Log the unit-test container's resource limits and CPU throttling counters."""

import argparse
import json
import os
from pathlib import Path
import sys


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def resource_snapshot(
    cgroup_file=Path("/proc/self/cgroup"), mountinfo_file=Path("/proc/self/mountinfo")
):
    memberships = {}
    for line in (_read(cgroup_file) or "").splitlines():
        _, controllers, path = line.split(":", 2)
        for controller in controllers.split(","):
            memberships[controller] = Path(path)
    groups = []
    quotas, memory_limits = [], []
    for line in (_read(mountinfo_file) or "").splitlines():
        before, after = line.split(" - ", 1)
        fields, options = before.split(), after.split()
        if options[0] not in ("cgroup", "cgroup2"):
            continue
        controllers = [""] if options[0] == "cgroup2" else options[2].split(",")
        member = next((memberships[c] for c in controllers if c in memberships), None)
        if member is None:
            continue
        root, mount = Path(fields[3]), Path(fields[4])
        try:
            relative = member.relative_to(root)
        except ValueError:
            # A private cgroup namespace reports its own root as /.
            if member != Path("/"):
                continue
            relative = Path(".")
        current = mount / relative
        while True:
            values = {}
            for filename in (
                "cpu.max",
                "cpu.cfs_quota_us",
                "cpu.cfs_period_us",
                "cpu.stat",
                "memory.max",
                "memory.limit_in_bytes",
                "memory.current",
                "memory.usage_in_bytes",
            ):
                value = _read(current / filename)
                if value is not None:
                    values[filename] = value
            if values:
                groups.append({"path": str(current), **values})
            quota = values.get("cpu.max", "").split()
            if quota and quota[0] != "max":
                quotas.append(int(quota[0]) / int(quota[1]))
            elif "cpu.cfs_quota_us" in values and int(values["cpu.cfs_quota_us"]) > 0:
                quotas.append(int(values["cpu.cfs_quota_us"]) / int(values["cpu.cfs_period_us"]))
            for filename in ("memory.max", "memory.limit_in_bytes"):
                value = values.get(filename)
                if value is not None and value != "max" and int(value) < 2**60:
                    memory_limits.append(int(value))
            if current == mount:
                break
            current = current.parent
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {
        "host_logical_cpus": os.cpu_count(),
        "affinity_cpus": affinity,
        "cpu_quota": min(quotas) if quotas else None,
        "memory_limit_bytes": min(memory_limits) if memory_limits else None,
        "cgroups": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    sys.stdout.write(
        "UNIT_TEST_RESOURCES=" + json.dumps({"phase": args.phase, **resource_snapshot()}) + "\n"
    )


if __name__ == "__main__":
    main()
