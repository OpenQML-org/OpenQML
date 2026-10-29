"""Windowed Linux RSS estimates, excluding inherited or earlier high-water marks.

Kernel RSS counters are approximate. This is not device memory or the sum of
descendant processes. Unsupported systems report no peak rather than substituting
ru_maxrss, which is a lifetime metric and can include launcher history.
See https://docs.kernel.org/filesystems/proc.html for clear_refs(5) semantics.
"""

from pathlib import Path


def _status():
    entries = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"VmRSS", "VmHWM"}:
            size, unit = value.split()
            if unit != "kB":
                raise ValueError("Unexpected procfs RSS units.")
            entries[name] = int(size) * 1024
    if set(entries) != {"VmRSS", "VmHWM"}:
        raise ValueError("procfs RSS counters are unavailable.")
    return entries


def measure_peak(function):
    try:
        Path("/proc/self/clear_refs").write_text("5\n")
        before = _status()["VmRSS"]
    except (OSError, ValueError) as exc:
        return function(), {"status": "unavailable", "reason": str(exc), "peak_rss_bytes": None}
    result = function()
    peak = _status()["VmHWM"]
    return result, {
        "status": "measured",
        "metric": "linux_procfs_reset_hwm",
        "scope": "current_process_call_window",
        "approximate": True,
        "start_rss_bytes": before,
        "peak_rss_bytes": peak,
        "peak_over_start_bytes": max(0, peak - before),
    }
