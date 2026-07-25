"""Generate synthetic Android logcat and dmesg files covering all eval categories.

The capstone doc's public sample-log download URL is a placeholder and doesn't
resolve, so this generator is the primary data source for local development
until real AOSP/CTS bugreports are added as a stretch enhancement.

Usage:
    python scripts/generate_synthetic_logs.py
    python scripts/generate_synthetic_logs.py --output-dir data/raw --seed 42
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

# Each scenario produces either a logcat file, a dmesg file, or both.
# Keep messages consistent with the 10 eval categories in the capstone doc.

SCENARIOS = [
    {
        "id": "anr",
        "kind": "logcat",
        "lines": [
            "01-15 14:23:07.410  1234  1235 I ActivityManager: Displayed com.example.myapp/.MainActivity",
            "01-15 14:23:07.412  1234  1235 E ActivityManager: ANR in com.example.myapp",
            "01-15 14:23:07.413  1234  1235 E ActivityManager: Reason: Input dispatching timed out",
            "01-15 14:23:07.414  1234  1235 E ActivityManager: Load: 8.2 / 6.1 / 4.0",
            "01-15 14:23:07.415  1234  1235 W InputDispatcher: main thread not responding",
        ],
    },
    {
        "id": "crash",
        "kind": "logcat",
        "lines": [
            "01-15 15:01:33.880  9012  9012 I ActivityManager: Start proc com.example.myapp",
            "01-15 15:01:33.881  9012  9012 E AndroidRuntime: FATAL EXCEPTION: main",
            "01-15 15:01:33.882  9012  9012 E AndroidRuntime: java.lang.NullPointerException",
            "01-15 15:01:33.882  9012  9012 E AndroidRuntime:   at com.example.myapp.MainActivity.onCreate(MainActivity.java:42)",
        ],
    },
    {
        "id": "oom",
        "kind": "logcat",
        "lines": [
            "01-15 15:44:01.900  7654  7654 I dalvikvm-heap: Grow heap to 256MB",
            "01-15 15:44:02.120  7654  7654 E dalvikvm: Out of memory: Heap Size=128MB",
            "01-15 15:44:02.121  7654  7654 E AndroidRuntime: FATAL EXCEPTION: main",
            "01-15 15:44:02.121  7654  7654 E AndroidRuntime: java.lang.OutOfMemoryError",
        ],
    },
    {
        "id": "gpu_fault",
        "kind": "dmesg",
        "lines": [
            "[ 1204.880001] kgsl kgsl-3d0: |kgsl_pwrctrl_change_state| clock enabled",
            "[ 1204.883001] kgsl kgsl-3d0: GPU fault detected for context 45 ts=78912",
            "[ 1204.883006] kgsl kgsl-3d0: Device hanged on active context 45",
        ],
    },
    {
        "id": "oom_kill",
        "kind": "dmesg",
        "lines": [
            "[  823.441020] kswapd0: reclaim pages, low on free memory",
            "[  823.441022] lowmemorykiller: Killing 'com.example.myapp' (9012), adj 900",
            "[  823.441025] lowmemorykiller: to free 65536kB on behalf of 'kswapd0'",
        ],
    },
    {
        "id": "thermal",
        "kind": "logcat",
        "lines": [
            "01-15 16:00:00.900  500   500 I ThermalEngine: monitoring CPU temperature",
            "01-15 16:00:01.001  500   500 W ThermalEngine: CPU temperature 89C exceeds threshold",
            "01-15 16:00:01.002  500   500 I ThermalEngine: Throttling CPU to 1.2GHz",
        ],
    },
    {
        "id": "camera_crash",
        "kind": "logcat",
        "lines": [
            "01-15 16:10:05.100  2345  2345 I CameraDeviceImpl: opening camera device",
            "01-15 16:10:05.233  2345  2345 E CameraDeviceImpl: Camera device encountered fatal error",
            "01-15 16:10:05.234  2345  2345 E CameraDeviceImpl: Camera service died",
            "01-15 16:10:05.235  2345  2345 E AndroidRuntime: FATAL EXCEPTION: CameraThread",
        ],
    },
    {
        "id": "kernel_panic",
        "kind": "dmesg",
        "lines": [
            "[  823.441010] kworker/3:2: starting scheduled work",
            "[  823.441022] BUG: spinlock already unlocked on CPU#3, PID 1234",
            "[  823.441030] CPU: 3 PID: 1234 Comm: kworker/3:2",
            "[  823.441039] pc : spin_bug+0x38/0x50",
        ],
    },
    {
        "id": "binder_failure",
        "kind": "logcat",
        "lines": [
            "01-15 17:00:00.000  1234  1235 I ActivityManager: system_server calling binder",
            "01-15 17:00:00.001  1234  1235 E Binder: Failed reply: -32",
            "01-15 17:00:00.002  1234  1235 W BpBinder: Binder transaction failed (err=-32)",
        ],
    },
    {
        "id": "memory_leak",
        "kind": "logcat",
        "lines": [
            "01-15 17:29:55.000  8888  8888 I dalvikvm-heap: Grow heap to 128MB",
            "01-15 17:30:00.001  8888  8888 I dalvikvm-heap: Grow heap to 256MB",
            "01-15 17:30:05.001  8888  8888 I dalvikvm-heap: Grow heap to 512MB",
            "01-15 17:30:10.001  8888  8888 E AndroidRuntime: java.lang.OutOfMemoryError: Java heap space",
        ],
    },
]

_NOISE_LOGCAT = [
    "01-15 {ts}  600   601 I SurfaceFlinger: onMessageReceived",
    "01-15 {ts}  600   601 D Gralloc: buffer allocated",
    "01-15 {ts}  700   701 I MediaCodec: configure called",
    "01-15 {ts}  700   701 D WindowManager: relayoutWindow",
]

_NOISE_DMESG = [
    "[ {ts}] binder: transaction complete",
    "[ {ts}] cpu_cooling: normal operating temperature",
    "[ {ts}] wlan: scan complete",
]


def _make_noise_lines(kind: str, count: int, rng: random.Random) -> list[str]:
    templates = _NOISE_LOGCAT if kind == "logcat" else _NOISE_DMESG
    lines = []
    for _ in range(count):
        template = rng.choice(templates)
        if kind == "logcat":
            ts = f"{rng.randint(10, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d}"
        else:
            ts = f"{rng.randint(1, 2000)}.{rng.randint(0, 999999):06d}"
        lines.append(template.format(ts=ts))
    return lines


def generate(output_dir: Path, seed: int, noise_lines: int) -> list[Path]:
    """Generate one synthetic file per scenario, interleaved with noise lines.

    Args:
        output_dir: Base data/raw directory to write into.
        seed: Random seed for reproducible noise generation.
        noise_lines: Number of unrelated noise lines to mix in per file.

    Returns:
        List of paths written.
    """
    rng = random.Random(seed)
    logcat_dir = output_dir / "sample_logcats"
    dmesg_dir = output_dir / "sample_dmesgs"
    logcat_dir.mkdir(parents=True, exist_ok=True)
    dmesg_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for scenario in SCENARIOS:
        noise = _make_noise_lines(scenario["kind"], noise_lines, rng)
        all_lines = noise[: noise_lines // 2] + scenario["lines"] + noise[noise_lines // 2 :]

        target_dir = logcat_dir if scenario["kind"] == "logcat" else dmesg_dir
        out_path = target_dir / f"{scenario['id']}.txt"
        out_path.write_text("\n".join(all_lines) + "\n")
        written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw", help="Base output directory (default: data/raw)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise generation")
    parser.add_argument("--noise-lines", type=int, default=6, help="Unrelated noise lines to mix into each file")
    args = parser.parse_args()

    written = generate(Path(args.output_dir), seed=args.seed, noise_lines=args.noise_lines)
    print(f"Generated {len(written)} synthetic log files under {args.output_dir}/:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
