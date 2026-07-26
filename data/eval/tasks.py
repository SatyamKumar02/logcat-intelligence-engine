"""10 labeled diagnostic tasks for the eval harness.

One task per root-cause category the system covers — matches the categories
scripts/generate_synthetic_logs.py produces and the seed cases in
data/processed/seed_cases.jsonl. Each task ships a self-contained log
snippet (not a file path) so the eval harness can run without depending on
files in data/raw/, which are gitignored and may not exist on a fresh clone.
"""

EVAL_TASKS = [
    {
        "id": "eval_001",
        "description": "App freezes and shows ANR dialog after 5 seconds of use",
        "logcat_snippet": (
            "01-15 14:23:07.412  1234  1235 E ActivityManager: ANR in com.example.myapp\n"
            "01-15 14:23:07.413  1234  1235 E ActivityManager: Reason: Input dispatching timed out\n"
            "01-15 14:23:07.414  1234  1235 E ActivityManager: Load: 8.2 / 6.1 / 4.0\n"
        ),
        "expected_root_cause_category": "anr",
        "expected_keywords": ["ANR", "dispatching timed out", "main thread"],
    },
    {
        "id": "eval_002",
        "description": "App crashes immediately on launch with NullPointerException",
        "logcat_snippet": (
            "01-15 15:01:33.881  9012  9012 E AndroidRuntime: FATAL EXCEPTION: main\n"
            "01-15 15:01:33.882  9012  9012 E AndroidRuntime: java.lang.NullPointerException\n"
            "01-15 15:01:33.882  9012  9012 E AndroidRuntime:   at com.example.myapp.MainActivity.onCreate\n"
        ),
        "expected_root_cause_category": "crash",
        "expected_keywords": ["NullPointerException", "FATAL EXCEPTION", "onCreate"],
    },
    {
        "id": "eval_003",
        "description": "Device low on memory, multiple apps killed in background",
        "logcat_snippet": (
            "01-15 15:44:02.120  7654  7654 E dalvikvm: Out of memory: Heap Size=128MB\n"
            "01-15 15:44:02.121  7654  7654 E AndroidRuntime: java.lang.OutOfMemoryError\n"
        ),
        "expected_root_cause_category": "oom",
        "expected_keywords": ["OutOfMemoryError", "memory", "heap"],
    },
    {
        "id": "eval_004",
        "description": "Device rebooted during 3D game play, no app crash seen",
        "dmesg_snippet": (
            "[ 1204.883001] kgsl kgsl-3d0: GPU fault detected for context 45 ts=78912\n"
            "[ 1204.883006] kgsl kgsl-3d0: Device hanged on active context 45\n"
        ),
        "expected_root_cause_category": "gpu_fault",
        "expected_keywords": ["GPU fault", "kgsl", "context"],
    },
    {
        "id": "eval_005",
        "description": "Background app silently killed, user loses unsaved data",
        "dmesg_snippet": (
            "[  823.441022] lowmemorykiller: Killing 'com.example.myapp' (9012), adj 900\n"
            "[  823.441025] lowmemorykiller: to free 65536kB on behalf of 'kswapd0'\n"
        ),
        "expected_root_cause_category": "oom_kill",
        "expected_keywords": ["lowmemorykiller", "adj", "kswapd"],
    },
    {
        "id": "eval_006",
        "description": "Device overheating warning, CPU running slow during video encode",
        "logcat_snippet": (
            "01-15 16:00:01.001  500   500 W ThermalEngine: CPU temperature 89C exceeds threshold\n"
            "01-15 16:00:01.002  500   500 I ThermalEngine: Throttling CPU to 1.2GHz\n"
        ),
        "expected_root_cause_category": "thermal",
        "expected_keywords": ["thermal", "throttling", "temperature"],
    },
    {
        "id": "eval_007",
        "description": "Camera app crashes with CameraDevice.StateCallback error",
        "logcat_snippet": (
            "01-15 16:10:05.233  2345  2345 E CameraDeviceImpl: Camera device encountered fatal error\n"
            "01-15 16:10:05.234  2345  2345 E CameraDeviceImpl: Camera service died\n"
            "01-15 16:10:05.235  2345  2345 E AndroidRuntime: FATAL EXCEPTION: CameraThread\n"
        ),
        "expected_root_cause_category": "camera_crash",
        "expected_keywords": ["CameraDevice", "fatal", "Camera service"],
    },
    {
        "id": "eval_008",
        "description": "Kernel panic with BUG() in spinlock, device hard rebooted",
        "dmesg_snippet": (
            "[  823.441022] BUG: spinlock already unlocked on CPU#3, PID 1234\n"
            "[  823.441030] CPU: 3 PID: 1234 Comm: kworker/3:2\n"
            "[  823.441039] pc : spin_bug+0x38/0x50\n"
        ),
        "expected_root_cause_category": "kernel_panic",
        "expected_keywords": ["BUG", "spinlock", "kernel"],
    },
    {
        "id": "eval_009",
        "description": "Binder transaction failed, system service unresponsive",
        "logcat_snippet": (
            "01-15 17:00:00.001  1234  1235 E Binder: Failed reply: -32\n"
            "01-15 17:00:00.002  1234  1235 W BpBinder: Binder transaction failed (err=-32)\n"
        ),
        "expected_root_cause_category": "binder_failure",
        "expected_keywords": ["Binder", "transaction failed", "reply"],
    },
    {
        "id": "eval_010",
        "description": "App uses excessive memory, heap grows to 512MB before OOM",
        "logcat_snippet": (
            "01-15 17:30:00.001  8888  8888 I dalvikvm-heap: Grow heap to 256MB\n"
            "01-15 17:30:05.001  8888  8888 I dalvikvm-heap: Grow heap to 512MB\n"
            "01-15 17:30:10.001  8888  8888 E AndroidRuntime: java.lang.OutOfMemoryError: Java heap space\n"
        ),
        "expected_root_cause_category": "memory_leak",
        "expected_keywords": ["heap", "Grow", "OutOfMemoryError"],
    },
]
