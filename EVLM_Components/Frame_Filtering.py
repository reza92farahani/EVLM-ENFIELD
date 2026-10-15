"""
filtering.py — Frame-filtering resource & energy measurement on NVIDIA Jetson.

This script benchmarks the three candidate frame-filtering strategies evaluated
in EVLM (Section V-E of the paper) — RGB histogram correlation, perceptual hash
(pHash), and vectorized Sum of Absolute Differences (SAD) — and measures their
onboard cost (power, CPU, GPU, RAM) on UAV-class edge hardware.

Measurement methodology (see Section V-G of the paper):
    Resource usage is sampled with NVIDIA `tegrastats`. Before running any
    filtering, the system is left idle to establish a baseline. The *active*
    cost of each method is then reported as (measured - idle), isolating the
    work done by the filter from the device's static power/compute draw.

Requirements:
    - An NVIDIA Jetson device (this script calls `sudo tegrastats`).
    - It will NOT run on a normal laptop/desktop. For a plain latency/throughput
      benchmark that runs anywhere, use `filtering_time.py` instead.

Input:
    A folder of already-clipped 4-second video segments (.mp4/.avi/.mkv), such
    as `Data/2D-Video/` (see the "Dataset" section of the repository README).
    Only a single sample clip is included in this repository; the full dataset
    is public but not redistributed here.

Output:
    A CSV (default `filtering_tegra.csv`) with per-video, per-method totals
    alongside idle baselines and the derived active values.

Usage (run from the repository root):
    sudo python EVLM_Components/Frame_Filtering.py --input_folder Data/2D-Video --csv_file filtering_tegra.csv
"""

import cv2
import time
import os
import numpy as np
import csv
import subprocess
import threading
import re
import argparse


# ---------------------------------------------------------
# 1. Tegrastats Logger
# ---------------------------------------------------------
class TegrastatsLogger:
    """Runs `tegrastats` in a background thread and parses its output.

    Each parsed sample is stored as (timestamp, metrics_dict). Call
    `get_interval_stats(start, end)` afterwards to average the samples that
    fall inside a given time window (e.g. one filtering run, or the idle phase).
    """

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.process = None
        self.running = False
        self.logs = []
        self.lock = threading.Lock()

        # Pre-compile the tegrastats field patterns once, so parsing each line
        # during logging stays cheap and doesn't inflate the CPU measurement.
        self.re_cpu = re.compile(r"CPU \[(.+?)\]")
        self.re_gpu = re.compile(r"GR3D_FREQ (\d+)%")
        self.re_ram = re.compile(r"RAM (\d+)/(\d+)MB")
        self.re_power_vdd = re.compile(r"VDD_IN (\d+)mW")
        self.re_power_pom = re.compile(r"POM_5V_IN (\d+)/")

    def start(self):
        """Launch tegrastats and begin collecting samples in a daemon thread."""
        self.logs = []
        self.running = True

        cmd = ["sudo", "tegrastats", "--interval", str(self.interval_ms)]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        time.sleep(1)  # Give tegrastats a moment to produce its first samples.

    def stop(self):
        """Terminate the tegrastats process cleanly (kill if it won't exit)."""
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
            except Exception:
                self.process.kill()

    def _reader_thread(self):
        """Continuously read tegrastats lines and extract CPU/GPU/RAM/power."""
        while self.running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line:
                continue

            now = time.time()
            data = {}

            # CPU: tegrastats reports per-core load, e.g. "CPU [12%,8%,...]".
            # We average across cores to a single utilization figure.
            cpu_match = self.re_cpu.search(line)
            if cpu_match:
                cores = re.findall(r"(\d+)%", cpu_match.group(1))
                data["cpu"] = np.mean([int(c) for c in cores]) if cores else 0
            else:
                data["cpu"] = 0

            # GPU: single "GR3D_FREQ NN%" field.
            gpu_match = self.re_gpu.search(line)
            data["gpu"] = int(gpu_match.group(1)) if gpu_match else 0

            # RAM: "RAM used/total MB" — we keep the used portion (MB).
            ram_match = self.re_ram.search(line)
            if ram_match:
                used = int(ram_match.group(1))
                data["ram"] = used
            else:
                # Fall back to 0 when the field is absent, mirroring CPU/GPU.
                # (Previously this referenced an undefined `used` and could
                # raise NameError on the first unparsed line.)
                data["ram"] = 0

            # Power: prefer VDD_IN; fall back to POM_5V_IN on boards that
            # expose that rail instead. Value is in milliwatts.
            power_match = self.re_power_vdd.search(line)
            if not power_match:
                power_match = self.re_power_pom.search(line)
            data["power"] = int(power_match.group(1)) if power_match else 0

            with self.lock:
                self.logs.append((now, data))

    def get_interval_stats(self, start_time, end_time):
        """Average all samples recorded within [start_time, end_time].

        Returns None if no samples fell inside the window (e.g. the run was
        shorter than the sampling interval).
        """
        relevant = []
        with self.lock:
            for ts, data in self.logs:
                if start_time <= ts <= end_time:
                    relevant.append(data)

        if not relevant:
            return None

        return {
            "cpu": np.mean([d["cpu"] for d in relevant]),
            "gpu": np.mean([d["gpu"] for d in relevant]),
            "ram": np.mean([d["ram"] for d in relevant]),
            "power": np.mean([d["power"] for d in relevant]),
            "count": len(relevant),
        }


# ---------------------------------------------------------
# 2. Frame-filtering strategies
# ---------------------------------------------------------
# Each function decodes a video and applies one redundancy-suppression rule,
# returning the number of frames processed. The retained-frame logic mirrors
# the strategies compared in Section V-E; here we only need to exercise the
# per-frame computation so its resource cost can be measured.

# ---------- pHash ----------
def get_phash_opencv(image, hash_size=8, highfreq_factor=4):
    """Compute a DCT-based perceptual hash of an image (OpenCV implementation).

    The image is reduced to a small grayscale square, transformed with the DCT,
    and the low-frequency block is thresholded against its median to yield a
    compact binary fingerprint that is robust to minor visual changes.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    dct_lowfreq = dct[:hash_size, :hash_size]
    med = np.median(dct_lowfreq)
    return dct_lowfreq > med


def run_phash(video_path):
    """Retain a frame only if its perceptual hash has not been seen before."""
    cap = cv2.VideoCapture(video_path)
    seen = []
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h = get_phash_opencv(frame)
        is_dup = any(np.array_equal(h, x) for x in seen)
        if not is_dup:
            seen.append(h)
        frames += 1
    cap.release()
    return frames


# ---------- Histogram (EVLM default) ----------
def run_histogram(video_path, threshold=0.95):
    """Retain a frame when its color distribution differs enough from the last.

    A 3D RGB histogram (8x8x8 = 512 bins) is computed per frame and
    L2-normalized for scale invariance. A frame becomes a new key frame when
    its correlation with the previously retained frame drops below `threshold`
    (tau = 0.95 in the paper). This is EVLM's default filtering strategy.
    """
    cap = cv2.VideoCapture(video_path)
    prev = None
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hist = cv2.calcHist([frame], [0, 1, 2], None,
                            [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev is None or cv2.compareHist(hist, prev, cv2.HISTCMP_CORREL) < threshold:
            prev = hist
        frames += 1
    cap.release()
    return frames


# ---------- Vectorized SAD ----------
def run_sad(video_path, threshold=8000):
    """Retain a frame when its Sum of Absolute Differences vs. the last is high.

    The difference is computed on a downscaled grayscale image using a single
    vectorized `cv2.absdiff` (executed in optimized C++) rather than Python-level
    block loops, which keeps the per-frame cost low. A frame is retained when the
    total absolute difference exceeds `threshold` (tau_sad = 8000 in the paper).
    """
    cap = cv2.VideoCapture(video_path)
    prev = None
    frames = 0

    # Downscaling to a fixed width keeps the comparison cheap; 256 px is ample
    # for detecting near-duplicate frames while cutting CPU cost further.
    process_width = 256

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize while roughly preserving aspect ratio.
        h, w = gray.shape
        scale = process_width / float(w)
        small_gray = cv2.resize(gray, (process_width, int(h * scale)))

        if prev is None:
            prev = small_gray
        else:
            diff = cv2.absdiff(small_gray, prev)  # Vectorized (C++) difference.
            sad_total = np.sum(diff)
            if sad_total > threshold:
                prev = small_gray

        frames += 1

    cap.release()
    return frames


# ---------------------------------------------------------
# 3. Main measurement loop
# ---------------------------------------------------------
def main(input_folder, csv_file, tegra_interval, idle_duration):
    # Write the CSV header once, on first run.
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Video", "Method", "Duration(s)", "FPS",
                "Total_Power(mW)", "Idle_Power(mW)", "Active_Power(mW)",
                "Active_Energy(mWh)",
                "Total_CPU(%)", "Idle_CPU(%)", "Active_CPU(%)",
                "Total_GPU(%)", "Idle_GPU(%)", "Active_GPU(%)",
                "Total_RAM(%)", "Idle_RAM(%)", "Active_RAM(%)",
            ])

    video_files = [f for f in os.listdir(input_folder)
                   if f.endswith((".mp4", ".avi", ".mkv"))]

    if not video_files:
        print(f"Error: no video segments found in '{input_folder}'. "
              f"See the repository README's Dataset section for how to obtain "
              f"and place the clips.")
        return

    logger = TegrastatsLogger(interval_ms=tegra_interval)

    try:
        print("Starting Tegrastats Logger...")
        logger.start()

        # --- Idle baseline phase ---
        # Measure the device at rest so each method's *active* cost can be
        # isolated as (measured - idle).
        print(f"\n[System] Measuring IDLE baseline for {idle_duration} seconds...")
        idle_start = time.time()
        time.sleep(idle_duration)
        idle_end = time.time()

        idle_stats = logger.get_interval_stats(idle_start, idle_end)
        if not idle_stats:
            print("Error: Idle logging failed.")
            return

        print(f"[System] Baseline Power: {idle_stats['power']:.0f} mW")
        print(f"[System] Baseline CPU: {idle_stats['cpu']:.1f}%")

        # --- Per-video, per-method runs ---
        for video_file in video_files:
            video_path = os.path.join(input_folder, video_file)

            pipelines = {
                "pHash": run_phash,
                "Histogram": run_histogram,
                "VectorizedSAD": run_sad,
            }

            for method_name, method_func in pipelines.items():
                print(f"\n>> Running {method_name} on {video_file}...")

                start_t = time.time()
                frames = method_func(video_path)
                end_t = time.time()
                duration = end_t - start_t

                stats = logger.get_interval_stats(start_t, end_t)

                if stats and duration > 0:
                    fps = frames / duration

                    # Active values = measured during the run minus idle baseline.
                    total_power = stats["power"]
                    idle_power = idle_stats["power"]
                    active_power = max(0, total_power - idle_power)
                    # Energy (mWh) = active power (W) x duration (h), scaled back to mWh.
                    active_energy_mwh = (active_power / 1000.0) * (duration / 3600.0) * 1000.0

                    total_cpu = stats["cpu"]
                    idle_cpu = idle_stats["cpu"]
                    active_cpu = max(0, total_cpu - idle_cpu)

                    total_gpu = stats["gpu"]
                    idle_gpu = idle_stats["gpu"]
                    active_gpu = max(0, total_gpu - idle_gpu)

                    total_ram = stats["ram"]
                    idle_ram = idle_stats["ram"]
                    active_ram = max(0, total_ram - idle_ram)

                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            video_file, method_name,
                            f"{duration:.2f}", f"{fps:.2f}",
                            f"{total_power:.0f}", f"{idle_power:.0f}", f"{active_power:.0f}",
                            f"{active_energy_mwh:.4f}",
                            f"{total_cpu:.1f}", f"{idle_cpu:.1f}", f"{active_cpu:.1f}",
                            f"{total_gpu:.1f}", f"{idle_gpu:.1f}", f"{active_gpu:.1f}",
                            f"{total_ram:.1f}", f"{idle_ram:.1f}", f"{active_ram:.1f}",
                        ])

                    print(f"   Done: {fps:.1f} FPS | Active Power: {active_power:.0f} mW")
                else:
                    print("   Warning: Experiment too short or logging failed.")

    finally:
        print("\nStopping Tegrastats...")
        logger.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure frame-filtering resource/energy cost on NVIDIA Jetson."
    )
    parser.add_argument("--input_folder", default="Data/2D-Video",
                        help="Folder of 4-second video segments (default: Data/2D-Video, "
                             "resolved relative to the current working directory — run "
                             "this script from the repository root).")
    parser.add_argument("--csv_file", default="filtering_tegra.csv",
                        help="Output CSV path (default: filtering_tegra.csv).")
    parser.add_argument("--tegra_interval", type=int, default=1,
                        help="tegrastats sampling interval in ms (default: 1).")
    parser.add_argument("--idle_duration", type=int, default=10,
                        help="Idle baseline measurement duration in seconds (default: 10).")
    args = parser.parse_args()

    main(args.input_folder, args.csv_file, args.tegra_interval, args.idle_duration)
