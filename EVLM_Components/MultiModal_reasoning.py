import argparse
import json
import time
from pathlib import Path
import os
import csv
import threading
import subprocess
import re
import numpy as np

# --- ML Imports ---
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
TEGRA_INTERVAL_MS = 1
IDLE_DURATION = 10 
CSV_METRICS_FILE = "VLM_tegra.csv"
# ==========================================

# ---------------------------------------------------------
# 1. Tegrastats Logger (Throttling-Aware)
# ---------------------------------------------------------
class TegrastatsLogger:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.process = None
        self.running = False
        self.logs = []
        self.lock = threading.Lock()

    def start(self):
        self.logs = []
        self.running = True
        cmd = ["sudo", "tegrastats", "--interval", str(self.interval_ms)]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        time.sleep(1) # Allow warm-up

    def stop(self):
        self.running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=1)
            except:
                self.process.kill()

    def _reader_thread(self):
        # Updated Regex to capture both % and @MHz
        re_cpu = re.compile(r"CPU \[(.+?)\]")
        re_gpu = re.compile(r"GR3D_FREQ (\d+)%(?:@(\d+))?")
        re_emc = re.compile(r"EMC_FREQ (\d+)%(?:@(\d+))?")
        re_ram = re.compile(r"RAM (\d+)/(\d+)MB")
        re_power_vdd = re.compile(r"VDD_IN (\d+)mW")
        re_power_pom = re.compile(r"POM_5V_IN (\d+)/")

        while self.running and self.process.poll() is None:
            line = self.process.stdout.readline()
            if not line: continue
            
            now = time.time()
            data = {}

            # --- CPU (% and MHz) ---
            cpu_match = re_cpu.search(line)
            if cpu_match:
                # Extracts pairs of (percent, freq) like [('14', '1190'), ('11', '1190')]
                cores = re.findall(r"(\d+)%@(\d+)", cpu_match.group(1))
                if cores:
                    data['cpu_pct'] = np.mean([int(c[0]) for c in cores])
                    data['cpu_freq'] = np.mean([int(c[1]) for c in cores])
                else:
                    data['cpu_pct'], data['cpu_freq'] = 0, 0
            else:
                data['cpu_pct'], data['cpu_freq'] = 0, 0

            # --- GPU (% and MHz) ---
            gpu_match = re_gpu.search(line)
            if gpu_match:
                data['gpu_pct'] = int(gpu_match.group(1))
                data['gpu_freq'] = int(gpu_match.group(2)) if gpu_match.group(2) else 0
            else:
                data['gpu_pct'], data['gpu_freq'] = 0, 0

            # --- EMC (% and MHz) ---
            emc_match = re_emc.search(line)
            if emc_match:
                data['emc_pct'] = int(emc_match.group(1))
                data['emc_freq'] = int(emc_match.group(2)) if emc_match.group(2) else 0
            else:
                data['emc_pct'], data['emc_freq'] = 0, 0

            # --- RAM (%) ---
            ram_match = re_ram.search(line)
            if ram_match:
                used, total = int(ram_match.group(1)), int(ram_match.group(2))
                data['ram_pct'] = (used / total) * 100
            else:
                data['ram_pct'] = 0

            # --- Power (mW) ---
            p_match = re_power_vdd.search(line)
            if not p_match: 
                p_match = re_power_pom.search(line)
            data['power'] = int(p_match.group(1)) if p_match else 0

            with self.lock:
                self.logs.append((now, data))

    def get_interval_stats(self, start_time, end_time):
        relevant = []
        with self.lock:
            for ts, data in self.logs:
                if start_time <= ts <= end_time:
                    relevant.append(data)
        
        if not relevant: return None
        
        return {
            'power': np.mean([d['power'] for d in relevant]),
            'cpu_pct': np.mean([d['cpu_pct'] for d in relevant]),
            'cpu_freq': np.mean([d['cpu_freq'] for d in relevant]),
            'gpu_pct': np.mean([d['gpu_pct'] for d in relevant]),
            'gpu_freq': np.mean([d['gpu_freq'] for d in relevant]),
            'emc_pct': np.mean([d['emc_pct'] for d in relevant]),
            'emc_freq': np.mean([d['emc_freq'] for d in relevant]),
            'ram_pct': np.mean([d['ram_pct'] for d in relevant])
        }

# ---------------------------------------------------------
# 2. Local AI Model Analyzer (Single-Image)
# ---------------------------------------------------------
class LocalVLMAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval() 

    def analyze_frames(self, frame_paths, prompt_text):
        results = []
        for idx, frame_path in enumerate(frame_paths):
            try:
                image = Image.open(frame_path).convert("RGB")
                start_req = time.perf_counter()

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are an expert infrastructure analyst. Answer the questions with a clear, precise short explanations."}]},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
                ]
                
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                answer = generated_text.split("Assistant:")[-1].strip() if "Assistant:" in generated_text else generated_text.strip()
                
                total_time = time.perf_counter() - start_req

                results.append({
                    'frame': frame_path.name,
                    'inference_time': total_time,
                    'result': answer
                })

            except Exception as e:
                print(f"     X Error on {frame_path.name}: {e}")
                
        return results

# ---------------------------------------------------------
# 3. Main Measurement Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="unique_frames", help="Folder containing subfolders of images")
    parser.add_argument("--prompts_csv", type=str, default="prompts.csv", help="CSV with 'prompt' column")
    parser.add_argument("--output_dir", type=str, default="vlm_results")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize CSV Header (Now includes Frequencies)
    if not os.path.isfile(CSV_METRICS_FILE):
        with open(CSV_METRICS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Folder", "Prompt_ID", "Frames", "Duration(s)", "FPS",
                "Total_Power(mW)", "Idle_Power(mW)", "Active_Power(mW)", "Energy(mWh)",
                "Total_CPU(%)", "Idle_CPU(%)", "Active_CPU(%)", "Avg_CPU_Freq(MHz)",
                "Total_GPU(%)", "Idle_GPU(%)", "Active_GPU(%)", "Avg_GPU_Freq(MHz)",
                "Total_EMC(%)", "Idle_EMC(%)", "Active_EMC(%)", "Avg_EMC_Freq(MHz)",
                "Total_RAM(%)", "Idle_RAM(%)", "Active_RAM(%)"
            ])

    # Load Prompts
    prompts = []
    if os.path.exists(args.prompts_csv):
        with open(args.prompts_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            prompts = [row['prompt'] for row in reader]
    else:
        print(f"Warning: {args.prompts_csv} not found. Using a default prompt.")
        prompts = ["Describe this image in detail."]

    logger = TegrastatsLogger(TEGRA_INTERVAL_MS)
    
    try:
        print("\nStarting Tegrastats Monitor...")
        logger.start()

        # --- PHASE 1: Measure Model Loading ---
        print(f"\n[System] Loading AI Model into Memory (Measuring Payload)...")
        t_load_start = time.time()
        
        analyzer = LocalVLMAnalyzer() # Triggers the heavy loading
        
        t_load_end = time.time()
        load_duration = t_load_end - t_load_start
        load_stats = logger.get_interval_stats(t_load_start, t_load_end)

        print(f"[System] Model Loaded in {load_duration:.1f} seconds.")
        if load_stats:
            print(f"   Avg Load Power: {load_stats['power']:.0f} mW")
            print(f"   Avg RAM Spike:  {load_stats['ram_pct']:.1f} %")
            
            # Log Model Loading as a special row in the CSV
            with open(CSV_METRICS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "MODEL_LOADING", "N/A", "N/A", f"{load_duration:.2f}", "N/A",
                    f"{load_stats['power']:.0f}", "N/A", "N/A", "N/A",
                    f"{load_stats['cpu_pct']:.1f}", "N/A", "N/A", f"{load_stats['cpu_freq']:.0f}",
                    f"{load_stats['gpu_pct']:.1f}", "N/A", "N/A", f"{load_stats['gpu_freq']:.0f}",
                    f"{load_stats['emc_pct']:.1f}", "N/A", "N/A", f"{load_stats['emc_freq']:.0f}",
                    f"{load_stats['ram_pct']:.1f}", "N/A", "N/A"
                ])

        # --- PHASE 2: Measure Idle Baseline (Post-Load) ---
        print(f"\n[System] Measuring Standby IDLE baseline for {IDLE_DURATION}s...")
        t_start = time.time()
        time.sleep(IDLE_DURATION)
        t_end = time.time()
        idle_stats = logger.get_interval_stats(t_start, t_end)
        
        print(f"[System] Standby Baseline Established:")
        print(f"   Power: {idle_stats['power']:.0f} mW")
        print(f"   RAM:   {idle_stats['ram_pct']:.1f} % (Model footprint)")

        # --- PHASE 3: Run Experiments ---
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            return

        for folder in results_dir.iterdir():
            if not folder.is_dir(): continue
            
            frames = sorted([p for p in folder.iterdir() if p.suffix.lower() in ['.jpg','.png','.jpeg']])
            if not frames: continue

            print(f"\n--- Processing Folder: {folder.name} ---")

            for p_idx, prompt in enumerate(prompts):
                print(f"   Prompt {p_idx+1}: {prompt[:40]}...")

                exp_start = time.time()
                json_data = analyzer.analyze_frames(frames, prompt)
                exp_end = time.time()
                
                duration = exp_end - exp_start
                
                # Save JSON
                out_name = Path(args.output_dir) / f"{folder.name}_prompt{p_idx+1}.json"
                with open(out_name, 'w') as f:
                    json.dump(json_data, f, indent=2)

                # Extract Metrics
                stats = logger.get_interval_stats(exp_start, exp_end)
                
                if stats and duration > 0:
                    fps = len(frames) / duration
                    
                    # Calculations
                    total_p = stats['power']
                    active_p = max(0, total_p - idle_stats['power'])
                    energy = (active_p / 1000.0) * (duration / 3600.0) * 1000.0 
                    
                    active_cpu = max(0, stats['cpu_pct'] - idle_stats['cpu_pct'])
                    active_gpu = max(0, stats['gpu_pct'] - idle_stats['gpu_pct'])
                    active_emc = max(0, stats['emc_pct'] - idle_stats['emc_pct'])
                    active_ram = max(0, stats['ram_pct'] - idle_stats['ram_pct'])
                    
                    # Write to CSV
                    with open(CSV_METRICS_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            folder.name, p_idx, len(frames),
                            f"{duration:.2f}", f"{fps:.2f}",
                            
                            f"{total_p:.0f}", f"{idle_stats['power']:.0f}", f"{active_p:.0f}", f"{energy:.2f}",
                            f"{stats['cpu_pct']:.1f}", f"{idle_stats['cpu_pct']:.1f}", f"{active_cpu:.1f}", f"{stats['cpu_freq']:.0f}",
                            f"{stats['gpu_pct']:.1f}", f"{idle_stats['gpu_pct']:.1f}", f"{active_gpu:.1f}", f"{stats['gpu_freq']:.0f}",
                            f"{stats['emc_pct']:.1f}", f"{idle_stats['emc_pct']:.1f}", f"{active_emc:.1f}", f"{stats['emc_freq']:.0f}",
                            f"{stats['ram_pct']:.1f}", f"{idle_stats['ram_pct']:.1f}", f"{active_ram:.1f}"
                        ])
                        
                    print(f"     -> Done: {fps:.2f} FPS | Power: {active_p:.0f}mW | GPU: {active_gpu:.1f}%@{stats['gpu_freq']:.0f}MHz")

    finally:
        logger.stop()
        print("\nMonitor stopped.")

if __name__ == "__main__":
    main()
