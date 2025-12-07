import argparse
import csv
import os
import subprocess
import time
from datetime import datetime


def get_gpu_stats(gpu_id=None):
    try:
        # Query nvidia-smi for utilization and memory
        cmd = ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"]
        if gpu_id is not None:
            cmd.extend(["-i", str(gpu_id)])

        result = subprocess.check_output(
            cmd,
            encoding="utf-8"
        )
        # If multiple GPUs are returned (shouldn't happen with -i but good to handle), take the first one or handle accordingly.
        # With -i, it returns just that GPU's stats.
        return result.strip().split(", ")
    except FileNotFoundError:
        # Mock for non-GPU envs or if nvidia-smi is missing
        return [datetime.now().isoformat(), "0", "0", "0", "0", "0"]
    except Exception as e:
        print(f"Error querying nvidia-smi: {e}")
        return [datetime.now().isoformat(), "-1", "-1", "-1", "-1", "-1"]


def monitor(interval=5, duration=60, output_file="logs/gpu_logs.csv", gpu_id=None):
    # Ensure logs directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Monitoring GPU {gpu_id if gpu_id is not None else 'all'} for {duration} seconds. Logging to {output_file}...")

    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "GPU Util (%)", "Mem Util (%)", "Mem Total (MB)", "Mem Free (MB)", "Mem Used (MB)"])

        start_time = time.time()
        while True:
            if duration > 0 and (time.time() - start_time > duration):
                break

            stats = get_gpu_stats(gpu_id)
            writer.writerow(stats)
            f.flush()
            time.sleep(interval)

    print(f"Monitoring finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU usage")
    parser.add_argument("--interval", type=int, default=5, help="Polling interval in seconds")
    parser.add_argument("--duration", type=int, default=60, help="Duration to monitor in seconds (0 for infinite)")
    parser.add_argument("--output", type=str, default="logs/gpu_logs.csv", help="Output CSV file")
    parser.add_argument("--gpu-id", type=int, default=3, help="GPU ID to monitor (default: 3)")
    args = parser.parse_args()

    monitor(args.interval, args.duration, args.output, args.gpu_id)
