#!/usr/bin/env python3
"""Kill all running vLLM server processes."""

import os
import signal
import subprocess
import sys
import time


def kill_vllm_processes():
    """Find and kill all vLLM server processes."""
    print("Killing all vLLM processes...")

    # Find processes running vllm.entrypoints.openai.api_server
    try:
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            pids = [pid for pid in pids if pid]

            if not pids:
                print("No vLLM processes found.")
                return

            print(f"Found {len(pids)} vLLM process(es): {', '.join(pids)}")

            # Send SIGTERM first (graceful shutdown)
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"Sent SIGTERM to process {pid}")
                except ProcessLookupError:
                    print(f"Process {pid} already terminated")
                except Exception as e:
                    print(f"Error terminating process {pid}: {e}")

            # Wait for processes to terminate
            print("Waiting 3 seconds for graceful shutdown...")
            time.sleep(3)

            # Check if any processes are still running
            result = subprocess.run(
                ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                remaining_pids = result.stdout.strip().split('\n')
                remaining_pids = [pid for pid in remaining_pids if pid]

                if remaining_pids:
                    print(f"Force killing {len(remaining_pids)} remaining process(es)...")
                    for pid in remaining_pids:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                            print(f"Sent SIGKILL to process {pid}")
                        except ProcessLookupError:
                            pass
                        except Exception as e:
                            print(f"Error force-killing process {pid}: {e}")

            print("All vLLM processes terminated.")
        else:
            print("No vLLM processes found.")

    except FileNotFoundError:
        print("Error: 'pgrep' command not found. Using fallback method...")
        # Fallback: use pkill
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])
        time.sleep(2)
        subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints.openai.api_server"])
        print("All vLLM processes terminated (fallback method).")


if __name__ == "__main__":
    kill_vllm_processes()
    print("\nYou can now restart the servers.")
