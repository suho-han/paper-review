import argparse
import sys
import time

import requests


def check_health(url, retries=3, delay=2):
    health_url = f"{url}/health"
    for i in range(retries):
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                print(f"Health check passed: {response.status_code}")
                return True
            else:
                print(f"Health check failed: {response.status_code}")
        except Exception as e:
            print(f"Health check attempt {i+1} failed: {e}")

        if i < retries - 1:
            time.sleep(delay)

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Health check vLLM Server")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="vLLM server base URL")
    args = parser.parse_args()

    if check_health(args.base_url):
        sys.exit(0)
    else:
        sys.exit(1)
