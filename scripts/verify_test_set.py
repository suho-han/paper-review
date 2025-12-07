
import json
import os
import sys


def verify_test_set():
    test_set_path = 'data/evaluation/test_set_2025.json'
    if not os.path.exists(test_set_path):
        print(f"Test set not found at {test_set_path}")
        return

    with open(test_set_path, 'r') as f:
        test_set = json.load(f)

    print(f"Test set size: {len(test_set)}")

    if len(test_set) > 0:
        sample = test_set[0]
        print("Sample keys:", sample.keys())
        print("Sample forum_id:", sample.get('forum_id'))
        print("Has abstract:", bool(sample.get('abstract')))
        print("Has reviews:", bool(sample.get('reviews')))
        if sample.get('reviews'):
            print("Review count:", len(sample['reviews']))


if __name__ == "__main__":
    verify_test_set()
