import json
import os
import sys

import autorootcwd

from src.vectordb.build_vectordb import build_vectordb_from_json


def build_separate_conferences():
    """
    ICLR과 NeurIPS를 년도별로 각각 별도의 컬렉션으로 구축합니다.
    """
    conferences = [
        # ICLR 년도별
        ("data/ICLR.cc_2021_Conference_reviews.json", "iclr_2021"),
        ("data/ICLR.cc_2022_Conference_reviews.json", "iclr_2022"),
        ("data/ICLR.cc_2023_Conference_reviews.json", "iclr_2023"),
        ("data/ICLR.cc_2024_Conference_reviews.json", "iclr_2024"),
        ("data/ICLR.cc_2025_Conference_reviews.json", "iclr_2025"),
        # NeurIPS 년도별
        ("data/NeurIPS.cc_2021_Conference_reviews.json", "neurips_2021"),
        ("data/NeurIPS.cc_2022_Conference_reviews.json", "neurips_2022"),
        ("data/NeurIPS.cc_2023_Conference_reviews.json", "neurips_2023"),
        ("data/NeurIPS.cc_2024_Conference_reviews.json", "neurips_2024"),
        ("data/NeurIPS.cc_2025_Conference_reviews.json", "neurips_2025"),
    ]

    for json_file, collection_name in conferences:
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping...")
            continue

        print("\n" + "=" * 80)
        print(f"Building {collection_name}...")
        print("=" * 80)
        build_vectordb_from_json(json_file, collection_name=collection_name)

    print("\n" + "=" * 80)
    print("모든 년도별 컬렉션 구축 완료!")
    print("=" * 80)


if __name__ == '__main__':

    build_separate_conferences()
