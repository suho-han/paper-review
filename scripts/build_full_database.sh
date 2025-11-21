#!/bin/bash
# ICLR & NeurIPS 2021-2025 데이터 수집 및 ChromaDB 생성 스크립트

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 로그 디렉토리 생성
mkdir -p logs data chromadb

# ICLR & NeurIPS 2021-2025 데이터 수집
VENUES=(
    "ICLR.cc/2021/Conference"
    "ICLR.cc/2022/Conference"
    "ICLR.cc/2023/Conference"
    "ICLR.cc/2024/Conference"
    "ICLR.cc/2025/Conference"
    "NeurIPS.cc/2021/Conference"
    "NeurIPS.cc/2022/Conference"
    "NeurIPS.cc/2023/Conference"
    "NeurIPS.cc/2024/Conference"
    "NeurIPS.cc/2025/Conference"
)

echo "=================================================="
echo "ICLR & NeurIPS 2021-2025 데이터 수집 시작"
echo "=================================================="

for venue in "${VENUES[@]}"; do
    # Extract conference name and year
    conf=$(echo $venue | cut -d'/' -f1 | cut -d'.' -f1)  # ICLR or NeurIPS
    year=$(echo $venue | grep -oP '\d{4}')
    conf_lower=$(echo "$conf" | tr '[:upper:]' '[:lower:]')
    log_file="logs/${conf_lower}_${year}_download.log"  # iclr_2021 or neurips_2021
    json_file="data/${venue//\//_}_reviews.json"  # ICLR.cc_2021_Conference_reviews.json
    
    # 이미 다운로드된 파일이 있으면 건너뛰기
    if [ -f "$json_file" ] && [ -s "$json_file" ]; then
        echo ""
        echo "[$conf $year] 이미 다운로드됨 - 건너뜀: $json_file"
        continue
    fi
    
    echo ""
    echo "[$conf $year] $venue 데이터 다운로드 중..."
    echo "로그: $log_file"
    
    uv run src/data/data_collection.py \
        --batch \
        --venue "$venue" \
        2>&1 | tee "$log_file"
    
    if [ $? -eq 0 ]; then
        echo "[$conf $year] ✓ 다운로드 완료"
        # 각 연도 다운로드 완료 시 알림
        uv run scripts/alert.py --repo "paper-review" --message "$conf ${year} 데이터 다운로드" 2>/dev/null || true
    else
        echo "[$conf $year] ✗ 다운로드 실패 (계속 진행)"
    fi
    
    # 각 연도 사이에 2분 대기 (rate limit 방지 - 1분에서 증가)
    echo "Rate limit 방지를 위해 120초 대기..."
    sleep 120
done

echo ""
echo "=================================================="
echo "데이터 다운로드 단계 완료"
echo "=================================================="

# 다운로드된 파일 목록 및 크기 확인
echo ""
echo "다운로드된 파일:"
ls -lh data/*.json

echo ""
echo "=================================================="
echo "ChromaDB 구축 시작"
echo "=================================================="

# 년도별 ChromaDB 컬렉션 생성
echo ""
echo "년도별 ChromaDB 컬렉션 생성 중..."
uv run src/vectordb/build_all_vectordb.py \
    2>&1 | tee logs/vectordb_build.log

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ 모든 작업 완료!"
    echo "=================================================="
    echo ""
    echo "생성된 컬렉션:"
    echo "  - iclr_2021, iclr_2022, iclr_2023, iclr_2024, iclr_2025"
    echo "  - neurips_2021, neurips_2022, neurips_2023, neurips_2024, neurips_2025"
    echo ""
    echo "Vector DB 테스트 예시:"
    echo "  uv run src/vectordb/build_vectordb.py --test --collection iclr_2025"
    echo "  uv run src/vectordb/build_vectordb.py --test --collection neurips_2024"
    echo ""
    echo "데이터 탐색:"
    echo "  jupyter notebook explore_chromadb.ipynb"
    
    # Slack 알림 전송
    echo ""
    echo "Slack 알림 전송 중..."
    uv run scripts/alert.py --repo "paper-review" --message "ICLR & NeurIPS 2021-2025 년도별 ChromaDB 구축 완료 (10개 컬렉션)" 2>/dev/null || true
else
    echo "✗ ChromaDB 생성 실패"
    # 실패 알림
    uv run scripts/alert.py --repo "paper-review" --message "ChromaDB 생성 실패" 2>/dev/null || true
    exit 1
fi
