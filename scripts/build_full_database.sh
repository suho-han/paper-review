#!/bin/bash
# ICLR, NeurIPS, ICML 2021-2025 데이터 수집 및 ChromaDB 생성 스크립트

set -e  # 에러 발생 시 중단

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 로그 디렉토리 생성
mkdir -p logs data chromadb

# ICLR, NeurIPS, ICML 2021-2025 (+ TMLR) 데이터 수집
ALL_VENUES=(
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
    "ICML.cc/2021/Conference"
    "ICML.cc/2022/Conference"
    "ICML.cc/2023/Conference"
    "ICML.cc/2024/Conference"
    "ICML.cc/2025/Conference"
    "TMLR"
)

venue_key() {
    local venue="$1"
    if [ "$venue" == "TMLR" ]; then
        echo "tmlr"
        return
    fi
    local conf
    conf=$(echo "$venue" | cut -d'/' -f1 | cut -d'.' -f1)
    local year
    year=$(echo "$venue" | grep -oP '\d{4}')
    conf=$(echo "$conf" | tr '[:upper:]' '[:lower:]')
    echo "${conf}_${year}"
}

AVAILABLE_TARGETS=()
for venue in "${ALL_VENUES[@]}"; do
    AVAILABLE_TARGETS+=("$(venue_key "$venue")")
done

TARGET=$1
VENUES=()

if [ -n "$TARGET" ]; then
    for venue in "${ALL_VENUES[@]}"; do
        key=$(venue_key "$venue")
        if [ "$key" == "$TARGET" ]; then
            VENUES+=("$venue")
        fi
    done

    if [ ${#VENUES[@]} -eq 0 ]; then
        echo "Error: Unknown target '$TARGET'. Available targets: ${AVAILABLE_TARGETS[*]}"
        exit 1
    fi
else
    VENUES=("${ALL_VENUES[@]}")
fi

echo "=================================================="
echo "ICLR, NeurIPS, ICML 2021-2025 데이터 수집 및 DB 구축 시작"
echo "Target Venues: ${VENUES[*]}"
echo "=================================================="

for venue in "${VENUES[@]}"; do
    # Extract conference name and year
    if [ "$venue" == "TMLR" ]; then
        conf="TMLR"
        year=""
        conf_lower="tmlr"
        collection_name="tmlr"
    else
        conf=$(echo $venue | cut -d'/' -f1 | cut -d'.' -f1)  # ICLR, NeurIPS, 또는 ICML
        year=$(echo $venue | grep -oP '\d{4}')
        conf_lower=$(echo "$conf" | tr '[:upper:]' '[:lower:]')
        collection_name="${conf_lower}_${year}"
    fi
    
    log_file="logs/${collection_name}_download.log"
    # 파일명에서 .cc 제거
    if [ "$venue" == "TMLR" ]; then
        json_file="data/TMLR_reviews.json"
    else
        # 파일명에서 .cc 제거
        temp_venue="${venue//.cc/}"
        json_file="data/${temp_venue//\//_}_reviews.json"
    fi
    
    # 1. 데이터 수집
    # 이미 다운로드된 파일이 있으면 건너뛰기
    if [ -f "$json_file" ] && [ -s "$json_file" ]; then
        echo ""
        echo "[$conf $year] 이미 다운로드됨 - 건너뜀: $json_file"
    else
        echo ""
        if [ "$venue" == "TMLR" ]; then
            echo "[$conf] $venue 데이터 다운로드 중..."
        else
            echo "[$conf $year] $venue 데이터 다운로드 중..."
        fi
        echo "로그: $log_file"
        
        uv run src/data/data_collection.py \
            --batch \
            --venue "$venue" \
            2>&1 | tee "$log_file"
        
        if [ $? -eq 0 ]; then
            if [ "$venue" == "TMLR" ]; then
                echo "[$conf] ✓ 다운로드 완료"
            else
                echo "[$conf $year] ✓ 다운로드 완료"
            fi
            # 각 연도 다운로드 완료 시 알림
            if [ "$venue" == "TMLR" ]; then
                uv run scripts/alert.py --repo "paper-review" --message "$conf 데이터 다운로드" 2>/dev/null || true
            else
                uv run scripts/alert.py --repo "paper-review" --message "$conf ${year} 데이터 다운로드" 2>/dev/null || true
            fi
        else
            if [ "$venue" == "TMLR" ]; then
                echo "[$conf] ✗ 다운로드 실패 (계속 진행)"
            else
                echo "[$conf $year] ✗ 다운로드 실패 (계속 진행)"
            fi
        fi
        
        # 각 연도 사이에 대기 (rate limit 방지) - 여러 개일 때만
        if [ "${#VENUES[@]}" -gt 1 ]; then
            echo "Rate limit 방지를 위해 120초 대기..."
            sleep 120
        fi
    fi

    # 2. ChromaDB 생성
    echo ""
    if [ "$venue" == "TMLR" ]; then
        echo "[$conf] ChromaDB 컬렉션 생성 중: $collection_name"
    else
        echo "[$conf $year] ChromaDB 컬렉션 생성 중: $collection_name"
    fi
    build_log="logs/${collection_name}_vectordb_build.log"
    
    uv run src/vectordb/build_vectordb.py \
        --json_file "$json_file" \
        --collection "$collection_name" \
        2>&1 | tee "$build_log"

    if [ $? -eq 0 ]; then
        if [ "$venue" == "TMLR" ]; then
            echo "[$conf] ✓ ChromaDB 생성 완료"
        else
            echo "[$conf $year] ✓ ChromaDB 생성 완료"
        fi
    else
        if [ "$venue" == "TMLR" ]; then
            echo "[$conf] ✗ ChromaDB 생성 실패"
        else
            echo "[$conf $year] ✗ ChromaDB 생성 실패"
        fi
    fi
done

echo ""
echo "=================================================="
echo "모든 작업 완료"
echo "=================================================="

# 다운로드된 파일 목록 및 크기 확인
echo ""
echo "다운로드된 파일:"
ls -lh data/*.json

echo ""
echo "Vector DB 테스트 예시:"
echo "  uv run src/vectordb/build_vectordb.py --test --collection iclr_2025"
echo "  uv run src/vectordb/build_vectordb.py --test --collection neurips_2024"
echo "  uv run src/vectordb/build_vectordb.py --test --collection icml_2024"
echo ""
echo "데이터 탐색:"
echo "  jupyter notebook explore_chromadb.ipynb"

# Slack 알림 전송
echo ""
echo "Slack 알림 전송 중..."
uv run scripts/alert.py --repo "paper-review" --message "작업 완료: ${VENUES[*]}" 2>/dev/null || true
