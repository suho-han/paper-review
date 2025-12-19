# ReviewerAgent

OpenReview 인간 리뷰 데이터를 RAG와 LangGraph 기반 멀티 에이전트 파이프라인으로 연결한 온프레미스 논문 리뷰어 에이전트입니다. vLLM으로 1B/8B 모델을 자체 호스팅해 Streamlit 데모와 평가 스크립트를 제공합니다.

## Quick Start (Streamlit 데모)

```bash
uv pip install -r requirements.txt
uv run streamlit run run_demo.py
```

- 기본값: 1B(포트 8001, GPU mem util 0.5, max-len 4096) + 8B(포트 8000, GPU mem util 0.7, max-len 6144)를 자동 기동 후 UI를 엽니다.
- 서버를 직접 띄우고 싶다면 `DEMO_SKIP_LLM_START=1` 또는 `uv run streamlit run run_demo.py -- --skip-llm-start`를 사용하세요.
- HuggingFace 토큰이 필요하면 `HF_TOKEN`을 설정하세요.

### 수동 vLLM 기동 예시

```bash
# 1B
uv run python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --port 8001 --gpu-memory-utilization 0.5 --max-model-len 4096 --tensor-parallel-size 1

# 8B (예: GPU 2 사용)
CUDA_VISIBLE_DEVICES=2 uv run python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --port 8000 --gpu-memory-utilization 0.7 --max-model-len 6144 --tensor-parallel-size 1

# 데모는 LLM 자동 기동 없이 실행
DEMO_SKIP_LLM_START=1 uv run streamlit run run_demo.py
```

문제가 생기면 `uv run python scripts/kill_vllm.py`(또는 `bash scripts/kill_vllm.sh`)로 잔여 프로세스를 정리하세요.

## 데이터 파이프라인

1) **수집**

```bash
# 단일 논문 기본 수집
uv run src/data/data_collection.py

# 배치 다운로드 (예: ICLR 2025)
uv run src/data/data_collection.py --batch --venue ICLR.cc/2025/Conference
```

1) **벡터 DB 구축 (ChromaDB)**

```bash
# 전체 컬렉션 일괄 구축
uv run src/vectordb/build_all_vectordb.py

# 특정 JSON → 특정 컬렉션
uv run src/vectordb/build_vectordb.py --json_file data/ICLR.cc_2025_Conference_reviews.json --collection iclr_2025
```

1) **검색/테스트**

```bash
# 기본 테스트 (all_reviews 컬렉션)
uv run src/vectordb/build_vectordb.py --test

# 특정 컬렉션 테스트
uv run src/vectordb/build_vectordb.py --test --collection iclr_2025 --query "What are the main weaknesses in experimental design?"
```

1) **탐색/시각화**: `notebooks/test_chromadb.ipynb` 또는 `notebooks/test_agents.ipynb`로 컬렉션과 에이전트 동작을 점검하세요.

## 평가 파이프라인

- 테스트 세트: `data/evaluation/test_set_2025.json`
- 실행 스크립트: `src/evaluation/run_eval.py`, `src/evaluation/run_baseline_vs_agent.py`
- 주요 지표: 약점 재현율, 평점 상관계수, 환각 비율

### Baseline vs Agent 실험 실행

Baseline은 **zero-RAG(검색/ArXiv 없이 Reviewer+Rating만)**, Agent는 **CoordinatorAgent 전체 워크플로(RAG+ArXiv+Rating)** 를 실행해 지표를 비교합니다.

1) (권장) vLLM 서버 상태 확인

```bash
uv run python scripts/healthcheck_vllm.py
```

1) 실험 실행 (기본: 25개 샘플)

```bash
uv run python src/evaluation/run_baseline_vs_agent.py \
  --data data/evaluation/test_set_2025.json \
  --sample 25 \
  --output outputs/results
```

1) 전체 샘플에 대해 테스트 (0)

```bash
uv run python src/evaluation/run_baseline_vs_agent.py \
  --data data/evaluation/test_set_2025.json \
  --sample 0 \
  --output outputs/results
```

1) 출력 확인

- `--output`을 디렉터리로 주면, 아래 경로에 번들로 저장됩니다.
  - `outputs/results/baseline_vs_agent/<UTC_TIMESTAMP>/report.json`
  - `outputs/results/baseline_vs_agent/<UTC_TIMESTAMP>/summary.md`
  - `outputs/results/baseline_vs_agent/<UTC_TIMESTAMP>/details.csv`

옵션

- `--no-llm`: 약점 재현율/환각(LLM 기반) 지표를 생략하고 실행 (LLM 서버 없이도 동작)
- `--output <path.json>`: JSON 파일 경로를 주면 해당 위치에 `report.json` 저장 + 같은 폴더에 `summary.md`, `details.csv` 생성

## 프로젝트 구조 (요약)

```plaintext
src/agents/          # LangGraph 기반 멀티 에이전트 (parser, retriever, reviewer, rating 등)
src/data/            # OpenReview 수집 스크립트
src/vectordb/        # ChromaDB 빌드 및 테스트
src/evaluation/      # 정량 평가 스크립트
scripts/             # vLLM 관리, 헬스체크, 모니터링 유틸
notebooks/           # 에이전트/DB 실험용 노트북
data/, chromadb/     # 원본 데이터 및 벡터 DB 저장소
```

## 트러블슈팅

- vLLM 메모리 오류: GPU mem util을 더 낮추거나 `--max-model-len`을 줄이고, 잔여 프로세스를 종료합니다.
- 포트 충돌: `lsof -i :8000` 또는 `lsof -i :8001`로 점유 확인 후 종료합니다.
- 데모가 재시도만 반복: 브라우저 강제 새로고침(Ctrl+Shift+R)으로 Streamlit 세션 상태를 초기화합니다.

## 참고

- vLLM 기본 설정 및 최근 수정 사항은 `FIXES.md`를 확인하세요.
- LangGraph 파이프라인과 모델 실행 흐름은 `src/agents/`를 참고하세요.

### AI Review by [Stanford Agentic Reviwer](https://paperreview.ai/)

Go to [https://paperreview.ai/review](https://paperreview.ai/review) and paste the token below

```bash
zHXQic4byq9tputwLWpDBzCr070DvWTgPuizthLlPSc
```

## TODO List

### Next

- [ ] Reviewer/Rating 프롬프트 고도화 (few-shot 예제 보강, rubric 명시)
- [ ] RAG 품질 개선: 하이브리드 검색, Cross-Encoder 리랭킹 검토
- [ ] vLLM 운영 안정화: healthcheck/alert 임계값 재검토, 실패 재시도 정책 튜닝
