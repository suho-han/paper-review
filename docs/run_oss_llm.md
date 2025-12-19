# Reviewer Agent를 위한 OSS LLM 실행 가이드

이 문서는 Reviewer Agent에서 사용할 오픈소스 LLM(Llama 3, Qwen2 등)을 로컬 서버(vLLM)로 실행하고 연동하는 방법을 설명합니다.

## 1. 사전 준비 (Prerequisites)

* **GPU 서버**: CUDA를 지원하는 GPU (VRAM 24GB 이상 권장)
* **환경 설정**: Python 환경 및 필수 패키지 설치 (`requirements.txt`)
* **모델 다운로드**: HuggingFace 등에서 사용할 모델을 미리 다운로드하거나, 실행 시 자동 다운로드되도록 설정.

## 2. vLLM 서버 실행 (Start vLLM Server)

통합 파이썬 런처 `scripts/start_vllm.py` 를 사용하여 OpenAI 호환 API 서버를 실행합니다.

### 환경 변수 설정 (선택)

```bash
# 사용할 모델 경로 또는 HuggingFace ID
export VLLM_MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

# 서버 호스트 및 포트 설정
export VLLM_HOST="0.0.0.0"
export VLLM_PORT="8000"

# GPU 설정 (단일 GPU 사용 시 1)
export VLLM_TP_SIZE=1
export VLLM_GPU_MEM_UTIL=0.90
```

### 실행 방법

```bash
# 8B 서버 (포그라운드)
python scripts/start_vllm.py start --model meta-llama/Meta-Llama-3-8B-Instruct --port 8000 --cuda 3

# 1B 서버 (포그라운드)
python scripts/start_vllm.py start-1b

# 8B 서버 (포그라운드, 프리셋)
python scripts/start_vllm.py start-8b

# 1B + 8B 동시 실행 (백그라운드)
python scripts/start_vllm.py start-both
```

서버가 정상적으로 실행되면 `http://localhost:8000` 또는 `http://localhost:8001` 주소로 API 요청을 받을 준비가 됩니다.

## 3. Reviewer Agent 연동 설정 (Configuration)

Reviewer Agent가 로컬 vLLM 서버를 바라보도록 `.env` 파일을 설정하거나 환경 변수를 주입합니다.

프로젝트 루트의 `.env` 파일에 다음 내용을 추가하거나 수정하세요.

```bash
# .env 파일 예시

# vLLM 서버 주소 (OpenAI 호환 엔드포인트)
VLLM_API_BASE="http://localhost:8000/v1"

# 사용할 모델 이름 (서버 실행 시 지정한 모델명과 일치해야 함)
VLLM_MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

# (선택) Google Gemini API 키가 있어도 VLLM_API_BASE가 설정되어 있으면 vLLM을 우선 사용합니다.
```

## 4. 연동 확인 (Verification)

설정이 올바른지 확인하기 위해 간단한 테스트를 수행할 수 있습니다.

### Curl 테스트

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello! Are you ready to review papers?"}
    ]
  }'
```

### Python 스크립트 테스트

`src/agents/llm.py`가 정상적으로 vLLM을 가져오는지 확인합니다.

```python
# test_llm.py
from src.agents.llm import get_llm

llm = get_llm()
if llm:
    print(f"LLM initialized: {llm}")
    response = llm.invoke("Hello, introduce yourself.")
    print(response.content)
else:
    print("Failed to initialize LLM.")
```

## 5. Reviewer Agent 실행

이제 Reviewer Agent를 실행하면 로컬 LLM을 사용하여 논문 리뷰를 생성합니다.

```bash
# 예: Coordinator를 통해 전체 파이프라인 실행
python src/agents/coordinator.py
```

또는 개별 에이전트 테스트:

```bash
python src/agents/reviewer.py
```
