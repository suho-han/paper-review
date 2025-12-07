# ReviewerAgent

OpenReview 데이터를 활용한 온프레미스(On-Premise) 기반 AI 논문 리뷰어 Agent 프로젝트입니다.

## 목차

- [ReviewerAgent](#revieweragent)
  - [목차](#목차)
  - [설치 방법 (Installation)](#설치-방법-installation)
  - [사용 방법 (Usage)](#사용-방법-usage)
    - [1. 데이터 수집 (Data Collection)](#1-데이터-수집-data-collection)
    - [2. Vector DB 구축 (Build Vector DB)](#2-vector-db-구축-build-vector-db)
    - [3. Vector DB 탐색 및 시각화](#3-vector-db-탐색-및-시각화)
    - [4. Vector DB 테스트 (Test Vector DB)](#4-vector-db-테스트-test-vector-db)
    - [5. 디버깅 도구 (Debug Tools)](#5-디버깅-도구-debug-tools)
  - [전체 파이프라인 실행 예시](#전체-파이프라인-실행-예시)
    - [단일 년도 (ICLR 2025)](#단일-년도-iclr-2025)
    - [전체 년도 (ICLR + NeurIPS 2021-2025)](#전체-년도-iclr--neurips-2021-2025)
  - [프로젝트 구조](#프로젝트-구조)
  - [프로젝트 기획](#프로젝트-기획)
    - [프로젝트 목표](#프로젝트-목표)
    - [해결하고자 하는 문제](#해결하고자-하는-문제)
    - [기대 효과](#기대-효과)
  - [KPI (핵심 성능 지표)](#kpi-핵심-성능-지표)
  - [데이터 구조](#데이터-구조)
    - [Vector DB 컬렉션](#vector-db-컬렉션)
    - [문서 구조](#문서-구조)
  - [다음 단계](#다음-단계)
  - [라이선스](#라이선스)
  - [기여](#기여)

## 설치 방법 (Installation)

필요한 Python 패키지를 설치합니다.

```bash
uv pip install -r requirements.txt
```

## 사용 방법 (Usage)

### 1. 데이터 수집 (Data Collection)

OpenReview에서 논문 및 리뷰 데이터를 수집합니다.

```bash
# 단일 논문 수집 (기본 설정)
uv run src/data/data_collection.py

# 전체 학회 데이터 배치 다운로드 (예: ICLR 2025)
uv run src/data/data_collection.py --batch --venue ICLR.cc/2025/Conference
```

다운로드된 데이터는 `data/` 디렉토리에 JSON 형식으로 저장됩니다.

### 2. Vector DB 구축 (Build Vector DB)

수집한 데이터를 바탕으로 ChromaDB Vector Database를 구축합니다.

```bash
# 모든 년도별 컬렉션 일괄 구축 (권장)
uv run src/vectordb/build_all_vectordb.py

# 개별 컬렉션 구축 (예: ICLR 2021)
uv run src/vectordb/build_vectordb.py --json_file data/ICLR.cc_2021_Conference_reviews.json --collection iclr_2021
```

Vector DB는 `./chromadb/` 디렉토리에 영구 저장됩니다.

### 3. Vector DB 탐색 및 시각화

Jupyter 노트북을 사용하여 ChromaDB의 내용을 탐색하고 시각화할 수 있습니다.

```bash
# Jupyter 노트북 실행
jupyter notebook explore_chromadb.ipynb
```

노트북 주요 기능:

- **컬렉션 목록 확인**: 생성된 모든 컬렉션 조회
- **데이터 통계**: 문서 타입별 분포, 논문당 리뷰 수, 문서 길이 분석
- **시각화**: 파이 차트, 히스토그램, 박스 플롯으로 데이터 분포 확인
- **유사도 검색**: 다양한 쿼리로 RAG 검색 테스트
- **특정 논문 조회**: 논문의 abstract와 모든 리뷰 확인
- **필터링 검색**: 타입별(abstract/review) 검색

### 4. Vector DB 테스트 (Test Vector DB)

구축된 Vector DB에서 유사도 검색을 테스트합니다.

```bash
# 기본 쿼리로 테스트 (all_reviews 컬렉션)
uv run src/vectordb/build_vectordb.py --test

# 특정 컬렉션 및 커스텀 쿼리로 테스트
uv run src/vectordb/build_vectordb.py --test --collection iclr_2025 --query "What are the main weaknesses in experimental design?"

# NeurIPS 2024 컬렉션 테스트
uv run src/vectordb/build_vectordb.py --test --collection neurips_2024 --query "What are common weaknesses in this research?"
```

### 5. 디버깅 도구 (Debug Tools)

Forum ID와 Note ID의 관계를 확인하고 디버깅합니다.

```bash
# Forum의 모든 Note ID 리스트 확인
uv run scripts/debug_openreview.py --forum_id odjMSBSWRt --note_id MtBx6vnXOc
```

## 전체 파이프라인 실행 예시

### 단일 년도 (ICLR 2025)

```bash
# 1단계: ICLR 2025 데이터 다운로드
uv run src/data/data_collection.py --batch --venue ICLR.cc/2025/Conference

# 2단계: Vector DB 구축
uv run src/vectordb/build_vectordb.py --json_file data/ICLR.cc_2025_Conference_reviews.json --collection iclr_2025

# 3단계: Vector DB 테스트
uv run src/vectordb/build_vectordb.py --test --collection iclr_2025 --query "What are common weaknesses in this research?"

# 4단계: 데이터 탐색 (Jupyter 노트북)
jupyter notebook explore_chromadb.ipynb
```

### 전체 년도 (ICLR + NeurIPS 2021-2025)

```bash
# 1단계: 모든 데이터 다운로드 (백그라운드 스크립트 사용 권장)
# 개별 다운로드:
uv run src/data/data_collection.py --batch --venue ICLR.cc/2021/Conference
uv run src/data/data_collection.py --batch --venue ICLR.cc/2022/Conference
# ... (나머지 년도 반복)

# 2단계: 모든 년도별 컬렉션 구축
uv run src/vectordb/build_all_vectordb.py

# 3단계: 탐색 및 시각화
jupyter notebook explore_chromadb.ipynb
```

## 프로젝트 구조

```plaintext
paper-review/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_collection.py      # OpenReview 데이터 수집
│   └── vectordb/
│       ├── __init__.py
│       ├── build_vectordb.py        # ChromaDB Vector DB 구축
│       └── build_all_vectordb.py    # 년도별 컬렉션 일괄 구축
├── scripts/
│   ├── debug_openreview.py          # OpenReview API 디버깅 도구
│   └── build_full_database.sh       # 전체 데이터 수집 스크립트
├── data/                            # 다운로드된 JSON 데이터
│   ├── ICLR.cc_2021_Conference_reviews.json
│   ├── ICLR.cc_2022_Conference_reviews.json
│   ├── ...
│   ├── NeurIPS.cc_2021_Conference_reviews.json
│   └── ICLR_NeurIPS_2021_2025_all_reviews.json
├── chromadb/                        # Vector DB 저장소
├── logs/                            # 로그 파일
├── explore_chromadb.ipynb           # ChromaDB 탐색 및 시각화 노트북
├── requirements.txt                 # Python 의존성 패키지
└── README.md                        # 사용 가이드
```

## 프로젝트 기획

### 프로젝트 목표

OpenReview.net의 방대한 **인간 리뷰 데이터**를 RAG(검색 증강 생성)의 **참조 예시(Reference)** 및 **평가 정답지**로 활용하여, 자체 GPU 서버에서 구동되는 **오픈소스 LLM**과 **Agentic Workflow**를 결합한 AI 논문 리뷰어 시스템을 개발합니다.

### 해결하고자 하는 문제

- **리뷰의 고비용 & 고변동성**: 논문 리뷰는 막대한 시간 소모와 리뷰어의 주관에 따른 품질 편차가 존재
- **기존 LLM의 한계**: 범용 LLM은 논문 요약에는 능하나, 논리적 허점이나 실험의 한계를 지적하는 비판적 분석 능력 부족
- **데이터 기반 검증의 부재**: 실제 학회 리뷰 데이터를 정답지로 삼아 객관적 성능 검증 필요

### 기대 효과

- **연구 효율성 극대화**: 논문 1편 파악 시간을 1시간에서 10분으로 단축
- **객관적 성능 지표 확보**: 정량적 평가 지표 기반 성능 측정
- **자체 기술 자산**: API 의존성 없는 온프레미스 AI 리뷰 시스템 구축
- **비용 및 보안 효율성**: 자체 GPU 서버와 오픈소스 LLM 활용으로 비용 절감 및 데이터 주권 확보

## KPI (핵심 성능 지표)

1. **약점(Weaknesses) 재현율 (Recall / Coverage)**
   - 측정: Test Set에서 인간 리뷰어가 지적한 약점 중 AI가 동일하게 지적한 비율
   - 목표: **40% 이상**

2. **평점(Rating) 상관관계 (Correlation)**
   - 측정: AI 예측 평점과 인간 실제 평점 간의 피어슨 상관계수
   - 목표: **0.3 이상**

3. **환각(Hallucination) 비율**
   - 측정: 논문 내용과 무관하거나 사실이 아닌 거짓 지적의 비율
   - 목표: **15% 미만**

## 데이터 구조

### Vector DB 컬렉션

- **년도별 컬렉션 (10개)**:
  - ICLR: `iclr_2021` ~ `iclr_2025`
  - NeurIPS: `neurips_2021` ~ `neurips_2025`

### 문서 구조

각 논문은 다음과 같이 저장됩니다:

- **Abstract 문서**: 논문 초록 (type: 'abstract')
- **Review 문서**: 각 리뷰 (type: 'review')

메타데이터:

- `forum_id`: 논문 고유 ID
- `note_id`: 리뷰 고유 ID (리뷰 문서만)
- `type`: 문서 타입 ('abstract' 또는 'review')

## 다음 단계

1. **RAG 시스템 구현**
   - LangGraph 기반 Agent 구조
   - 유사 리뷰 검색 및 In-Context Learning
   - 논문 분석 결과 생성

2. **성능 평가**
   - KPI 측정 (약점 재현율, 평점 상관관계, 환각 비율)
   - 평가 리포트 생성

3. **웹 인터페이스**
   - Streamlit 기반 UI
   - PDF 업로드 및 리뷰 생성 기능

## 라이선스

MIT License

## 기여

이슈 및 풀 리퀘스트를 환영합니다.
