# 사용자 프롬프트의 할루시네이션 위험 분석 (HaluEval)

데이터 분석을 중심으로, 사용자가 질문을 AI에 전송하기 **전에** 과거 데이터에서 관찰된 할루시네이션 위험 신호를 탐색하는 학교 데이터 과학 프로젝트입니다.

> **중요한 한계:** 질문만으로 실제 답변의 할루시네이션 발생을 확정하거나 방지할 수는 없습니다. 이 저장소의 `yes/no`는 HaluEval에 기록된 **특정 과거 응답**의 라벨입니다. 모델의 예측점수를 다른 AI 모델의 실제 발생 확률로 곧바로 해석하지 않습니다.

## 현재 작업 상태 (2026-10-31)

| 단계 | 상태 | 주요 내용 |
| --- | --- | --- |
| 데이터 탐색·전처리 | 수행 | 질문 길이·단어 수·결측치·라벨 분포, 중복 질문 제거 |
| 질문/단어 군집화 | 실험 | TF-IDF, SVD, K-means, 단어 역할·주제 기반 세분화 |
| 군집 교차검증 | 실험 | 질문 유형과 단어 역할/주제 교차 집계 및 5겹 교차검증 |
| 위험 예측 | 초기 기준선 비교 | TF-IDF + 로지스틱 회귀 등; 추가 군집 특성의 이득 검토 |
| 모델 오디션 | 예정 | kNN, 결정트리, 랜덤 포레스트, XGBoost 등 같은 분할·지표에서 비교 |
| 프롬프트 입력 전 경고 기능 | 예정 | 예측 성능·확률 보정·한계를 확인한 뒤 구현 검토 |

현재 `main/main.py`는 **기초통계 탐색용 Streamlit 앱**입니다. 프롬프트 위험 예측 UI는 아직 없습니다.

## 관찰한 핵심 결과

- HaluEval 일반 질문 4,507개 중 `yes` 815개, `no` 3,692개.
- 중복·상충 라벨 처리 후 학습 3,605개와 보류 시험 902개로 분리 (`random_state=42`).
- 최근 실험의 **학습 데이터 내부 5겹 교차검증**에서 TF-IDF 기준선 ROC-AUC 평균 약 **0.661**, AP 평균 약 **0.317**.
- 같은 비교에서 TF-IDF + 8개 세부 주제 군집 ROC-AUC 평균 약 **0.659**. 군집을 더 세분화했다고 예측 성능이 개선되었다고 볼 수 없었습니다.
- 위 숫자는 **보류 시험 데이터에서 최종 검증한 성능이 아닙니다**. 실험 과정에서 후보들을 반복 비교했으므로 별도의 최종 시험과 확률 보정 점검이 필요합니다.

수치는 `outputs/halueval_final_refinement/final_model_comparison.csv`에 보관합니다. 중간 결과는 같은 `outputs/`의 각 실험 요약 CSV·JSON을 참고하세요.

## 프로젝트 구성

| 경로 | 내용 |
| --- | --- |
| `main/` | 기존 Streamlit 탐색 앱 (GitHub 기존 파일 유지) |
| `log/` | 학교 차시별 기록 (기존 기록 유지) |
| `halueval_project.py` | 질문 군집 유무에 따른 초기 기준선 실험 |
| `halueval_preprocess_pipeline.py` | 원본 질문 전처리와 학습/시험 분리 |
| `refine_halueval_clusters.py` | 질문·단어 계층 특성 구성 |
| `re_cluster_halueval_without_templates.py` | 질문 군집 추가 수정 |
| `halueval_keyword_topic_pilot.py`, `expand_halueval_topic_groups.py` | 주제·질문 유형 탐색 |
| `recluster_unclassified_halueval.py`, `density_secondpass_halueval.py`, `recluster_halueval_remaining_459.py` | 미분류 질문 재검토 |
| `consolidate_halueval_word_clusters.py`, `refine_halueval_word_clusters.py` | 단어군집 통합 및 품질 개선 |
| `halueval_qs_cross_experiment.py` | 질문군집 × 단어군집 교차실험 |
| `halueval_final_refinement.py` | 마지막 세부 주제 군집과 기준선 비교 |
| `outputs/` | **선별한 집계·성능 표** (원본 질문별 대용량 파일은 제외) |

## 실행과 재현

```bash
python -m pip install -r analysis_requirements.txt
```

공식 [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval)의 `data/general_data.json`을 로컬에 받아 프로젝트 루트에 `general_data.json`으로 저장합니다. 해당 파일은 JSON Lines 형식입니다. 데이터의 라이선스·이용 조건은 원본 저장소에서 확인하세요.

```bash
python halueval_project.py --input general_data.json --output results
python halueval_preprocess_pipeline.py --input general_data.json --output outputs/halueval_prep
python refine_halueval_clusters.py
```

후속 실험 스크립트는 `outputs/`의 선행 중간 파일을 입력으로 사용하므로, 스크립트의 `SOURCE`, `SRC`, `IN` 경로를 확인하고 위 표의 대략적인 진행 순서대로 실행해야 합니다. **전체 파이프라인을 한 번에 재실행하는 통합 실행기는 아직 없으며**, 일부 군집 통합 스크립트는 이번 데이터의 행 수를 검증합니다. 수록한 작은 결과 표는 실험 기록용이며, 중간 데이터 전체를 대신하지 않습니다.

앱 실행은 기존 `main/requirements`를 설치한 다음 다음 명령을 사용합니다.

```bash
python -m pip install -r main/requirements
streamlit run main/main.py
```

## 다음 작업

1. 현재 군집의 **해석 가능성**과 **예측 성능 기여**를 구분해 평가합니다. 성능 향상이 없으면 군집 구조를 더 복잡하게 만들지 않습니다.
2. 같은 학습/검증 분할과 같은 입력 정보로 기준선, kNN, 결정트리, 랜덤 포레스트, XGBoost 등 **3종 이상**을 비교합니다.
3. 정확도 외에 소수 클래스(`yes`)의 **재현율, 정밀도, F1, ROC-AUC, AP**를 기록합니다. 최종 단계에서는 시험 세트·확률 보정·오경고율을 점검합니다.
4. 사용자에게 제공할 것은 단정적인 판정이 아니라, 위험 신호와 **질문을 더 명확히 하는 제안**입니다.

원본 데이터는 [HaluEval](https://github.com/RUCAIBox/HaluEval) 연구용 자료입니다. 실험 결과는 해당 자료에 한정된 관찰이며 일반적인 AI 시스템의 성능을 보장하지 않습니다.
