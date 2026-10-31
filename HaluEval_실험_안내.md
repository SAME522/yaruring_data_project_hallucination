# HaluEval 질문 군집화 + 로지스틱 회귀: 초기 실험

## 탐구 질문

사용자 질문의 의미적·언어적 특징을 이용해 과거 ChatGPT 답변의 할루시네이션 여부를 얼마나 잘 예측할 수 있는가? 질문 군집을 추가하면 성능이 개선되는가?

## 실행

1. HaluEval 1.0 공식 저장소의 `data/general_data.json`을 다운로드한다.
   https://github.com/RUCAIBox/HaluEval
2. Python 환경에서 `pip install pandas numpy scipy scikit-learn`을 실행한다.
3. `python halueval_project.py --input general_data.json --output results`를 실행한다.

`general_data.json`은 JSON Lines 형식이다. 입력은 질문인 `user_query`와 정답인 `hallucination`만 활용한다. 질문 중복 검사와 학습/시험 분할은 코드에 포함되어 있다.

## 설계

- Yes=1, No=0, 결측/기타 라벨 제외
- 완전히 같은 질문 중 서로 다른 라벨이 달린 사례는 제외. 중복 질문 제거 후 분할
- Stratified random split: 학습 80%, 시험 20%, random_state=42
- 학습 질문만 사용해 TF-IDF(1~2gram, 영어 불용어 제외)를 적합
- 학습 질문 TF-IDF에 MiniBatchKMeans (군집 6개)를 적합하고 시험 질문은 predict로 배정
- 모델: 다수 범주 기준선, 군집만 사용한 로지스틱 회귀, TF-IDF 로지스틱 회귀, TF-IDF + 군집 로지스틱 회귀
- 로지스틱 회귀는 `class_weight='balanced'`; 결정 문턱값 0.5를 사용
- 평가: 정확도, 균형 정확도, 정밀도(Yes), 재현율(Yes), F1(Yes), ROC-AUC

## 이번 실행 결과

`results/model_comparison.csv`와 `results/cluster_summary_train_only.csv`를 참고한다.

- 원본 4,507개, 학습 3,605개, 시험 902개
- 항상 No의 정확도 81.93%, Yes F1 0
- TF-IDF의 Yes F1 0.3663, ROC-AUC 0.6631
- TF-IDF + 군집의 Yes F1 0.3675, ROC-AUC 0.6647
- 군집 추가의 개선은 매우 작아 우연일 가능성도 있다. 반복 검증 전에는 확정하지 않는다.

## 유의점 및 다음 실험

- 자료는 HaluEval에 선별된 과거 ChatGPT 응답의 라벨이다. 오늘날 모든 모델·사용자 질문에 일반화할 수 없다.
- 그룹별 할루시네이션 비율은 탐색용이다. 학습 라벨로 만든 그룹별 비율을 시험 입력값에 넣지 않는다.
- `cluster_summary_train_only.csv`의 군집은 주제가 겹치며 불균형하다. 군집 수나 텍스트 임베딩 표현을 바꾸어 비교할 필요가 있다.
- 향후 군집 수 3~10개를 학습 내부 검증으로 선택하고, 시험 세트는 마지막 한 번만 평가한다.
- 로지스틱 회귀의 예측점수를 실제 위험 확률로 해석하려면 캘리브레이션(확률 보정) 평가가 필요하다.
- 탐구 기록에는 모델의 단점, 버린 가설, 직접 결정한 전처리 기준을 남긴다.

## 파일

- `halueval_project.py`: 재현 가능한 초기 실험 코드
- `results/model_comparison.csv`: 네 모델의 시험 성능
- `results/cluster_summary_train_only.csv`: 학습 데이터에서 본 군집별 사례 수·Yes 비율·상위 단어
- `results/test_predictions.csv`: 시험 질문별 실제 라벨과 최종 모델 예측
- `results/dataset_overview.json`: 사용 데이터 규모
