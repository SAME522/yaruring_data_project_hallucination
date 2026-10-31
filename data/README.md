# HaluEval 1.0 군집화 데이터

이 폴더에는 HaluEval 질문을 전처리하고 질문/단어 군집을 분석한 **가공 CSV**가 있습니다. 공식 데이터셋의 원본 파일이 아닙니다.

## 원본 데이터 링크

- **공식 저장소:** https://github.com/RUCAIBox/HaluEval
- **원본 데이터(JSON Lines):** https://github.com/RUCAIBox/HaluEval/blob/main/data/general_data.json
- **원본 Raw 다운로드:** https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/general_data.json
- **원본 라이선스(MIT):** https://github.com/RUCAIBox/HaluEval/blob/main/LICENSE
- 원본 라이선스 문구는 이 폴더의 `HaluEval_LICENSE.txt`에도 보존했습니다.

## 제공 CSV

| 파일 | 행 × 열 | 각 행의 단위 | 내용 및 범위 |
| --- | ---: | --- | --- |
| `questions_clustered.csv` | 4,507 × 68 | 질문 1개 | 전체 질문의 길이·단어 특징, 초기 질문군집, 단어군집 비중, 주제/요청 유형, 교차 특성. `split`은 train 3,605개/test 902개. |
| `questions_multistage_q1.csv` | 2,734 × 37 | 질문 1개 | 수정 질문군집 Q′1의 **학습 데이터만** 대상으로 14개 사전 주제, 밀도 기반 후보, 재군집 후보를 통합한 결과. 전체 데이터가 아님. |
| `questions_subtopics_train.csv` | 3,605 × 8 | 학습 질문 1개 | 학습용 질문에 부여한 세부 주제 8종 및 질문×주제 교차 특성. 시험 902개는 포함하지 않음. |
| `words_clustered_refined.csv` | 5,650 × 18 | 단어 1개 | 학습 질문에서 관측한 단어의 기존 W군집, 역할, 세부 의미 그룹, 검토 상태 및 학습 집계. |

## 해석할 때 주의할 점

1. **`source_row`** 는 원본 JSONL의 행 순서에 기반한 식별자입니다. 질문 관련 CSV들은 이를 키로 연결할 수 있습니다. `original_id`는 원본 값을 보존한 참고용 값이며 일부 빈 값이 있습니다.
2. `label`의 `yes/no` 및 `y`의 1/0은 **해당 데이터셋에 기록된 과거 답변의 할루시네이션 판정**입니다. 질문 자체가 모든 AI에서 같은 확률로 오류를 유발한다는 뜻이 아닙니다.
3. 질문·단어 군집의 번호는 데이터에 맞춰 만들어진 구분이며, 완전히 검증된 의미적 정답 분류가 아닙니다. 사전 단어 매칭에 의한 주제 태그와 비지도 군집 결과도 구분해야 합니다.
4. `questions_multistage_q1.csv`에서는 1,596개가 사전 주제로 분류되고, 679개와 291개가 추가 **군집 후보**로 배정되었으며, 168개는 미분류로 남습니다. 후보의 의미는 추가 검증이 필요합니다.
5. 단어 파일의 `train_yes`, `train_yes_rate` 등은 **학습 라벨에서 계산한 사후 통계**입니다. 예측 모델의 입력 특성으로 그대로 사용하면 정답 정보가 섞일 수 있습니다. 시험 데이터를 포함해 발생률을 계산하지 마세요.
6. 이후 모델 평가는 기존 `split`을 유지하고, 시험 데이터의 정답을 이용해 군집을 다시 정하거나 특성을 선택하지 않도록 주의해야 합니다.

이 폴더는 CSV 자료와 출처를 공개하는 목적이며, 군집 생성 스크립트 전체와 재실행용 원본 JSONL은 포함하지 않았습니다.
