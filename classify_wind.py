"""Create chronological binary-label training and evaluation datasets."""

import csv
from datetime import datetime, timedelta
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path(__file__).resolve().with_name("wind_dataset.csv")
TRAIN_PATH = Path(__file__).resolve().with_name("wind_train.csv")
EVALUATION_PATH = Path(__file__).resolve().with_name("wind_eval.csv")
PREDICTIONS_PATH = Path(__file__).resolve().with_name("classification_predictions.csv")
SCORES_PATH = Path(__file__).resolve().with_name("classification_scores.csv")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
PREDICTION_COLUMN = "prediction_time"
TARGET_COLUMN = "target_time"
WIND_COLUMN = "actual_wind_speed_ms"
LABEL_COLUMN = "wind_next1h"
REQUIRED_COLUMNS = {PREDICTION_COLUMN, TARGET_COLUMN, WIND_COLUMN}
FEATURE_COLUMNS = [
    "current_wind_speed_ms",
    "current_temperature_C",
    "current_humidity_pct",
    "wind_speed_1h_before_ms",
    "wind_speed_2h_before_ms",
]
PREDICTION_COLUMNS = [
    PREDICTION_COLUMN,
    TARGET_COLUMN,
    WIND_COLUMN,
    LABEL_COLUMN,
    "current_wind_speed_ms",
    "baseline_always_0",
    "baseline_current_wind_ge_4",
    "logistic_wind_next1h_probability",
    "logistic_wind_next1h_prediction",
    "logistic_wind_next1h_prediction_0_20",
]
METHOD_PREDICTIONS = {
    "항상 0": "baseline_always_0",
    "현재 풍속 4.0 m/s 이상": "baseline_current_wind_ge_4",
    "표준화 로지스틱 회귀 (임계값 0.50)": "logistic_wind_next1h_prediction",
    "표준화 로지스틱 회귀 (임계값 0.20)": "logistic_wind_next1h_prediction_0_20",
}


def load_labeled_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load rows, verify one-hour prediction targets, label, and sort by KST target time."""
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 헤더를 읽을 수 없습니다: {path}")
        input_columns = reader.fieldnames
        missing_columns = REQUIRED_COLUMNS.difference(input_columns)
        if missing_columns:
            raise ValueError(
                f"필수 열이 없습니다: {', '.join(sorted(missing_columns))}"
            )

        rows: list[tuple[datetime, dict[str, str]]] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                prediction_time = datetime.strptime(
                    row[PREDICTION_COLUMN], TIMESTAMP_FORMAT
                )
                target_time = datetime.strptime(row[TARGET_COLUMN], TIMESTAMP_FORMAT)
                wind_speed = float(row[WIND_COLUMN])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{line_number}행의 시각 또는 풍속 값이 올바르지 않습니다."
                ) from error

            if not math.isfinite(wind_speed):
                raise ValueError(f"{line_number}행의 정답 풍속이 유한한 숫자가 아닙니다.")
            if target_time - prediction_time != timedelta(hours=1):
                raise ValueError(
                    f"{line_number}행의 prediction_time과 target_time이 "
                    "정확히 1시간 차이가 아닙니다."
                )

            row[LABEL_COLUMN] = "1" if wind_speed >= 4.0 else "0"
            rows.append((target_time, row))

    rows.sort(key=lambda item: item[0])
    target_times = [target_time for target_time, _ in rows]
    if any(current <= previous for previous, current in zip(target_times, target_times[1:])):
        raise ValueError("정답 시각이 중복되어 시간순 데이터로 나눌 수 없습니다.")

    return [*input_columns, LABEL_COLUMN], [row for _, row in rows]


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    """Write one chronological split without adding a row index."""
    with path.open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_labels(rows: list[dict[str, str]]) -> tuple[int, int]:
    """Return counts of negative and positive labels."""
    return sum(row[LABEL_COLUMN] == "0" for row in rows), sum(
        row[LABEL_COLUMN] == "1" for row in rows
    )


def predict_evaluation_rows(
    training_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
) -> pd.DataFrame:
    """Fit scaling and logistic regression only on training rows, then score December."""
    training = pd.DataFrame.from_records(training_rows)
    evaluation = pd.DataFrame.from_records(evaluation_rows)
    missing_features = set(FEATURE_COLUMNS).difference(training.columns)
    if missing_features:
        raise ValueError(f"필수 입력 열이 없습니다: {', '.join(sorted(missing_features))}")

    x_train = training[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    x_evaluation = evaluation[FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x_train.to_numpy(dtype=float)).all():
        raise ValueError("학습 입력에 결측값 또는 유한하지 않은 숫자가 있습니다.")
    if not np.isfinite(x_evaluation.to_numpy(dtype=float)).all():
        raise ValueError("평가 입력에 결측값 또는 유한하지 않은 숫자가 있습니다.")

    y_train = pd.to_numeric(training[LABEL_COLUMN], errors="raise").astype(int)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight=None, max_iter=1000),
    )
    model.fit(x_train, y_train)

    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_evaluation)[:, positive_class_index]
    current_wind = pd.to_numeric(
        evaluation["current_wind_speed_ms"], errors="raise"
    )
    results = pd.DataFrame(
        {
            PREDICTION_COLUMN: evaluation[PREDICTION_COLUMN],
            TARGET_COLUMN: evaluation[TARGET_COLUMN],
            WIND_COLUMN: evaluation[WIND_COLUMN],
            LABEL_COLUMN: evaluation[LABEL_COLUMN],
            "current_wind_speed_ms": current_wind,
            "baseline_always_0": 0,
            "baseline_current_wind_ge_4": (current_wind >= 4.0).astype(int),
            "logistic_wind_next1h_probability": probabilities,
            "logistic_wind_next1h_prediction": (probabilities >= 0.50).astype(int),
            "logistic_wind_next1h_prediction_0_20": (probabilities >= 0.20).astype(int),
        },
        columns=PREDICTION_COLUMNS,
    )
    return results


def score_predictions(
    evaluation_rows: list[dict[str, str]],
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Verify evaluation-row identity and calculate confusion counts and metrics."""
    evaluation = pd.DataFrame.from_records(evaluation_rows)
    identity_columns = [
        PREDICTION_COLUMN,
        TARGET_COLUMN,
        WIND_COLUMN,
        LABEL_COLUMN,
    ]
    expected_identity = evaluation[identity_columns].reset_index(drop=True).astype(str)
    actual_identity = predictions[identity_columns].reset_index(drop=True).astype(str)
    if not expected_identity.equals(actual_identity):
        raise ValueError("예측 결과와 평가 데이터의 행·정답이 일치하지 않습니다.")

    y_true = pd.to_numeric(evaluation[LABEL_COLUMN], errors="raise").to_numpy(dtype=int)
    actual_strong_wind = int((y_true == 1).sum())
    score_rows: list[dict[str, object]] = []
    for method, prediction_column in METHOD_PREDICTIONS.items():
        y_pred = predictions[prediction_column].to_numpy(dtype=int)
        hits = int(((y_true == 1) & (y_pred == 1)).sum())
        misses = int(((y_true == 1) & (y_pred == 0)).sum())
        false_alarms = int(((y_true == 0) & (y_pred == 1)).sum())
        correct_rejections = int(((y_true == 0) & (y_pred == 0)).sum())
        predicted_strong = hits + false_alarms
        score_rows.append(
            {
                "method": method,
                "evaluation_rows": len(evaluation),
                "actual_strong_wind": actual_strong_wind,
                "hits": hits,
                "misses": misses,
                "false_alarms": false_alarms,
                "correct_rejections": correct_rejections,
                "accuracy_pct": 100 * (hits + correct_rejections) / len(evaluation),
                "precision_pct": (
                    100 * hits / predicted_strong if predicted_strong else "해당없음"
                ),
                "recall_pct": 100 * hits / actual_strong_wind if actual_strong_wind else "해당없음",
            }
        )
    return pd.DataFrame(score_rows)


def main() -> None:
    """Create chronological splits, train the model, and save December predictions."""
    fieldnames, rows = load_labeled_rows(DATA_PATH)
    training_rows = [
        row
        for row in rows
        if datetime(2023, 1, 1) <= datetime.strptime(
            row[TARGET_COLUMN], TIMESTAMP_FORMAT
        ) < datetime(2024, 12, 1)
    ]
    evaluation_rows = [
        row
        for row in rows
        if datetime(2024, 12, 1) <= datetime.strptime(
            row[TARGET_COLUMN], TIMESTAMP_FORMAT
        ) < datetime(2025, 1, 1)
    ]

    if not training_rows:
        raise ValueError("학습 기간에 해당하는 행이 없습니다.")
    if not evaluation_rows:
        raise ValueError("평가 기간에 해당하는 행이 없습니다.")

    write_rows(TRAIN_PATH, fieldnames, training_rows)
    write_rows(EVALUATION_PATH, fieldnames, evaluation_rows)
    predictions = predict_evaluation_rows(training_rows, evaluation_rows)
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")
    scores = score_predictions(evaluation_rows, predictions)
    scores.to_csv(SCORES_PATH, index=False, encoding="utf-8-sig")

    train_zeros, train_ones = count_labels(training_rows)
    eval_zeros, eval_ones = count_labels(evaluation_rows)
    print("학습 기간 (정답 시각 기준): 2023-01-01 ~ 2024-11-30")
    print(f"전체 행수: {len(training_rows)} (0: {train_zeros}, 1: {train_ones})")
    print(f"학습 파일: {TRAIN_PATH}")
    print("\n평가 기간 (정답 시각 기준): 2024-12-01 ~ 2024-12-31")
    print(f"전체 행수: {len(evaluation_rows)} (0: {eval_zeros}, 1: {eval_ones})")
    print("첫 평가행:")
    print(",".join(fieldnames))
    print(",".join(evaluation_rows[0][field] for field in fieldnames))
    print(f"평가 파일: {EVALUATION_PATH}")
    print(f"\n예측 결과 파일: {PREDICTIONS_PATH}")
    print("예측 CSV 첫 5행:")
    print(predictions.head(5).to_string(index=False))
    print("\n평가행 일치 확인: 예측 결과와 평가 데이터의 시각·정답이 모두 일치합니다.")
    print(f"실제 강풍 수: {int(scores['actual_strong_wind'].iloc[0])}")
    print(f"분류 성능 파일: {SCORES_PATH}")
    print("예측 방법별 성능 비교:")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
