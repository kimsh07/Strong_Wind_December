"""Plot actual wind speed and December strong-wind classifications."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path(__file__).resolve().with_name("classification_predictions.csv")
PLOT_PATH = Path(__file__).resolve().with_name("strong_wind_december.png")
START_TIME = pd.Timestamp("2024-12-01 00:00:00")
END_TIME = pd.Timestamp("2024-12-08 00:00:00")
ACTUAL_COLUMN = "actual_wind_speed_ms"
LABEL_COLUMN = "wind_next1h"
PREDICTION_COLUMNS = {
    "0.50 예측": "logistic_wind_next1h_prediction",
    "0.20 예측": "logistic_wind_next1h_prediction_0_20",
}


def load_december_week(path: Path) -> pd.DataFrame:
    """Load December 1-7 rows and verify labels against the 4.0 m/s definition."""
    frame = pd.read_csv(path, parse_dates=["target_time"])
    required = {
        "target_time",
        ACTUAL_COLUMN,
        LABEL_COLUMN,
        *PREDICTION_COLUMNS.values(),
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {', '.join(sorted(missing_columns))}")

    frame = frame.loc[
        (frame["target_time"] >= START_TIME) & (frame["target_time"] < END_TIME)
    ].sort_values("target_time")
    if frame.empty:
        raise ValueError("CSV에 12월 1~7일 평가행이 없습니다.")

    if frame["target_time"].duplicated().any():
        raise ValueError("12월 1~7일 데이터에 중복 정답 시각이 있습니다.")

    actual_strong = frame[ACTUAL_COLUMN] >= 4.0
    if not actual_strong.astype(int).equals(frame[LABEL_COLUMN].astype(int)):
        raise ValueError("실제 강풍 라벨이 실제 풍속 4.0 m/s 기준과 일치하지 않습니다.")
    for column in PREDICTION_COLUMNS.values():
        if not frame[column].isin([0, 1]).all():
            raise ValueError(f"{column}에 0 또는 1이 아닌 예측값이 있습니다.")
    return frame


def plot_week(frame: pd.DataFrame) -> None:
    """Create aligned actual-wind and binary-event plots on a shared time axis."""
    figure, (wind_axis, event_axis) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1.5]},
    )

    wind_axis.plot(
        frame["target_time"],
        frame[ACTUAL_COLUMN],
        color="tab:blue",
        linewidth=1.8,
        label="Actual wind speed",
    )
    wind_axis.axhline(
        4.0,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        label="Strong-wind threshold (4.0 m/s)",
    )
    wind_axis.set_ylabel("Wind speed (m/s)")
    wind_axis.set_title("Strong-wind observations and predictions (Dec 1-7, 2024 KST)")
    wind_axis.grid(True, alpha=0.3)
    wind_axis.legend(loc="upper left")

    categories = [
        ("Actual strong wind", LABEL_COLUMN, "black", "o"),
        ("Prediction (0.50)", PREDICTION_COLUMNS["0.50 예측"], "tab:orange", "s"),
        ("Prediction (0.20)", PREDICTION_COLUMNS["0.20 예측"], "tab:green", "^"),
    ]
    for y_position, (label, column, color, marker) in enumerate(categories):
        strong_rows = frame[column].astype(int) == 1
        event_axis.scatter(
            frame.loc[strong_rows, "target_time"],
            [y_position] * int(strong_rows.sum()),
            color=color,
            marker=marker,
            s=35,
            label=label,
        )

    event_axis.set_yticks(range(len(categories)))
    event_axis.set_yticklabels([category[0] for category in categories])
    event_axis.invert_yaxis()
    event_axis.set_xlabel("Target time (KST)")
    event_axis.grid(True, axis="x", alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(PLOT_PATH, dpi=160)
    print(f"그래프 저장 위치: {PLOT_PATH}")
    plt.show()


def print_december_2_mistake(frame: pd.DataFrame) -> None:
    """Print the first December 2 disagreement between actual and either model rule."""
    day = frame.loc[frame["target_time"].dt.date == pd.Timestamp("2024-12-02").date()]
    for _, row in day.iterrows():
        actual = int(row[LABEL_COLUMN])
        for label, column in PREDICTION_COLUMNS.items():
            predicted = int(row[column])
            if predicted != actual:
                print("12월 2일 오분류 사례:")
                print(f"정답 시각: {row['target_time']}")
                print(f"실제 풍속: {row[ACTUAL_COLUMN]:.1f} m/s (실제 강풍: {actual})")
                print(f"{label}: {predicted}")
                print(
                    "강풍 확률: "
                    f"{row['logistic_wind_next1h_probability']:.6f}"
                )
                return
    print("12월 2일에 오분류 사례가 없습니다.")


def main() -> None:
    """Load the saved prediction CSV, plot December 1-7, and find one error."""
    frame = load_december_week(DATA_PATH)
    print(f"선택 행 수: {len(frame)}")
    print(f"정답 시각 범위: {frame['target_time'].min()} ~ {frame['target_time'].max()}")
    print_december_2_mistake(frame)
    plot_week(frame)


if __name__ == "__main__":
    main()
