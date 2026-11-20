# 강풍 관측 및 예측 시각화

`classification_predictions.csv`의 풍속 관측값과 강풍 분류 예측값을 읽어 2024년 12월 1~7일 구간을 시각화하는 Python 스크립트입니다.

## 실행 파일

- `plot_strong_wind_december.py`: 데이터 검증, 오분류 사례 출력, 그래프 생성
- `classification_predictions.csv`: 실제 풍속과 강풍 라벨 및 두 로지스틱 예측값이 포함된 입력 데이터
- `strong_wind_december.png`: 스크립트가 생성하는 결과 그래프

이 스크립트는 실행 파일과 같은 폴더에 있는 CSV와 그래프 경로를 사용하므로, 실행 위치와 관계없이 상대 경로 문제가 없습니다.

## 필요한 환경

- Python 3
- pandas
- matplotlib

패키지 설치:

```powershell
python -m pip install pandas matplotlib
```

## 실행 방법

PowerShell에서 다음과 같이 실행합니다.

```powershell
cd "C:\Users\User\Desktop\weather_windspeed"
python ".\plot_strong_wind_december.py"
```

Windows Python 실행기(`py`)를 사용할 수 있습니다.

```powershell
py ".\plot_strong_wind_december.py"
```

## 입력 CSV 형식

기본 입력 파일은 `classification_predictions.csv`입니다. 다음 열이 필요합니다.

| 열 이름 | 설명 |
| --- | --- |
| `target_time` | 예측 목표 시각 |
| `actual_wind_speed_ms` | 실제 풍속(m/s) |
| `wind_next1h` | 실제 풍속을 기준으로 계산한 강풍 라벨 |
| `logistic_wind_next1h_probability` | 강풍 예측 확률 |
| `logistic_wind_next1h_prediction` | 임계값 0.50 적용 예측 |
| `logistic_wind_next1h_prediction_0_20` | 임계값 0.20 적용 예측 |

강풍 기준은 실제 풍속이 `4.0 m/s` 이상인 경우입니다.

```text
실제 강풍 라벨 = 1 (actual_wind_speed_ms >= 4.0)
실제 강풍 라벨 = 0 (actual_wind_speed_ms < 4.0)
```

스크립트는 강풍 라벨이 실제 풍속 기준과 일치하는지 검증합니다.

## 분석 범위

기본적으로 다음 기간의 데이터만 선택합니다.

- 시작: 2024-12-01 00:00:00
- 종료: 2024-12-08 00:00:00

따라서 그래프와 콘솔 출력에는 2024년 12월 1~7일 데이터가 포함됩니다.

## 출력

실행하면 콘솔에 다음 내용이 출력됩니다.

- 선택한 행 수
- 선택 데이터의 정답 시각 범위
- 12월 2일의 첫 번째 오분류 사례가 있으면 해당 시각과 실제 풍속
- 해당 예측 라벨과 강풍 확률

그래프는 상단의 실제 풍속 그래프와 하단의 강풍 발생 여부 그래프를 하나의 시간 축으로 표시합니다.

- 상단 그래프: 실제 풍속과 `4.0 m/s` 기준선
- 하단 그래프:
  - 실제 강풍: 검은 원
  - 임계값 0.50 예측: 주황색 사각형
  - 임계값 0.20 예측: 초록색 삼각형

그래프는 다음 파일로 저장됩니다.

```text
strong_wind_december.png
```

저장 후 그래프 창도 표시됩니다.

## 데이터 검증 및 오류 처리

다음 문제가 발견되면 실행을 중단하고 오류 메시지를 표시합니다.

- 필수 열 누락
- 2024년 12월 1~7일 데이터가 없음
- 중복된 목표 시각
- 강풍 라벨과 실제 풍속 기준 불일치
- 예측값이 `0` 또는 `1`이 아닌 경우
