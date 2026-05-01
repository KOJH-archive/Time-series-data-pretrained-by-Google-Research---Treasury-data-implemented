import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import koreanize_matplotlib
import torch
import timesfm

def find_target_columns(df):
    """CSV에서 날짜와 값 컬럼을 자동으로 찾습니다."""
    cols = df.columns.tolist()
    date_col = None
    value_col = None
    
    # 날짜 컬럼 후보
    date_candidates = ['date', 'datetime', '날짜', '일자', 'time']
    for c in cols:
        if c.lower() in date_candidates:
            date_col = c
            break
            
    # 값 컬럼 후보 (숫자형 데이터 중 첫 번째)
    value_candidates = ['value', 'sales', 'price', '값', '매출', '가격', '수량']
    for c in cols:
        if c.lower() in value_candidates:
            value_col = c
            break
    
    # 후보가 없으면 그냥 첫 번째 컬럼을 날짜, 두 번째를 값으로 설정
    if not date_col: date_col = cols[0]
    if not value_col: value_col = cols[1] if len(cols) > 1 else cols[0]
    
    return date_col, value_col

def main():
    print("=== TimesFM 자동 예측 프로세스 시작 ===")
    
    # 1. 출력 폴더 생성
    if not os.path.exists('output'):
        os.makedirs('output')
        
    # 2. 모델 로드
    print("모델 로딩 중...")
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    model.compile(timesfm.ForecastConfig(
        max_context=1024, max_horizon=256, normalize_inputs=True,
        use_continuous_quantile_head=True, fix_quantile_crossing=True,
    ))

    # 3. 입력 파일 탐색
    files = glob.glob('input_data/*.csv')
    if not files:
        print("[알림] input_data 폴더에 CSV 파일이 없습니다.")
        return

    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"\n파일 처리 중: {filename}")
        
        try:
            # 데이터 로드
            df = pd.read_csv(file_path)
            date_col, value_col = find_target_columns(df)
            print(f" - 감지된 컬럼: 날짜='{date_col}', 값='{value_col}'")
            
            # 데이터 준비
            values = df[value_col].values.astype(np.float32)
            
            # 예측 (데이터 길이의 20% 또는 최대 24단계 예측)
            horizon = min(24, max(5, int(len(values) * 0.2)))
            point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[values])
            
            # 결과 시각화
            plt.figure(figsize=(12, 6))
            plt.plot(df[value_col].values, label="실제 데이터", color="gray", alpha=0.6)
            
            x_forecast = range(len(values), len(values) + horizon)
            plt.plot(x_forecast, point_forecast[0], label="예측(중앙값)", color="red", linewidth=2)
            plt.fill_between(x_forecast, quantile_forecast[0, :, 1], quantile_forecast[0, :, 9],
                             color="red", alpha=0.15, label="80% 신뢰 구간")
            
            plt.title(f"예측 결과: {filename}")
            plt.xlabel("시간 지표")
            plt.ylabel(value_col)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 저장
            output_name = filename.replace('.csv', '_result.png')
            plt.savefig(f'output/{output_name}')
            print(f" - 시각화 완료: output/{output_name}")
            
            # 예측 데이터 저장
            res_df = pd.DataFrame({
                'forecast_step': range(1, horizon + 1),
                'prediction': point_forecast[0],
                'lower_80': quantile_forecast[0, :, 1],
                'upper_80': quantile_forecast[0, :, 9]
            })
            res_df.to_csv(f'output/{filename.replace(".csv", "_forecast.csv")}', index=False)
            
        except Exception as e:
            print(f" - [오류] {filename} 처리 중 에러 발생: {e}")

    print("\n모든 작업이 완료되었습니다! 'output' 폴더를 확인하세요.")

if __name__ == "__main__":
    main()
