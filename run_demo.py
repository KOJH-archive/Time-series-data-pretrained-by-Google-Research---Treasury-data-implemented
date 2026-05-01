import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import koreanize_matplotlib
import torch

# 1. 환경 확인 및 라이브러리 임포트
try:
    import timesfm
except ImportError:
    print("Error: 'timesfm' 라이브러리가 설치되어 있지 않습니다.")
    print("가상환경에서 'uv pip install timesfm[torch]'를 실행해 주세요.")
    sys.exit(1)

def run_demo():
    print("=== TimesFM 2.5 예측 데모 시작 ===")
    
    # 2. 모델 로드 (최초 실행 시 다운로드로 인해 시간이 소요될 수 있습니다)
    print("모델을 로딩 중입니다 (google/timesfm-2.5-200m-pytorch)...")
    try:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=256,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            fix_quantile_crossing=True,
        ))
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return

    # 3. 샘플 데이터 생성 (사인파 + 노이즈)
    print("샘플 데이터를 생성하고 예측을 수행합니다...")
    x = np.linspace(0, 50, 500)
    y = np.sin(x) + np.random.normal(0, 0.1, 500)
    
    # 4. 예측 수행 (24단계 미래 예측)
    horizon = 24
    point_forecast, quantile_forecast = model.forecast(horizon=horizon, inputs=[y])
    
    # 5. 결과 시각화
    plt.figure(figsize=(12, 6))
    
    # 과거 데이터 (마지막 100개만 표시)
    history_len = 100
    plt.plot(range(history_len), y[-history_len:], label="과거 데이터", color="gray", alpha=0.6)
    
    # 예측 데이터
    forecast_range = range(history_len, history_len + horizon)
    plt.plot(forecast_range, point_forecast[0], label="예측(중앙값)", color="blue", linewidth=2)
    
    # 신뢰 구간 (80% 구간: q10 ~ q90)
    plt.fill_between(
        forecast_range, 
        quantile_forecast[0, :, 1], 
        quantile_forecast[0, :, 9], 
        color="blue", alpha=0.2, label="80% 신뢰 구간"
    )
    
    plt.title("TimesFM 2.5 시계열 예측 결과")
    plt.xlabel("시간")
    plt.ylabel("값")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # 결과 저장
    output_img = "demo_result.png"
    plt.savefig(output_img)
    print(f"예측이 완료되었습니다! 결과 이미지가 '{output_img}'로 저장되었습니다.")
    
    # 결과 출력
    print("\n[예측값 (처음 5단계)]")
    print(point_forecast[0][:5])

if __name__ == "__main__":
    # 성능 최적화 설정
    torch.set_float32_matmul_precision("high")
    run_demo()
