import pandas as pd
import os
from datetime import datetime

def generate_analysis():
    print("분석 리포트를 생성 중입니다...")
    
    report_content = f"# 미국채 금리차(Spread) 예측 분석 리포트\n"
    report_content += f"**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    files = [
        ('treasury_spread_10_2.csv', '10-2 Spread (10년물 - 2년물)'),
        ('treasury_spread_30_10.csv', '30-10 Spread (30년물 - 10년물)')
    ]
    
    for filename, title in files:
        input_path = f'input_data/{filename}'
        forecast_path = f'output/{filename.replace(".csv", "_forecast.csv")}'
        
        if not os.path.exists(input_path) or not os.path.exists(forecast_path):
            continue
            
        # 데이터 로드
        df_input = pd.read_csv(input_path)
        df_forecast = pd.read_csv(forecast_path)
        
        latest_val = df_input.iloc[-1, 1]
        start_forecast = df_forecast.iloc[0]['prediction']
        end_forecast = df_forecast.iloc[-1]['prediction']
        trend = end_forecast - start_forecast
        
        # 분석 논리
        status = "정상(Normal)" if latest_val >= 0 else "역전(Inverted)"
        trend_desc = "상승(Widening)" if trend > 0.005 else ("하락(Narrowing)" if trend < -0.005 else "횡보(Stable)")
        
        report_content += f"## {title}\n"
        report_content += f"- **현재 수치**: {latest_val:.3f} (%p)\n"
        report_content += f"- **현재 상태**: **{status}**\n"
        report_content += f"- **향후 전망**: **{trend_desc}** (예측 종료 시점: {end_forecast:.3f})\n"
        
        report_content += f"- **상세 분석**: "
        if status == "역전(Inverted)":
            if trend > 0: report_content += "역전 현상이 점차 해소되는 국면에 진입할 것으로 예측됩니다. "
            else: report_content += "현재의 역전 현상이 당분간 심화되거나 유지될 것으로 보입니다. "
        else:
            if trend > 0: report_content += "장단기 금리차가 벌어지며 경기 회복에 대한 기대감이 반영되는 추세입니다. "
            else: report_content += "금리차가 좁혀지는 경향이 있어 향후 경기 둔화 가능성에 대한 모니터링이 필요합니다. "
            
        report_content += "\n\n"

    report_content += "---\n*본 리포트는 TimesFM 2.5 모델의 예측 데이터를 기반으로 자동 생성되었습니다.*"

    # 파일 저장
    os.makedirs('output', exist_ok=True)
    with open('output/analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"리포트 생성 완료: output/analysis_report.md")

if __name__ == "__main__":
    generate_analysis()
