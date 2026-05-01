import pandas as pd
import requests
import io
import os
from datetime import datetime

def fetch_multi_year_treasury():
    print("미국 재무부 사이트에서 2022년부터 현재까지의 데이터를 수집합니다...")
    
    combined_df = pd.DataFrame()
    current_year = datetime.now().year
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 2022년부터 현재 연도까지 루프
    for year in range(2022, current_year + 1):
        print(f" - {year}년 데이터 수집 중...")
        url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value={year}"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # pandas의 read_html을 사용하여 테이블 추출
            tables = pd.read_html(io.StringIO(response.text))
            if not tables:
                continue
                
            df = tables[0]
            # 컬럼명 정리
            df.columns = [str(col).strip() for col in df.columns]
            
            # 필요한 컬럼: Date, 2 Yr, 10 Yr, 30 Yr
            # 사이트마다 컬럼명이 미세하게 다를 수 있어 유연하게 매핑
            col_map = {
                'Date': 'date',
                '2 Yr': '2Y',
                '5 Yr': '5Y',
                '10 Yr': '10Y',
                '30 Yr': '30Y'
            }
            
            # 실제 존재하는 컬럼 필터링
            available_map = {k: v for k, v in col_map.items() if k in df.columns}
            df = df[list(available_map.keys())].rename(columns=available_map)
            
            combined_df = pd.concat([combined_df, df], ignore_index=True)
            
        except Exception as e:
            print(f"   [경고] {year}년 데이터 수집 실패: {e}")

    if combined_df.empty:
        print("데이터 수집에 실패했습니다.")
        return False

    # 데이터 정제
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    for col in ['2Y', '5Y', '10Y', '30Y']:
        if col in combined_df.columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    combined_df = combined_df.sort_values('date').dropna().drop_duplicates('date')

    # Spread 계산
    print("Spread 계산 중 (10-2, 30-10)...")
    combined_df['Spread_10_2'] = combined_df['10Y'] - combined_df['2Y']
    combined_df['Spread_30_10'] = combined_df['30Y'] - combined_df['10Y']

    # 저장
    if not os.path.exists('input_data'):
        os.makedirs('input_data')
        
    combined_df[['date', 'Spread_10_2']].to_csv('input_data/treasury_spread_10_2.csv', index=False)
    combined_df[['date', 'Spread_30_10']].to_csv('input_data/treasury_spread_30_10.csv', index=False)
    
    print(f"성공적으로 {len(combined_df)}건의 데이터를 수집하여 저장했습니다.")
    return True

if __name__ == "__main__":
    fetch_multi_year_treasury()
