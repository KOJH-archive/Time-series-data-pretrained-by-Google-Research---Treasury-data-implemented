import requests
import pandas as pd
import io
import os
from datetime import datetime

def fetch_direct_fred():
    print("FRED에서 직접 데이터를 다운로드 중입니다 (2Y, 10Y, 30Y)...")
    
    # FRED CSV 다운로드 URL
    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    params = {
        "id": "DGS2,DGS10,DGS30",
        "cosd": "2022-01-01",
        "coed": datetime.now().strftime("%Y-%m-%d")
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # CSV 읽기
        df = pd.read_csv(io.StringIO(response.text))
        print(f" - 수집된 컬럼: {df.columns.tolist()}")
        
        # 날짜 컬럼명 대응 (DATE 또는 observation_date)
        if 'DATE' in df.columns:
            df.rename(columns={'DATE': 'date'}, inplace=True)
        elif 'observation_date' in df.columns:
            df.rename(columns={'observation_date': 'date'}, inplace=True)
        
        # 만기별 컬럼명 정리
        col_map = {'DGS2': '2Y', 'DGS10': '10Y', 'DGS30': '30Y'}
        for old_col, new_col in col_map.items():
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        
        # 숫자형 변환 및 결측치 처리
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        for col in ['2Y', '10Y', '30Y']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 결측치 제거
        df = df.dropna()
        
        # Spread 계산
        print("Spread 계산 중...")
        df['Spread_10_2'] = df['10Y'] - df['2Y']
        df['Spread_30_10'] = df['30Y'] - df['10Y']
        
        # 저장
        os.makedirs('input_data', exist_ok=True)
        df[['date', 'Spread_10_2']].to_csv('input_data/treasury_spread_10_2.csv', index=False)
        df[['date', 'Spread_30_10']].to_csv('input_data/treasury_spread_30_10.csv', index=False)
        
        print(f"데이터 수집 완료! ({len(df)} 영업일 데이터)")
        return True
        
    except Exception as e:
        print(f"데이터 다운로드 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    fetch_direct_fred()
