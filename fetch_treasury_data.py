import requests
import pandas as pd
import os

def fetch_treasury_yields():
    print("미국 재무부 API에서 금리 데이터를 수집 중입니다...")
    
    # 미국 재무부 Fiscal Data API 엔드포인트
    # 테이블명: daily_treasury_yield_curve_rate
    url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/daily_treasury_yield_curve_rate"
    
    # 2022년 1월 1일 이후 데이터 필터링 및 정렬
    params = {
        "filter": "record_date:gte:2022-01-01",
        "sort": "-record_date", # 최신순으로 가져오기
        "page[size]": 10000
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data_json = response.json()
        raw_data = data_json['data']
        
        df = pd.DataFrame(raw_data)
        
        # 실제 API 응답 컬럼명 확인 및 매핑
        # 보통 2년물은 'series_2year' 또는 'yield_2yr' 등으로 들어올 수 있음
        # API 문서상 표준 컬럼명 확인 결과: series_2year, series_5year, series_10year, series_30year
        target_cols = {
            'record_date': 'date',
            'series_2year': '2Y',
            'series_5year': '5Y',
            'series_10year': '10Y',
            'series_30year': '30Y'
        }
        
        # 존재하는 컬럼만 선택
        available_cols = [c for c in target_cols.keys() if c in df.columns]
        df = df[available_cols].rename(columns=target_cols)
        
        # 숫자형 변환 및 결측치 처리
        for col in ['2Y', '5Y', '10Y', '30Y']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').dropna()
        
        # Spread 계산
        print("Spread 계산 중 (10-2, 30-10)...")
        df['Spread_10_2'] = df['10Y'] - df['2Y']
        df['Spread_30_10'] = df['30Y'] - df['10Y']
        
        # CSV 저장
        if not os.path.exists('input_data'):
            os.makedirs('input_data')
            
        df[['date', 'Spread_10_2']].to_csv('input_data/treasury_spread_10_2.csv', index=False)
        df[['date', 'Spread_30_10']].to_csv('input_data/treasury_spread_30_10.csv', index=False)
        
        print(f"데이터 수집 완료! ({len(df)} 영업일 데이터)")
        return True
        
    except Exception as e:
        print(f"데이터 수집 중 오류 발생: {e}")
        return False

if __name__ == "__main__":
    fetch_treasury_yields()
