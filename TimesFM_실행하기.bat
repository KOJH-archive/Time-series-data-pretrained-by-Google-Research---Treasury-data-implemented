@echo off
chcp 65001 >nul
title TimesFM 2.5 종합 분석 시스템
cd /d "%~dp0"

echo ==================================================
echo       TimesFM 2.5 미국채 금리 분석 시스템
echo ==================================================
echo.

REM 1. 가상환경 확인
if not exist ".venv" (
    echo [오류] .venv 폴더를 찾을 수 없습니다.
    pause
    exit /b
)

REM 2. 가상환경 활성화
call .venv\Scripts\activate

REM 3. 전 과정 실행
echo [1/3] 미국 재무부 데이터 수집 중 (FRED)...
python fetch_fred.py

echo [2/3] TimesFM AI 예측 진행 중...
python predict_input.py

echo [3/3] 분석 리포트 생성 중...
python generate_report.py

echo.
echo ==================================================
echo   모든 분석이 완료되었습니다! 
echo   'output' 폴더에서 리포트와 이미지를 확인하세요.
echo ==================================================
pause
