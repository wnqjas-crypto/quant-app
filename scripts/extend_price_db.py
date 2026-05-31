import FinanceDataReader as fdr
import pandas as pd
import os
import warnings
import time

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR   = os.path.join(BASE_DIR, 'db')
OUTPUT_FILE = os.path.join(DB_DIR, 'price_db.parquet')

# 2015Q1 signal_date(2015-05-15) 기준 120거래일 이전 = 약 2014-10월
# 여유있게 2014-01-01 부터 수집
START_DATE = '2014-01-01'
END_DATE   = '2019-12-31'

print("\n📦 KRX 종목 리스트 로드 중...")
krx = fdr.StockListing('KRX')
krx['Code'] = krx['Code'].astype(str).str.zfill(6)
tickers = krx['Code'].tolist()
print(f"✅ 총 {len(tickers):,}개 종목")

# 기존 DB 로드
print("\n📂 기존 가격 DB 로드 중...")
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_parquet(OUTPUT_FILE)
    print(f"✅ 기존 DB: {len(existing_df):,}건")
else:
    existing_df = pd.DataFrame()
    print("⚠️ 기존 DB 없음 — 새로 생성")

# 2014-2019 가격 수집 (기존 DB에 이미 해당 기간 있는 종목은 건너뜀)
if not existing_df.empty:
    date_col = 'Date' if 'Date' in existing_df.columns else 'date'
    existing_df[date_col] = pd.to_datetime(existing_df[date_col])
    cutoff = pd.Timestamp('2020-01-01')
    already_historical = set(
        existing_df[existing_df[date_col] < cutoff]['ticker'].unique()
    )
else:
    already_historical = set()

print(f"\n🔍 이미 과거 데이터 있는 종목: {len(already_historical):,}개 → 건너뜀")
print(f"📥 수집 대상: {len(tickers) - len(already_historical):,}개 종목")
print(f"\n🚀 2014~2019 가격 데이터 수집 시작...\n")

new_data = []
success = 0
failed  = 0

for idx, ticker in enumerate(tickers):

    if ticker in already_historical:
        continue

    try:
        df = fdr.DataReader(ticker, START_DATE, END_DATE)

        if df.empty:
            failed += 1
            continue

        df = df.reset_index()
        df['ticker'] = ticker

        keep_cols = ['Date', 'ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        safe_cols = [c for c in keep_cols if c in df.columns]
        df = df[safe_cols]

        new_data.append(df)
        success += 1

        if success % 100 == 0:
            print(f"✅ 성공:{success:,}  ❌ 실패:{failed:,}  진행:{idx+1:,}/{len(tickers):,}")

    except Exception:
        failed += 1

    time.sleep(0.05)

print(f"\n✅ 수집 완료 — 성공:{success:,}  실패:{failed:,}")

if not new_data:
    print("ℹ️ 새로 추가할 데이터 없음 — 기존 DB 유지")
else:
    print("\n📦 기존 DB와 병합 중...")
    new_df = pd.concat(new_data, ignore_index=True)
    new_df['Date'] = pd.to_datetime(new_df['Date'])

    if not existing_df.empty:
        # 컬럼명 통일
        if 'date' in existing_df.columns and 'Date' not in existing_df.columns:
            existing_df = existing_df.rename(columns={'date': 'Date'})
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df['Date'] = pd.to_datetime(final_df['Date'])
    final_df = final_df.sort_values(['ticker', 'Date'])
    final_df = final_df.drop_duplicates(subset=['ticker', 'Date'], keep='last')

    print("\n💾 저장 중...")
    final_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n🎉 가격 DB 확장 완료!")
    print(f"📊 총 데이터: {len(final_df):,}건")
    print(f"🏢 종목 수: {final_df['ticker'].nunique():,}개")
    date_col = 'Date' if 'Date' in final_df.columns else 'date'
    print(f"📅 기간: {final_df[date_col].min().date()} ~ {final_df[date_col].max().date()}")
