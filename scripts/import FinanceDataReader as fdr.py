import FinanceDataReader as fdr
import pandas as pd
import os
import warnings
import time

warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# 절대경로 설정
# ─────────────────────────────────────
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_DIR = os.path.join(BASE_DIR, 'db')

os.makedirs(DB_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    DB_DIR,
    'price_db.parquet'
)

# ─────────────────────────────────────
# 저장 기간
# ─────────────────────────────────────
START_DATE = '2020-01-01'
END_DATE   = '2026-06-30'

# ─────────────────────────────────────
# KRX 종목 리스트
# ─────────────────────────────────────
print("\n📦 KRX 종목 리스트 로드 중...")

krx = fdr.StockListing('KRX')

krx['Code'] = krx['Code'].astype(str).str.zfill(6)

tickers = krx['Code'].tolist()

# 테스트용
#tickers = tickers[:30]

print(f"✅ 총 {len(tickers):,}개 종목")

# ─────────────────────────────────────
# 기존 DB 이어받기
# ─────────────────────────────────────
all_data = []

existing_tickers = set()

if os.path.exists(OUTPUT_FILE):

    try:

        old_df = pd.read_parquet(OUTPUT_FILE)

        all_data.append(old_df)

        existing_tickers = set(
            old_df['ticker'].unique()
        )

        print(
            f"📂 기존 DB 발견 "
            f"({len(existing_tickers):,}개 종목)"
        )

    except Exception as e:

        print(f"⚠️ 기존 DB 로드 실패: {e}")

# ─────────────────────────────────────
# 가격 데이터 수집
# ─────────────────────────────────────
print("\n🚀 가격 데이터 수집 시작...\n")

success = 0
failed = 0

for idx, ticker in enumerate(tickers):

    if ticker in existing_tickers:
        continue

    try:

        df = fdr.DataReader(
            ticker,
            START_DATE,
            END_DATE
        )

        if df.empty:
            failed += 1
            continue

        df = df.reset_index()

        df['ticker'] = ticker

        # 컬럼 정리
        keep_cols = [
            'Date',
            'ticker',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
        ]

        safe_cols = [
            c for c in keep_cols
            if c in df.columns
        ]

        df = df[safe_cols]

        all_data.append(df)

        success += 1

        # 진행상황 출력
        if success % 100 == 0:

            print(
                f"✅ 성공:{success:,} "
                f"❌ 실패:{failed:,} "
                f"진행:{idx+1:,}/{len(tickers):,}"
            )

    except Exception:

        failed += 1

    # 서버 과부하 방지
    time.sleep(0.05)

# ─────────────────────────────────────
# 병합
# ─────────────────────────────────────
print("\n📦 병합 중...")

final_df = pd.concat(
    all_data,
    ignore_index=True
)

# 날짜 변환
final_df['Date'] = pd.to_datetime(
    final_df['Date']
)

# 정렬
final_df = final_df.sort_values(
    ['ticker', 'Date']
)

# 중복 제거
final_df = final_df.drop_duplicates(
    subset=['ticker', 'Date'],
    keep='last'
)

# ─────────────────────────────────────
# 저장
# ─────────────────────────────────────
print("\n💾 저장 중...")

final_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

# ─────────────────────────────────────
# 완료
# ─────────────────────────────────────
print("\n🎉 가격 DB 생성 완료!")
print(f"📁 저장 위치:")
print(OUTPUT_FILE)

print(f"\n📊 총 데이터:")
print(f"{len(final_df):,}건")

print(f"\n🏢 종목 수:")
print(f"{final_df['ticker'].nunique():,}개")