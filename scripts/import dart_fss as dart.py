import dart_fss as dart
import FinanceDataReader as fdr
import requests
import pandas as pd
import time
import os
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
# 경로 설정 (절대경로)
# ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_DIR = os.path.join(BASE_DIR, 'db')

os.makedirs(DB_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(
    DB_DIR,
    'dart_bulk_db.csv'
)

# ─────────────────────────────────────────────────────
# DART API
# ─────────────────────────────────────────────────────
DART_API_KEY = "9fb94f827cede50a21efe6faa33d227081f655dc"

dart.set_api_key(DART_API_KEY)

# ─────────────────────────────────────────────────────
# 전체 기간 설정
# 2015Q1 ~ 2026Q1
# ─────────────────────────────────────────────────────
QUARTERS = {

    '2015-03-31': ('2015', '11013'),
    '2015-06-30': ('2015', '11012'),
    '2015-09-30': ('2015', '11014'),
    '2015-12-31': ('2015', '11011'),

    '2016-03-31': ('2016', '11013'),
    '2016-06-30': ('2016', '11012'),
    '2016-09-30': ('2016', '11014'),
    '2016-12-31': ('2016', '11011'),

    '2017-03-31': ('2017', '11013'),
    '2017-06-30': ('2017', '11012'),
    '2017-09-30': ('2017', '11014'),
    '2017-12-31': ('2017', '11011'),

    '2018-03-31': ('2018', '11013'),
    '2018-06-30': ('2018', '11012'),
    '2018-09-30': ('2018', '11014'),
    '2018-12-31': ('2018', '11011'),

    '2019-03-31': ('2019', '11013'),
    '2019-06-30': ('2019', '11012'),
    '2019-09-30': ('2019', '11014'),
    '2019-12-31': ('2019', '11011'),

    '2020-03-31': ('2020', '11013'),
    '2020-06-30': ('2020', '11012'),
    '2020-09-30': ('2020', '11014'),
    '2020-12-31': ('2020', '11011'),

    '2021-03-31': ('2021', '11013'),
    '2021-06-30': ('2021', '11012'),
    '2021-09-30': ('2021', '11014'),
    '2021-12-31': ('2021', '11011'),

    '2022-03-31': ('2022', '11013'),
    '2022-06-30': ('2022', '11012'),
    '2022-09-30': ('2022', '11014'),
    '2022-12-31': ('2022', '11011'),

    '2023-03-31': ('2023', '11013'),
    '2023-06-30': ('2023', '11012'),
    '2023-09-30': ('2023', '11014'),
    '2023-12-31': ('2023', '11011'),

    '2024-03-31': ('2024', '11013'),
    '2024-06-30': ('2024', '11012'),
    '2024-09-30': ('2024', '11014'),
    '2024-12-31': ('2024', '11011'),

    '2025-03-31': ('2025', '11013'),
    '2025-06-30': ('2025', '11012'),
    '2025-09-30': ('2025', '11014'),
    '2025-12-31': ('2025', '11011'),

    '2026-03-31': ('2026', '11013'),
}

# ─────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────
def chunker(seq, size):
    return (
        seq[pos:pos + size]
        for pos in range(0, len(seq), size)
    )

# ─────────────────────────────────────────────────────
# 메인 시작
# ─────────────────────────────────────────────────────
print("\n📦 상장사 목록 로드 중...")

krx = fdr.StockListing('KRX')
tickers = krx['Code'].astype(str).tolist()

corp_list = dart.get_corp_list()

valid_corps = []

for ticker in tickers:

    try:
        c = corp_list.find_by_stock_code(ticker)

        if c:

            corp = c if not isinstance(c, list) else c[0]

            valid_corps.append({
                'stock_code': ticker,
                'corp_code': corp.corp_code,
                'corp_name': corp.corp_name
            })

    except:
        continue

corp_df = pd.DataFrame(valid_corps)

print(f"✅ 매핑 완료: {len(corp_df):,}개 기업")

# ─────────────────────────────────────────────────────
# 기존 DB 불러오기
# ─────────────────────────────────────────────────────
all_data = []

if os.path.exists(OUTPUT_FILE):

    try:

        old_df = pd.read_csv(
            OUTPUT_FILE,
            dtype={'stock_code': str}
        )

        all_data.append(old_df)

        print(f"📂 기존 DB 로드 완료: {len(old_df):,}건")

    except Exception as e:

        print(f"⚠️ 기존 DB 로드 실패: {e}")

# ─────────────────────────────────────────────────────
# 재무 데이터 수집
# ─────────────────────────────────────────────────────
print("\n🚀 DART 재무데이터 수집 시작...\n")

for q_end, (year, reprt_code) in QUARTERS.items():

    print("\n" + "=" * 60)
    print(f"📊 {q_end} 수집 시작")
    print("=" * 60)

    chunks = list(
        chunker(
            corp_df.to_dict('records'),
            10
        )
    )

    quarter_data = []

    for idx, chunk in enumerate(chunks):

        corp_codes = [
            x['corp_code']
            for x in chunk
        ]

        corp_code_str = ",".join(corp_codes)

        url = (
            f"https://opendart.fss.or.kr/api/fnlttMultiAcnt.json"
            f"?crtfc_key={DART_API_KEY}"
            f"&corp_code={corp_code_str}"
            f"&bsns_year={year}"
            f"&reprt_code={reprt_code}"
        )

        try:

            res = requests.get(
                url,
                timeout=30
            ).json()

            status = res.get('status')

            if status == '000':

                df = pd.DataFrame(res['list'])

                if not df.empty:

                    df['quarter_end'] = q_end

                    quarter_data.append(df)

            elif status == '013':
                pass

            else:

                print(
                    f"⚠️ [{idx+1}/{len(chunks)}] "
                    f"{res.get('message')}"
                )

        except Exception as e:

            print(
                f"⚠️ [{idx+1}/{len(chunks)}] "
                f"예외 발생: {e}"
            )

        # 진행률 출력
        if (idx + 1) % 20 == 0:

            print(
                f"진행률: "
                f"{idx+1}/{len(chunks)}"
            )

        time.sleep(0.3)

    # ─────────────────────────────────────
    # 분기 데이터 정리
    # ─────────────────────────────────────
    if quarter_data:

        print("\n📦 분기 데이터 병합 중...")

        q_df = pd.concat(
            quarter_data,
            ignore_index=True
        )

        # 필요한 컬럼만
        target_cols = [
            'quarter_end',
            'stock_code',
            'corp_code',
            'corp_name',
            'fs_div',
            'sj_div',
            'account_nm',
            'thstrm_amount',
            'frmtrm_amount',
            'ord',
        ]

        safe_cols = [
            c for c in target_cols
            if c in q_df.columns
        ]

        q_df = q_df[safe_cols]

        # 숫자 변환
        if 'thstrm_amount' in q_df.columns:

            q_df['thstrm_amount'] = (
                q_df['thstrm_amount']
                .astype(str)
                .str.replace(',', '')
            )

            q_df['thstrm_amount'] = pd.to_numeric(
                q_df['thstrm_amount'],
                errors='coerce'
            )

        # 임시 저장
        temp_file = os.path.join(
            DB_DIR,
            f"temp_{q_end}.csv"
        )

        q_df.to_csv(
            temp_file,
            index=False,
            encoding='utf-8-sig'
        )

        print(f"💾 임시 저장 완료")
        print(f"   {temp_file}")

        all_data.append(q_df)

        print(f"✅ 수집 완료: {len(q_df):,}건")

    else:

        print("❌ 수집 데이터 없음")

# ─────────────────────────────────────────────────────
# 최종 병합
# ─────────────────────────────────────────────────────
print("\n📦 최종 병합 중...")

final_df = pd.concat(
    all_data,
    ignore_index=True
)

# stock_code 없으면 생성
if 'stock_code' not in final_df.columns:

    if 'corp_code' in final_df.columns:

        mapper = corp_df.set_index(
            'corp_code'
        )['stock_code'].to_dict()

        final_df['stock_code'] = (
            final_df['corp_code']
            .map(mapper)
        )

# 중복 제거
subset_cols = [
    c for c in [
        'stock_code',
        'quarter_end',
        'fs_div',
        'account_nm'
    ]
    if c in final_df.columns
]

final_df = final_df.drop_duplicates(
    subset=subset_cols,
    keep='last'
)

# ─────────────────────────────────────────────────────
# 최종 저장
# ─────────────────────────────────────────────────────
print("\n💾 최종 CSV 저장 중...")

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding='utf-8-sig'
)

print("\n🎉 완료!")
print(f"저장 위치:")
print(OUTPUT_FILE)

print(f"\n총 데이터:")
print(f"{len(final_df):,}건")

print("\n수집 분기:")
print(sorted(final_df['quarter_end'].unique()))