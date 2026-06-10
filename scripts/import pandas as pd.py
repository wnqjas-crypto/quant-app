import sys; sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import warnings
import os

warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# 1. 절대경로 설정 (경로 오류 방지)
# ─────────────────────────────────────
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_DIR = os.path.join(BASE_DIR, 'db')

os.makedirs(DB_DIR, exist_ok=True)

FIN_DB = os.path.join(
    DB_DIR,
    'dart_bulk_db.csv'
)

PRICE_DB = os.path.join(
    DB_DIR,
    'price_db.parquet'
)

OUTPUT_FILE = os.path.join(
    DB_DIR,
    'factor_db.parquet'
)

# ─────────────────────────────────────
# 2. 기본 설정
# ─────────────────────────────────────
# 💡 [추가] 빠른 테스트를 위한 스위치 설정
TEST_MODE = False  # True로 두면 최근 2개 분기만 빠르게 테스트합니다. 실전용은 False로 바꾸세요!

LAG_DAYS = 45

SECTOR_BENCHMARKS = {

    '반도체': {
        'PER': 25,
        'POR': 18,
        'PBR': 2.5,
        'ROE': 15,
        'OPM': 20
    },

    'IT': {
        'PER': 20,
        'POR': 14,
        'PBR': 2.0,
        'ROE': 12,
        'OPM': 15
    },

    '자동차': {
        'PER': 8,
        'POR': 6,
        'PBR': 0.7,
        'ROE': 10,
        'OPM': 6
    },

    '금융': {
        'PER': 6,
        'POR': 5,
        'PBR': 0.5,
        'ROE': 8,
        'OPM': 20
    },

    '바이오': {
        'PER': 50,
        'POR': 35,
        'PBR': 3.5,
        'ROE': 5,
        'OPM': 5
    },

    '산업재': {
        'PER': 10,
        'POR': 7,
        'PBR': 0.8,
        'ROE': 8,
        'OPM': 6
    },

    '소비재': {
        'PER': 15,
        'POR': 10,
        'PBR': 1.5,
        'ROE': 12,
        'OPM': 8
    },

    '엔터/미디어': {
        'PER': 25,
        'POR': 15,
        'PBR': 2.5,
        'ROE': 10,
        'OPM': 10
    },

    '에너지/유틸리티': {
        'PER': 10,
        'POR': 8,
        'PBR': 0.9,
        'ROE': 8,
        'OPM': 8
    },

    '기본': {
        'PER': 12,
        'POR': 8,
        'PBR': 1.0,
        'ROE': 10,
        'OPM': 8
    },
}

# ─────────────────────────────────────
# 3. 유틸 함수
# ─────────────────────────────────────
def normalize(value, good, bad):

    if (
        value is None or
        pd.isna(value) or
        np.isinf(value)
    ):
        return 50

    score = (
        (value - bad)
        /
        (good - bad)
    ) * 100

    return max(
        0,
        min(100, score)
    )

# ─────────────────────────────────────
# 4. 데이터 로드
# ─────────────────────────────────────
print("\n📦 재무 DB 로드 중...")

fin_df = pd.read_csv(
    FIN_DB,
    dtype={'stock_code': str}
)

# 숫자 변환
fin_df['thstrm_amount'] = (
    fin_df['thstrm_amount']
    .astype(str)
    .str.replace(',', '')
)

fin_df['thstrm_amount'] = pd.to_numeric(
    fin_df['thstrm_amount'],
    errors='coerce'
)

print(f"✅ 재무 DB 로드 완료: {len(fin_df):,}건")

# ─────────────────────────────────────
# 종목별 분기 이력 사전 구축 (GARP·촉매 팩터용)
# ─────────────────────────────────────
print("\n📊 종목별 분기 이력 구축 중 (GARP·촉매 팩터용)...")

_IS_MASK = fin_df['account_nm'].str.contains('매출|영업이익|당기순이익', na=False, regex=True)
_is_fin  = fin_df[_IS_MASK][['stock_code', 'quarter_end', 'account_nm', 'thstrm_amount']].copy()

ticker_history: dict = {}
for (t_code, t_q), grp in _is_fin.groupby(['stock_code', 'quarter_end']):
    row_d = {'quarter_end': t_q}
    for _, r in grp.iterrows():
        nm  = str(r['account_nm'])
        val = r['thstrm_amount']
        if pd.isna(val):
            continue
        fval = float(val)
        if '매출' in nm:
            row_d.setdefault('sales', fval)
        elif '영업이익' in nm:
            row_d.setdefault('op_income', fval)
        elif '당기순이익' in nm:
            row_d.setdefault('net_income', fval)
    ticker_history.setdefault(t_code, []).append(row_d)

for t_code in ticker_history:
    ticker_history[t_code].sort(key=lambda x: x['quarter_end'])

del _IS_MASK, _is_fin
print(f"✅ {len(ticker_history):,}개 종목 이력 구축 완료")

# ─────────────────────────────────────
# 가격 DB 로드
# ─────────────────────────────────────
print("\n📦 가격 DB 로드 중...")

price_df = pd.read_parquet(PRICE_DB)

price_df['ticker'] = (
    price_df['ticker']
    .astype(str)
    .str.zfill(6)
)

# Date / date 자동 대응
if 'Date' in price_df.columns:

    price_df['date'] = pd.to_datetime(
        price_df['Date']
    )

else:

    price_df['date'] = pd.to_datetime(
        price_df['date']
    )

print(f"✅ 가격 DB 로드 완료: {len(price_df):,}건")

# 종목별 사전 분리 — 루프 내 전체 스캔(O(n)) → 딕셔너리 조회(O(1)) + 이진탐색(O(log n))
print("📊 가격 데이터 종목별 인덱싱 중...")
price_by_ticker = {
    t: grp.sort_values('date').reset_index(drop=True)
    for t, grp in price_df.groupby('ticker')
}
print(f"✅ {len(price_by_ticker):,}개 종목 인덱싱 완료")

# ─────────────────────────────────────
# KRX 정보 (섹터 매핑 기능 강화)
# ─────────────────────────────────────
print("\n🏢 KRX 정보 로드 중...")

# 1. 가격 및 시총 데이터 로드
krx = fdr.StockListing('KRX')
krx['Code'] = krx['Code'].astype(str).str.zfill(6)

# 2. 섹터(업종) 상세 데이터 로드 (KRX-DESC)
try:
    krx_desc = fdr.StockListing('KRX-DESC')
    krx_desc['Code'] = krx_desc['Code'].astype(str).str.zfill(6)
    
    # 💡 [수정 포인트 1] 기존 krx에 Sector 컬럼이 있다면 병합 시 충돌(Sector_x)을 막기 위해 미리 제거
    if 'Sector' in krx.columns:
        krx = krx.drop(columns=['Sector'])
        
    # 💡 [수정 포인트 2] 정확도 향상을 위해 Sector(업종)뿐만 아니라 Industry(주요제품) 컬럼도 가져옵니다.
    if 'Industry' in krx_desc.columns:
        krx = krx.merge(krx_desc[['Code', 'Sector', 'Industry']], on='Code', how='left')
    else:
        krx = krx.merge(krx_desc[['Code', 'Sector']], on='Code', how='left')
        
except Exception as e:
    print(f"⚠️ 상세 섹터 정보 로드 실패 (기본값으로 진행): {e}")

shares_dict = {}
sector_dict = {}

for _, row in krx.iterrows():

    code = row['Code']
    price = row.get('Close', 0)
    marcap = row.get('Marcap', 0)

    # 주식수 역산
    shares_dict[code] = (
        (marcap / price)
        if pd.notna(price) and price > 0
        else 0
    )

    # 💡 [수정 포인트 3] NaN 값 안전 처리 및 Industry 정보 결합
    sec_val = row.get('Sector', '')
    ind_val = row.get('Industry', '')
    
    sec_str = str(sec_val) if pd.notna(sec_val) else ''
    ind_str = str(ind_val) if pd.notna(ind_val) else ''
    
    # 두 정보를 합쳐서 키워드 매칭 확률을 극대화합니다.
    raw_sector = (sec_str + " " + ind_str).strip()
    
    # 회사명도 매핑에 활용 (이름에 '반도체' 포함이면 반도체 생태계로 분류)
    company_name = krx[krx['Code'] == code]['Name'].values
    cname = str(company_name[0]) if len(company_name) > 0 else ''

    r = raw_sector  # 가독성을 위한 별칭

    # ── 반도체 ──────────────────────────────────────────────
    # 삼성전자(005930): KRX Industry가 "통신 및 방송 장비"로 등록되어 IT로 빠지는 것 방지
    if ('반도체' in r or '전자부품' in r or '집적회로' in r
            or '디스플레이' in r or '반도체' in cname
            or cname == '삼성전자' or '솔브레인' in cname):
        mapped_sector = '반도체'
    # ── IT / 통신 ──────────────────────────────────────────
    elif ('소프트웨어' in r or '컴퓨터' in r or '정보통신' in r
          or '통신업' in r or '인터넷' in r or '자료처리' in r
          or '포털' in r or '통신 및 방송 장비' in r
          or '통신장비' in r or '전기통신' in r
          or '정보 서비스' in r or '측정, 시험' in r):
        mapped_sector = 'IT'
    # ── 자동차 ────────────────────────────────────────────
    elif '자동차' in r or '운송장비' in r or '모터' in r:
        mapped_sector = '자동차'
    # ── 금융 ──────────────────────────────────────────────
    elif ('은행' in r or '보험' in r or '증권' in r
          or '금융업' in r
          or '금융 지원 서비스' in r or '신탁업' in r
          or '집합투자' in r or '여신' in r or '대부업' in r):
        mapped_sector = '금융'
    # ── 바이오 / 헬스케어 ─────────────────────────────────
    elif ('의약' in r or '의료' in r or '바이오' in r or '제약' in r
          or '연구개발' in r or '기초 의약물질' in r or '의약품 제조' in r
          or '녹십자' in cname):
        mapped_sector = '바이오'
    # ── 에너지 / 유틸리티 ─────────────────────────────────
    elif ('석유 정제' in r or '원유' in r or '정유' in r
          or '연료용 가스' in r or '가스 배관' in r
          or '전기업' in r or '수력 발전' in r or '원자력' in r
          or ('가스' in cname and '메가' not in cname and '이차전지' not in r and '배터리' not in r)
          or '정유' in cname
          or cname == 'INVENI'
          or '증기, 냉' in r or '에너비스' in cname):
        mapped_sector = '에너지/유틸리티'
    # ── 엔터 / 미디어 ─────────────────────────────────────
    elif ('게임' in r or '엔터' in r or '방송프로그램' in r
          or '비디오물' in r or '미디어' in r or '콘텐츠' in r
          or '광고' in r or '영화' in r or '음악' in r or '공연' in r
          or '텔레비전 방송' in r or '창작 및 예술' in r or '오디오물' in r):
        mapped_sector = '엔터/미디어'
    # ── 소비재 ───────────────────────────────────────────
    elif ('음식' in r or '식품' in r or '음료' in r or '유통' in r
          or '의류' in r or '의복' in r or '봉제' in r
          or '섬유' in r or '화장품' in r or '생활용품' in r
          or '도소매' in r or '패션' in r
          or '소매업' in r or '종합 소매' in r
          or '가정용 기기' in r or '영상 및 음향기기' in r or '가전' in r
          or '가구' in r or '출판' in r
          or '낙농' in r or '수산물' in r or '곰물' in r
          or '가죽' in r or '담배' in r):
        mapped_sector = '소비재'
    # ── 산업재 ───────────────────────────────────────────
    elif ('조선' in r or '선박' in r or '보트 건조' in r
          or '건설' in r or '철강' in r or '기계' in r
          or '화학' in r or '에너지' in r or '전력' in r
          or '가스' in r or '금속' in r or '운송' in r
          or '물류' in r or '항공' in r
          or '전동기' in r or '발전기' in r or '전기 변환' in r
          or '절연선' in r or '케이블' in r
          or '이차전지' in r or '일차전지' in r
          or '플라스틱' in r or '고무' in r
          or '비료' in r or '농약' in r
          or '시멘트' in r or '유리' in r or '세라믹' in r
          or '펄프' in r or '종이' in r or '엔지니어링' in r
          or '전기장비' in r or '광학기기' in r
          or '무기' in r or '요업' in r):
        mapped_sector = '산업재'
    else:
        mapped_sector = '기본'

    sector_dict[code] = mapped_sector

print("✅ KRX 정보 로드 및 섹터 매핑 완료")

# ─────────────────────────────────────
# 5. 팩터 계산 시작
# ─────────────────────────────────────
print("\n🚀 팩터 DB 생성 시작...")
print("📌 실적 발표 Lag 반영 중")
print(f"📌 Lag Days: {LAG_DAYS}일\n")

all_factors = []

# 기존 팩터 DB 로드 — 이미 계산된 분기는 건너뜀
existing_quarters = set()
if os.path.exists(OUTPUT_FILE):
    try:
        existing_factor_df = pd.read_parquet(OUTPUT_FILE)
        existing_quarters = set(existing_factor_df['quarter_end'].unique())
        all_factors.extend(existing_factor_df.to_dict('records'))
        print(f"📂 기존 팩터 DB 로드: {len(existing_factor_df):,}건 ({len(existing_quarters)}분기) — 해당 분기 건너뜀")
    except Exception as e:
        print(f"⚠️ 기존 팩터 DB 로드 실패 (전체 재계산): {e}")

quarters = sorted(
    fin_df['quarter_end']
    .dropna()
    .unique()
)

# 💡 [추가] 테스트 모드일 경우 리스트의 맨 마지막 2개 분기만 잘라냅니다.
if TEST_MODE:
    quarters = quarters[-1:]
    print(f"⚠️ [TEST 모드 작동] 가장 최근 {len(quarters)}개 분기만 테스트합니다: {quarters}\n")

for q_idx, q_end in enumerate(quarters):

    print("=" * 60)
    print(
        f"[{q_idx+1}/{len(quarters)}] "
        f"{q_end}"
    )

    if q_end in existing_quarters:
        print("⏭️ 기존 데이터 있음 — 건너뜀")
        continue

    q_end_dt = pd.to_datetime(q_end)

    signal_date_dt = (
        q_end_dt
        +
        pd.Timedelta(days=LAG_DAYS)
    )

    print(
        f"📅 Signal Date: "
        f"{signal_date_dt.strftime('%Y-%m-%d')}"
    )

    prev_q_dt = (
        q_end_dt
        -
        pd.DateOffset(years=1)
    )

    prev_q_str = prev_q_dt.strftime('%Y-%m-%d')

    q_fin = fin_df[
        fin_df['quarter_end'] == q_end
    ]

    prev_fin = fin_df[
        fin_df['quarter_end'] == prev_q_str
    ]

    # 피벗
    q_pivot = q_fin.pivot_table(
        index='stock_code',
        columns='account_nm',
        values='thstrm_amount',
        aggfunc='first'
    )

    p_pivot = prev_fin.pivot_table(
        index='stock_code',
        columns='account_nm',
        values='thstrm_amount',
        aggfunc='first'
    )

    success = 0
    q_start_idx = len(all_factors)  # 이 분기 시작 인덱스 기록

    for ticker in q_pivot.index:

        try:

            # ─────────────────────────
            # 재무 데이터 추출
            # ─────────────────────────
            curr = {}

            for col in q_pivot.columns:

                val = q_pivot.loc[ticker, col]

                if pd.isna(val):
                    continue

                if '매출' in col:
                    curr.setdefault('sales', val)

                elif '영업이익' in col:
                    curr.setdefault('op_income', val)

                elif '당기순이익' in col:
                    curr.setdefault('net_income', val)

                elif '자본총계' in col:
                    curr.setdefault('equity', val)

                elif '부채총계' in col:
                    curr.setdefault('debt', val)

                elif '자산총계' in col:
                    curr.setdefault('assets', val)

            prev = {}

            if ticker in p_pivot.index:

                for col in p_pivot.columns:

                    val = p_pivot.loc[ticker, col]

                    if pd.isna(val):
                        continue

                    if '매출' in col:
                        prev.setdefault('sales', val)

                    elif '영업이익' in col:
                        prev.setdefault('op_income', val)

                    elif '당기순이익' in col:
                        prev.setdefault('net_income', val)

            # ─────────────────────────
            # 가격 데이터
            # ─────────────────────────
            t_all = price_by_ticker.get(ticker)
            if t_all is None:
                continue

            # 이진탐색으로 날짜 필터링 — 전체 스캔 대비 100배 이상 빠름
            idx = int(t_all['date'].searchsorted(signal_date_dt, side='right'))
            t_price = t_all.iloc[:idx]

            if len(t_price) < 120:
                continue

            curr_price_row = t_price.iloc[-1]

            curr_price = curr_price_row['Close']

            # ─────────────────────────
            # 시총 계산
            # ─────────────────────────
            shares = shares_dict.get(
                ticker,
                0
            )

            hist_marcap = (
                curr_price * shares
            )

            # 시총 1000억 미만 제외
            if hist_marcap < (
                1000 * 100_000_000
            ):
                continue

            # ─────────────────────────
            # 모멘텀
            # ─────────────────────────
            high52w = (
                t_price.iloc[-252:]['Close']
                .max()
                if len(t_price) >= 252
                else t_price['Close'].max()
            )

            high52w_pct = (
                (curr_price / high52w) * 100
                if high52w
                else 50
            )

            mom_6m = (
                (
                    curr_price
                    /
                    t_price.iloc[-120]['Close']
                ) - 1
            ) * 100

            # ─────────────────────────
            # 변동성
            # ─────────────────────────
            returns = (
                t_price.iloc[-252:]['Close']
                .pct_change()
                .dropna()
            )

            volatility = (
                returns.std()
                *
                np.sqrt(252)
                if not returns.empty
                else 0.5
            )

            # ─────────────────────────
            # 벤치마크
            # ─────────────────────────
            bench = SECTOR_BENCHMARKS.get(
                sector_dict.get(
                    ticker,
                    '기타'
                ),
                SECTOR_BENCHMARKS['기본']
            )

            # ─────────────────────────
            # 성장성
            # ─────────────────────────
            sales_yoy = (
                (
                    curr.get('sales', 0)
                    -
                    prev.get('sales', 0)
                )
                /
                abs(prev.get('sales', 1))
            ) * 100 if prev.get('sales') else 0

            op_yoy = (
                (
                    curr.get('op_income', 0)
                    -
                    prev.get('op_income', 0)
                )
                /
                abs(prev.get('op_income', 1))
            ) * 100 if prev.get('op_income') else 0

            ni_yoy = (
                (
                    curr.get('net_income', 0)
                    -
                    prev.get('net_income', 0)
                )
                /
                abs(prev.get('net_income', 1))
            ) * 100 if prev.get('net_income') else 0

            # 영업이익 성장 40% 반영 — 순이익보다 일회성 요인 적고 안정적
            # 상한 확대: 반도체 등 고성장 기업 차별화 (구: sales 50, op/ni 80)
            growth_score = (
                normalize(sales_yoy, 150, -30) * 0.30
                + normalize(op_yoy,  300, -50) * 0.40
                + normalize(ni_yoy,  300, -50) * 0.30
            )

            # ─────────────────────────
            # 가치 (GARP: 싸면서 돈도 버는 기업)
            # ─────────────────────────
            per = (
                hist_marcap
                /
                (curr['net_income'] * 4)
            ) if curr.get(
                'net_income',
                0
            ) > 0 else None

            # P/영업이익: 순이익보다 조작 어렵고 안정적
            por = (
                hist_marcap
                /
                (curr['op_income'] * 4)
            ) if curr.get(
                'op_income',
                0
            ) > 0 else None

            # 적자 기업은 PER/POR 점수 0 (중립 50이 아닌 패널티)
            if curr.get('net_income', 0) <= 0:
                per_score = 0
            else:
                per_score = normalize(
                    per,
                    bench['PER'] * 0.4,
                    bench['PER'] * 3.0
                )

            if curr.get('op_income', 0) <= 0:
                por_score = 0
            else:
                por_score = normalize(
                    por,
                    bench['POR'] * 0.4,
                    bench['POR'] * 3.0
                )

            # ROE 필터: 수익성 없는 저평가는 가치주가 아님
            roe_for_value = (
                (curr['net_income'] / curr['equity']) * 400
            ) if (
                curr.get('equity', 0) > 0 and
                curr.get('net_income', 0) > 0
            ) else None

            roe_value_score = normalize(
                roe_for_value,
                bench['ROE'] * 2.0,
                0
            )

            # PBR 제거: 자산 기반 → 이익 기반으로 전환 (가치함정 방지)
            value_score = (
                per_score       * 0.35
                + por_score     * 0.30
                + roe_value_score * 0.35
            )

            # ─────────────────────────
            # 퀄리티
            # ─────────────────────────
            roe = (
                (
                    curr['net_income']
                    /
                    curr['equity']
                ) * 400
            ) if (
                curr.get('equity', 0) > 0
                and curr.get('net_income')
            ) else None

            debt_ratio = (
                (
                    curr['debt']
                    /
                    curr['equity']
                ) * 100
            ) if (
                curr.get('equity', 0) > 0
                and curr.get('debt')
            ) else None

            # 영업이익 플러스 여부: 수익성 안정성의 기본 조건
            op_consistency = 100.0 if curr.get('op_income', 0) > 0 else 0.0

            quality_score = (
                normalize(roe,        bench['ROE'] * 3, 0) * 0.60
                + normalize(debt_ratio, 30, 300)           * 0.25
                + op_consistency                            * 0.15
            )

            # ─────────────────────────
            # 수익성
            # ─────────────────────────
            roa = (
                (
                    curr['op_income']
                    /
                    curr['assets']
                ) * 100
            ) if (
                curr.get('assets', 0) > 0
                and curr.get('op_income')
            ) else None

            opm = (
                (curr['op_income'] / curr['sales']) * 100
            ) if (
                curr.get('sales', 0) > 0
                and curr.get('op_income')
            ) else None

            profitability_score = (
                normalize(roa, 10, 0) * 0.60
                + normalize(opm, bench['OPM'], 0) * 0.40
            )

            # ─────────────────────────
            # 모멘텀 점수
            # ─────────────────────────
            momentum_score = (
                normalize(
                    mom_6m,
                    60,
                    -30
                ) * 0.7
                +
                normalize(
                    high52w_pct,
                    100,
                    60
                ) * 0.3
            )

            # ─────────────────────────
            # 저변동성
            # ─────────────────────────
            lowvol_score = normalize(
                volatility,
                0.15,
                0.60
            )

            # ─────────────────────────
            # GARP + 촉매 팩터
            # ─────────────────────────
            garp_score     = 50.0  # 데이터 부족 시 중립값
            catalyst_score = 50.0

            t_hist     = ticker_history.get(ticker, [])
            t_hist_q   = [h for h in t_hist if h['quarter_end'] <= q_end]

            if len(t_hist_q) >= 2:
                curr_h    = t_hist_q[-1]
                prev_h    = t_hist_q[-2]
                curr_op   = curr_h.get('op_income',  0) or 0
                prev_q_op = prev_h.get('op_income',  0) or 0
                curr_sl   = curr_h.get('sales',       0) or 0
                prev_q_sl = prev_h.get('sales',       0) or 0

                # ── GARP ────────────────────────────────────────
                # ① 최근 4분기 영업이익 전부 흑자
                if len(t_hist_q) >= 4:
                    _g4 = 1 if all((h.get('op_income', 0) or 0) > 0 for h in t_hist_q[-4:]) else 0
                else:
                    _g4 = 1 if curr_op > 0 else 0

                # ② 이익 가속: 현재 YoY > 직전분기 YoY
                _prev_q_end  = prev_h['quarter_end']
                _prev_1y_end = (pd.to_datetime(_prev_q_end) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
                _prev_1y_lst = [h for h in t_hist if h['quarter_end'] == _prev_1y_end]
                _g_accel = 0
                if _prev_1y_lst:
                    _p1y_op = _prev_1y_lst[0].get('op_income', 0) or 0
                    if _p1y_op != 0:
                        _op_yoy_prev = (prev_q_op - _p1y_op) / abs(_p1y_op) * 100
                        _g_accel = 1 if (op_yoy > 0 and op_yoy > _op_yoy_prev) else 0

                # ③ 매출 YoY 양 (이미 계산된 sales_yoy 활용)
                _g_rev = 1 if sales_yoy > 0 else 0

                # ④ PEG (per·op_yoy 활용)
                _g_peg = _g_peg_bonus = 0
                if per and per > 0 and op_yoy > 0:
                    _peg = per / op_yoy
                    _g_peg       = 1 if _peg <= 1.5 else 0
                    _g_peg_bonus = 1 if _peg <= 1.0 else 0

                # ⑤ ROE 대리: 최근 4Q 순이익 합계 양수
                if len(t_hist_q) >= 4:
                    _net4q = sum((h.get('net_income', 0) or 0) for h in t_hist_q[-4:])
                    _g_roe = 1 if _net4q > 0 else 0
                else:
                    _g_roe = 1 if (curr_h.get('net_income', 0) or 0) > 0 else 0

                _raw_garp  = _g4 + _g_accel + _g_rev + _g_peg + _g_peg_bonus + _g_roe
                garp_score = round(_raw_garp / 6 * 100, 1)

                # ── 촉매 ────────────────────────────────────────
                # ① 어닝 서프라이즈: 현재 분기 영업이익 > 직전 4분기 평균 +30%
                _c_surprise = 0
                if len(t_hist_q) >= 5:
                    _prior4_ops = [(h.get('op_income', 0) or 0) for h in t_hist_q[-5:-1]]
                    _avg_prior  = np.mean(_prior4_ops)
                    if _avg_prior > 0:
                        _c_surprise = 1 if (curr_op - _avg_prior) / abs(_avg_prior) * 100 >= 30 else 0

                # ② 마진 변곡점: OPM QoQ 개선 + YoY 개선
                _c_margin = 0
                if curr_sl > 0 and prev_q_sl > 0:
                    _curr_opm   = curr_op   / curr_sl
                    _prev_q_opm = prev_q_op / prev_q_sl
                    if _curr_opm > _prev_q_opm:
                        _curr_1y_end = (pd.to_datetime(q_end) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
                        _curr_1y_lst = [h for h in t_hist if h['quarter_end'] == _curr_1y_end]
                        if _curr_1y_lst and (_curr_1y_lst[0].get('sales', 0) or 0) > 0:
                            _1y_opm = (_curr_1y_lst[0].get('op_income', 0) or 0) / _curr_1y_lst[0]['sales']
                            _c_margin = 1 if _curr_opm > _1y_opm else 0
                        else:
                            _c_margin = 1  # QoQ 개선만으로도 인정

                # ③ 턴어라운드: 직전분기 적자 → 현재 흑자
                _c_turn = 1 if (prev_q_op < 0 and curr_op > 0) else 0

                _raw_catalyst  = _c_surprise + _c_margin + _c_turn
                catalyst_score = round(_raw_catalyst / 3 * 100, 1)

            # ─────────────────────────
            # 자본잠식 패널티
            # paid_in_capital 미보유 → 부채비율 400% 초과를 부분잠식 대리지표로 사용
            # ─────────────────────────
            _equity = curr.get('equity', 0)
            _debt   = curr.get('debt', 0)

            if _equity <= 0:
                # 완전자본잠식: 모든 팩터 대폭 하향 → 사실상 선택 불가
                _impairment = 0.3
            elif _equity > 0 and _debt > 0 and (_debt / _equity) > 4.0:
                # 부채비율 400% 초과: 재무 위기 신호 → 완만한 패널티
                _impairment = 0.7
            else:
                _impairment = 1.0

            if _impairment < 1.0:
                growth_score        *= _impairment
                value_score         *= _impairment
                quality_score       *= _impairment
                profitability_score *= _impairment
                momentum_score      *= _impairment
                lowvol_score        *= _impairment
                garp_score          *= _impairment
                catalyst_score      *= _impairment

            # ─────────────────────────
            # 저장
            # ─────────────────────────
            def _r(v, d=1):
                """None/NaN/inf 안전 반올림"""
                try:
                    if v is None or np.isnan(v) or np.isinf(v):
                        return None
                    return round(float(v), d)
                except Exception:
                    return None

            all_factors.append({
                'quarter_end':  q_end,
                'signal_date':  signal_date_dt.strftime('%Y-%m-%d'),
                'ticker':       ticker,
                'sector':       sector_dict.get(ticker, '기타'),

                # ── 팩터 점수 ──────────────────────────────
                'growth_score':        round(growth_score, 1),
                'value_score':         round(value_score, 1),
                'quality_score':       round(quality_score, 1),
                'profitability_score': round(profitability_score, 1),
                'momentum_score':      round(momentum_score, 1),
                'lowvol_score':        round(lowvol_score, 1),
                'garp_score':          round(garp_score, 1),
                'catalyst_score':      round(catalyst_score, 1),

                # ── 원본 수치 (추천 이유 생성용) ────────────
                'sales_yoy':   _r(sales_yoy),
                'op_yoy':      _r(op_yoy),
                'ni_yoy':      _r(ni_yoy),
                'per':         _r(per),
                'por':         _r(por),
                'roe':         _r(roe),
                'debt_ratio':  _r(debt_ratio),
                'roa':         _r(roa),
                'opm':         _r(opm),
                'mom_6m':      _r(mom_6m),
                'volatility':  _r(volatility * 100),   # % 단위
            })

            success += 1

        except:
            continue

    print(f"✅ 완료: {success:,}개 종목")

    # 저변동성 퍼센타일 재정규화 — 절대 변동성 기준 대신 이 분기 유니버스 내 상대 순위
    # (변동성이 낮을수록 lowvol_score 높음 → rank를 뒤집어서 적용)
    q_new = all_factors[q_start_idx:]
    if q_new:
        lowvol_vals = pd.Series([f['lowvol_score'] for f in q_new])
        # 높은 점수 = 낮은 변동성이므로 ascending rank 그대로 사용
        percentiles = lowvol_vals.rank(pct=True, method='average') * 100
        for i, f in enumerate(q_new):
            f['lowvol_score'] = round(float(percentiles.iloc[i]), 1)

# ─────────────────────────────────────
# 6. 저장
# ─────────────────────────────────────
print("\n📦 최종 병합 중...")

factor_df = pd.DataFrame(all_factors)

factor_df = factor_df.drop_duplicates(
    subset=[
        'ticker',
        'quarter_end'
    ],
    keep='last'
)

print("\n💾 저장 중...")

factor_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

# ─────────────────────────────────────
# 완료
# ─────────────────────────────────────
print("\n🎉 factor DB 생성 완료!")

print(f"\n📁 저장 위치:")
print(OUTPUT_FILE)

print(f"\n📊 총 데이터:")
print(f"{len(factor_df):,}건")

print(f"\n🏢 종목 수:")
print(f"{factor_df['ticker'].nunique():,}개")