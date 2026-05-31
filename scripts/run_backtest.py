import sys; sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import FinanceDataReader as fdr

warnings.filterwarnings('ignore')

# =========================================================
# 1. 경로 설정
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, '..', 'db')
RESULT_DIR = os.path.join(BASE_DIR, '..', 'results')

os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

FACTOR_DB = os.path.join(DB_DIR, 'factor_db.parquet')
PRICE_DB  = os.path.join(DB_DIR, 'price_db.parquet')

# =========================================================
# 2. 백테스트 설정
# =========================================================
TOP_N = 20  # 기본값 (유동성 필터 통과 여부 체크에 사용)
STRATEGY_TOP_N = {
    '밸런스형':   20,
    '저평가형':   20,
    '모멘텀형':   15,
    '국면대응형': 20,
}

# 버퍼 존: 포트폴리오에 이미 있는 종목은 TOP_N + BUFFER 밖으로 밀려야 교체
# USE_BUFFER = False 로 설정 시 기존 방식(매 분기 완전 교체)
USE_BUFFER = True
STRATEGY_BUFFER = {
    '밸런스형':   5,   # Top 25 밖이어야 퇴출
    '저평가형':   10,  # Top 30 밖이어야 퇴출 (장투 성격)
    '모멘텀형':   3,   # Top 18 밖이어야 퇴출 (모멘텀은 빠르게 교체)
    '국면대응형': 5,   # Top 25 밖이어야 퇴출
}
MIN_TRADING_VALUE = 500_000_000  # 일평균 거래대금 5억원 (최근 20거래일 기준)
# 시총 필터는 팩터 빌더(import pandas as pd.py)에서 적용됨 — 1000억 미만 제외
TRANSACTION_COST = 0.005  # 중형주 기준 0.5% (대형주 0.3% / 중형주 0.5% 평균 적용)
MAX_PER_SECTOR = 4  # 섹터별 최대 편입 종목 수 (모멘텀형 제외)
RISK_FREE_RATE_QTR = 0.035 / 4  # 연 3.5% 국고채 기준 → 분기 환산

QUARTERS = [
    ('2015-01-01', '2015-03-31'),
    ('2015-04-01', '2015-06-30'),
    ('2015-07-01', '2015-09-30'),
    ('2015-10-01', '2015-12-31'),

    ('2016-01-01', '2016-03-31'),
    ('2016-04-01', '2016-06-30'),
    ('2016-07-01', '2016-09-30'),
    ('2016-10-01', '2016-12-31'),

    ('2017-01-01', '2017-03-31'),
    ('2017-04-01', '2017-06-30'),
    ('2017-07-01', '2017-09-30'),
    ('2017-10-01', '2017-12-31'),

    ('2018-01-01', '2018-03-31'),
    ('2018-04-01', '2018-06-30'),
    ('2018-07-01', '2018-09-30'),
    ('2018-10-01', '2018-12-31'),

    ('2019-01-01', '2019-03-31'),
    ('2019-04-01', '2019-06-30'),
    ('2019-07-01', '2019-09-30'),
    ('2019-10-01', '2019-12-31'),

    ('2020-01-01', '2020-03-31'),
    ('2020-04-01', '2020-06-30'),
    ('2020-07-01', '2020-09-30'),
    ('2020-10-01', '2020-12-31'),

    ('2021-01-01', '2021-03-31'),
    ('2021-04-01', '2021-06-30'),
    ('2021-07-01', '2021-09-30'),
    ('2021-10-01', '2021-12-31'),

    ('2022-01-01', '2022-03-31'),
    ('2022-04-01', '2022-06-30'),
    ('2022-07-01', '2022-09-30'),
    ('2022-10-01', '2022-12-31'),

    ('2023-01-01', '2023-03-31'),
    ('2023-04-01', '2023-06-30'),
    ('2023-07-01', '2023-09-30'),
    ('2023-10-01', '2023-12-31'),

    ('2024-01-01', '2024-03-31'),
    ('2024-04-01', '2024-06-30'),
    ('2024-07-01', '2024-09-30'),
    ('2024-10-01', '2024-12-31'),

    ('2025-01-01', '2025-03-31'),
    ('2025-04-01', '2025-06-30'),
    ('2025-07-01', '2025-09-30'),
    ('2025-10-01', '2025-12-31'),

    ('2026-01-01', '2026-03-31'),
    ('2026-04-01', '2026-06-30'),
]

# =========================================================
# 3. 전략 가중치
# =========================================================

# -------------------------------------
# 밸런스형
# 올라운드 전략
# -------------------------------------
BALANCED_WEIGHTS = {
    'growth_score':       0.20,
    'value_score':        0.20,
    'quality_score':      0.20,
    'profitability_score':0.15,
    'momentum_score':     0.15,
    'lowvol_score':       0.10,
}

# -------------------------------------
# 저평가형
# 가치 + 방어
# -------------------------------------
VALUE_WEIGHTS = {
    'growth_score':       0.10,
    'value_score':        0.35,
    'quality_score':      0.20,
    'profitability_score':0.25,
    'momentum_score':     0.00,
    'lowvol_score':       0.10,
}

# -------------------------------------
# 모멘텀형
# CANSLIM 느낌 + 최소 방어장치
# -------------------------------------
MOMENTUM_WEIGHTS = {
    'growth_score':       0.25,
    'value_score':        0.00,
    'quality_score':      0.10,
    'profitability_score':0.10,
    'momentum_score':     0.45,
    'lowvol_score':       0.10,
}

# -------------------------------------
# 국면대응형 - 강세장
# -------------------------------------
BULL_WEIGHTS = {
    'growth_score':       0.25,
    'value_score':        0.05,
    'quality_score':      0.15,
    'profitability_score':0.15,
    'momentum_score':     0.30,
    'lowvol_score':       0.10,
}

# -------------------------------------
# 국면대응형 - 횡보장
# MA200 ±3% 이내: 방향성 불명확 → 퀄리티 중심
# -------------------------------------
SIDEWAYS_WEIGHTS = {
    'growth_score':       0.15,
    'value_score':        0.20,
    'quality_score':      0.25,
    'profitability_score':0.15,
    'momentum_score':     0.15,
    'lowvol_score':       0.10,
}

# -------------------------------------
# 국면대응형 - 약세장
# -------------------------------------
BEAR_WEIGHTS = {
    'growth_score':       0.05,
    'value_score':        0.40,
    'quality_score':      0.25,
    'profitability_score':0.20,
    'momentum_score':     0.00,
    'lowvol_score':       0.10,
}

# =========================================================
# 4. 데이터 로드
# =========================================================
print("📦 DB 로드 중...")

factor_df = pd.read_parquet(FACTOR_DB)
price_df  = pd.read_parquet(PRICE_DB)

factor_df['ticker'] = factor_df['ticker'].astype(str).str.zfill(6)
price_df['ticker']  = price_df['ticker'].astype(str).str.zfill(6)

# 💡 [핵심 해결책] 가격 DB의 모든 컬럼명을 소문자로 싹 통일합니다!
# 이렇게 하면 'Date'는 'date'로, 'Close'는 'close'로 자동 변환되어 에러가 원천 차단됩니다.
price_df.columns = [str(c).lower() for c in price_df.columns]

# 이제 소문자 'date'로 안심하고 접근!
price_df['date'] = pd.to_datetime(price_df['date'])

print(f"✅ 팩터 DB: {len(factor_df):,}건")
print(f"✅ 가격 DB : {len(price_df):,}건")

print("📊 가격 데이터 종목별 인덱싱 중...")
price_by_ticker = {
    t: grp.sort_values('date').reset_index(drop=True)
    for t, grp in price_df.groupby('ticker')
}
print(f"✅ {len(price_by_ticker):,}개 종목 인덱싱 완료")

# =========================================================
# 5. 가격 조회 함수
# =========================================================

def get_avg_trading_value(ticker, as_of_date, days=20):
    """매수일 기준 최근 N거래일 일평균 거래대금 (원 단위)"""
    t_all = price_by_ticker.get(ticker)
    if t_all is None:
        return 0
    idx = int(t_all['date'].searchsorted(pd.to_datetime(as_of_date), side='right'))
    recent = t_all.iloc[max(0, idx - days):idx]
    if recent.empty:
        return 0
    return (recent['close'] * recent['volume']).mean()
def get_price_return(ticker, start_date, end_date):

    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)
    holding_days = max((end_dt - start_dt).days, 1)

    t_all = price_by_ticker.get(ticker)
    if t_all is None:
        return None

    # 이진탐색으로 날짜 범위 슬라이싱 — 전체 스캔 대비 100배 이상 빠름
    i_start = int(t_all['date'].searchsorted(start_dt, side='left'))
    i_end   = int(t_all['date'].searchsorted(end_dt,   side='right'))
    df = t_all.iloc[i_start:i_end]

    if df.empty:
        return None

    start_price = df.iloc[0]['close']
    if start_price <= 0:
        return None

    last_price  = df.iloc[-1]['close']
    base_ret    = ((last_price / start_price) - 1) * 100
    actual_days = (df.iloc[-1]['date'] - start_dt).days
    coverage    = actual_days / holding_days

    if coverage >= 0.9:
        return base_ret
    elif coverage >= 0.3:
        return base_ret - 20.0
    else:
        return -50.0

# =========================================================
# 6. 시장 국면 판별
# 200일 이동평균 사용
# =========================================================
print("\n📈 코스피 데이터 로드 중...")

kospi = fdr.DataReader(
    'KS11',
    '2014-01-01',
    '2026-12-31'
)

kospi['MA200'] = kospi['Close'].rolling(200).mean()

def get_market_regime(date):
    """200MA 기준 3단계 시장 국면 판별
    bull    : Close > MA200 * 1.03
    sideways: MA200 * 0.97 <= Close <= MA200 * 1.03
    bear    : Close < MA200 * 0.97
    """
    dt = pd.to_datetime(date)
    sub = kospi[kospi.index <= dt]

    if len(sub) == 0 or pd.isna(sub.iloc[-1]['MA200']):
        return 'bull'

    last = sub.iloc[-1]
    close, ma200 = last['Close'], last['MA200']

    if close > ma200 * 1.03:
        return 'bull'
    elif close < ma200 * 0.97:
        return 'bear'
    else:
        return 'sideways'

# =========================================================
# 7. 점수 계산 함수
# =========================================================
def calc_score(row, weights):

    total = 0

    for factor, weight in weights.items():
        total += row.get(factor, 50) * weight

    return round(total, 2)

# =========================================================
# 8. 메인 백테스트
# =========================================================
print("\n🚀 백테스트 시작")

def select_with_sector_cap(df, score_col, top_n, max_per_sector):
    """점수 순위대로 뽑되 섹터당 max_per_sector개 상한"""
    sorted_df = df.sort_values(score_col, ascending=False)
    selected, counts = [], {}
    for _, row in sorted_df.iterrows():
        s = row['sector']
        if counts.get(s, 0) < max_per_sector:
            selected.append(row['ticker'])
            counts[s] = counts.get(s, 0) + 1
        if len(selected) >= top_n:
            break
    return selected

results = []
# 버퍼 모드: 전략별 이전 분기 보유 종목 추적
prev_holdings: dict = {strat: set() for strat in STRATEGY_TOP_N}

for i in range(len(QUARTERS) - 1):

    q_start, q_end = QUARTERS[i]
    next_start, next_end = QUARTERS[i + 1]

    print(f"\n{'='*60}")
    print(f"[{i+1}/{len(QUARTERS)-1}] {q_start} ~ {q_end}")

    # ---------------------------------------------
    # 현재 분기 팩터
    # ---------------------------------------------
    q_factor = factor_df[
        factor_df['quarter_end'] == q_end
    ].copy()

    if q_factor.empty:
        print("❌ 팩터 없음")
        continue

    # 💡 수정포인트 2: 팩터 DB에서 실제 매매일(signal_date) 추출
    sig_date = q_factor.iloc[0]['signal_date']
    
    # 다음 분기 매도일 추출 (마지막 분기 대비 안전장치 포함)
    next_factor = factor_df[factor_df['quarter_end'] == next_end]
    if not next_factor.empty:
        next_sig_date = next_factor.iloc[0]['signal_date']
    else:
        next_sig_date = (pd.to_datetime(sig_date) + pd.Timedelta(days=90)).strftime('%Y-%m-%d')

    # 매도일이 오늘 이후면 미완성 포지션 → 제외
    if pd.to_datetime(next_sig_date) > pd.Timestamp.today().normalize():
        print(f"⏭️ 매도일({next_sig_date})이 미래 — 미완성 분기 제외")
        continue

    # ---------------------------------------------
    # 유동성 필터: 일평균 거래대금 5억원 미만 제거
    # ---------------------------------------------
    before_liq = len(q_factor)
    liq_mask = q_factor['ticker'].map(
        lambda t: get_avg_trading_value(t, sig_date) >= MIN_TRADING_VALUE
    )
    q_factor = q_factor[liq_mask].copy()
    filtered_out = before_liq - len(q_factor)
    if filtered_out > 0:
        print(f"  유동성 필터: {before_liq}개 → {len(q_factor)}개 ({filtered_out}개 제외)")

    max_top_n = max(STRATEGY_TOP_N.values())
    if len(q_factor) < max_top_n:
        print(f"  ⚠️ 유동성 필터 후 종목 수({len(q_factor)})가 부족 (필요: {max_top_n}개)")
        continue

    # ---------------------------------------------
    # 시장 국면 판단
    # ---------------------------------------------
    regime = get_market_regime(sig_date)

    if regime == 'bull':
        print("강세장")
        regime_weights = BULL_WEIGHTS
    elif regime == 'sideways':
        print("횡보장")
        regime_weights = SIDEWAYS_WEIGHTS
    else:
        print("약세장")
        regime_weights = BEAR_WEIGHTS

    # ---------------------------------------------
    # 전략별 점수 계산 (numpy 벡터 연산 — apply 대비 수십 배 빠름)
    # ---------------------------------------------
    factor_cols = ['growth_score', 'value_score', 'quality_score',
                   'profitability_score', 'momentum_score', 'lowvol_score']
    factor_matrix = q_factor[factor_cols].fillna(50).values  # shape: (n_stocks, 6)

    for strat_name, weights in [
        ('밸런스형',   BALANCED_WEIGHTS),
        ('저평가형',   VALUE_WEIGHTS),
        ('모멘텀형',   MOMENTUM_WEIGHTS),
        ('국면대응형', regime_weights),
    ]:
        w = np.array([weights[c] for c in factor_cols])
        q_factor[strat_name] = np.round(factor_matrix @ w, 2)

    # 섹터 중립화 제거: 원점수 그대로 사용
    # 포트폴리오 선택 단계에서 섹터 캡(MAX_PER_SECTOR) 적용
    # 모멘텀형은 캡 없음 — 섹터 집중이 알파 소스
    for strat_name in ['밸런스형', '저평가형', '모멘텀형', '국면대응형']:
        q_factor[f'{strat_name}_sn'] = q_factor[strat_name]

    # ---------------------------------------------
    # 코스피 수익률 (개별 주식 DB가 아닌 kospi 변수에서 직접 계산!)
    # ---------------------------------------------
    try:
        k_buy_rows = kospi[kospi.index >= sig_date]
        k_sell_rows = kospi[kospi.index <= next_sig_date]
        if k_buy_rows.empty or k_sell_rows.empty:
            raise ValueError(f"KOSPI 날짜 범위 밖: {sig_date} ~ {next_sig_date}")
        k_buy = k_buy_rows.iloc[0]['Close']
        k_sell = k_sell_rows.iloc[-1]['Close']
        kospi_ret = ((k_sell / k_buy) - 1) * 100
    except Exception as e:
        print(f"  ⚠️ KOSPI 수익률 계산 실패 ({sig_date}~{next_sig_date}): {e}")
        kospi_ret = None

    quarter_result = {
        'quarter': q_end,
        'sig_date': sig_date,
        'next_sig_date': next_sig_date,
        'regime': regime,
        'kospi_return': kospi_ret,
    }

    # =====================================================
    # 전략별 백테스트
    # =====================================================
    for strategy in ['밸런스형', '저평가형', '모멘텀형', '국면대응형']:

        top_n  = STRATEGY_TOP_N[strategy]
        sn_col = f'{strategy}_sn'

        # 저평가형은 약세장에서 버퍼 해제 (약세장에서 보유 연장이 오히려 손해)
        use_buf = USE_BUFFER and not (strategy == '저평가형' and regime == 'bear')

        use_cap = (strategy != '모멘텀형')

        if use_buf:
            buffer = STRATEGY_BUFFER[strategy]
            if use_cap:
                extended   = set(select_with_sector_cap(q_factor, sn_col, top_n + buffer, MAX_PER_SECTOR + 1))
                held       = prev_holdings[strategy] & extended
                strict_top = select_with_sector_cap(q_factor, sn_col, top_n, MAX_PER_SECTOR)
            else:
                extended   = set(q_factor.nlargest(top_n + buffer, sn_col)['ticker'].tolist())
                held       = prev_holdings[strategy] & extended
                strict_top = q_factor.nlargest(top_n, sn_col)['ticker'].tolist()
            new_entries = [t for t in strict_top if t not in held]
            slots      = max(0, top_n - len(held))
            added      = set(new_entries[:slots])
            portfolio  = held | added
            exits      = prev_holdings[strategy] - portfolio
        else:
            if use_cap:
                strict_top = select_with_sector_cap(q_factor, sn_col, top_n, MAX_PER_SECTOR)
            else:
                strict_top = q_factor.nlargest(top_n, sn_col)['ticker'].tolist()
            portfolio  = set(strict_top)
            held       = prev_holdings[strategy] & portfolio
            added      = portfolio - held
            exits      = prev_holdings[strategy] - portfolio

        prev_holdings[strategy] = portfolio

        returns = []
        for ticker in portfolio:
            ret = get_price_return(ticker, sig_date, next_sig_date)
            if ret is None:
                continue
            cost = (TRANSACTION_COST * 100 if ticker in added else 0) if use_buf else TRANSACTION_COST * 100
            returns.append(ret - cost)

        # 퇴출 매도비용: 포트폴리오 평균에서 비례 차감
        if use_buf and len(returns) > 0 and len(exits) > 0:
            exit_drag = len(exits) * TRANSACTION_COST * 100 / len(returns)
            returns = [r - exit_drag for r in returns]

        if len(returns) == 0:
            continue

        avg_ret = np.mean(returns)
        winrate = (np.array(returns) > 0).mean() * 100
        excess  = avg_ret - (kospi_ret if kospi_ret is not None else 0)

        turnover_n = len(added) + len(exits)
        overlap_n  = len(held)

        if abs(avg_ret) > 50:
            print(f"  ⚠️ 이상 수익률 감지 [{strategy}]: {avg_ret:+.1f}%")

        quarter_result[f'{strategy}_return']  = avg_ret
        quarter_result[f'{strategy}_winrate'] = winrate

        print(
            f"[{strategy}] "
            f"수익률: {avg_ret:+.1f}% | "
            f"승률: {winrate:.0f}% | "
            f"초과수익: {excess:+.1f}%p | "
            f"유지 {overlap_n}개 / 교체 {turnover_n}개"
        )

    results.append(quarter_result)

# =========================================================
# 9. 결과 정리
# =========================================================
result_df = pd.DataFrame(results)

print(f"\n{'='*70}")
print("📊 최종 결과")
print(f"{'='*70}")

strategies = [
    '밸런스형',
    '저평가형',
    '모멘텀형',
    '국면대응형'
]

for strategy in strategies:

    col = f'{strategy}_return'

    if col not in result_df.columns:
        continue

    valid = result_df[col].dropna()

    if valid.empty:
        continue

    avg_return = valid.mean()

    cum_series = (1 + valid / 100).cumprod()
    cumulative = (cum_series.iloc[-1] - 1) * 100

    # 실제 MDD: 누적 수익률 고점 대비 최대 낙폭
    rolling_max = cum_series.cummax()
    drawdowns = (cum_series - rolling_max) / rolling_max * 100
    mdd = drawdowns.min()

    best = valid.max()
    worst = valid.min()

    winrate = result_df[
        f'{strategy}_winrate'
    ].mean()

    kospi_valid = result_df['kospi_return'].dropna()
    kospi_avg = kospi_valid.mean() if not kospi_valid.empty else 0
    kospi_cum = (np.prod(1 + kospi_valid / 100) - 1) * 100

    excess = avg_return - kospi_avg

    print(f"\n[{strategy}]")
    print(f"누적 수익률      : {cumulative:+.1f}%  (코스피 {kospi_cum:+.1f}%)")
    print(f"평균 분기 수익률 : {avg_return:+.1f}%")
    print(f"코스피 초과수익  : {excess:+.1f}%p")
    print(f"평균 승률        : {winrate:.1f}%")
    print(f"최고 분기        : {best:+.1f}%")
    print(f"최악 분기        : {worst:+.1f}%")
    print(f"MDD              : {mdd:+.1f}%")

    # CAGR (연환산)
    sig_dates      = pd.to_datetime(result_df['sig_date'].dropna())
    next_sig_dates = pd.to_datetime(result_df['next_sig_date'].dropna())
    if not sig_dates.empty and not next_sig_dates.empty:
        years = (next_sig_dates.max() - sig_dates.min()).days / 365.25
        cagr  = ((cum_series.iloc[-1]) ** (1 / years) - 1) * 100
    else:
        cagr = cumulative / (len(valid) / 4)

    # 샤프 비율 (연환산)
    excess_rf = valid / 100 - RISK_FREE_RATE_QTR
    sharpe = (excess_rf.mean() / excess_rf.std() * np.sqrt(4)) if excess_rf.std() > 0 else 0

    # 정보 비율 IR (연환산)
    aligned_df  = result_df[[col, 'kospi_return']].dropna()
    qtr_excess  = aligned_df[col] - aligned_df['kospi_return']
    ir = (qtr_excess.mean() / qtr_excess.std() * np.sqrt(4)) if qtr_excess.std() > 0 else 0

    print(f"CAGR             : {cagr:+.1f}%  (연환산)")
    print(f"샤프 비율        : {sharpe:.2f}  (무위험금리 3.5% 기준)")
    print(f"정보 비율(IR)    : {ir:.2f}")

# =========================================================
# 9-1. 연도별 성과 분석
# =========================================================
result_df['year'] = pd.to_datetime(result_df['sig_date']).dt.year

print(f"\n{'='*70}")
print("📅 연도별 성과 (분기 평균)")
print(f"{'='*70}")

yearly_rows = []
for year in sorted(result_df['year'].dropna().unique().astype(int)):
    yr  = result_df[result_df['year'] == year]
    row = {
        '연도':   int(year),
        '분기수': len(yr),
        '코스피': round(yr['kospi_return'].mean(), 1),
    }
    for strat in strategies:
        scol = f'{strat}_return'
        if scol in yr.columns:
            row[strat] = round(yr[scol].mean(), 1)
    yearly_rows.append(row)

yearly_df = pd.DataFrame(yearly_rows)
print(yearly_df.to_string(index=False))

# =========================================================
# 10. 저장
# =========================================================
now = datetime.now().strftime('%Y%m%d_%H%M%S')

# ----------------------------------------------------------
# 10-1. 상세 결과: 컬럼명 한글화 + regime 값 한글화
# ----------------------------------------------------------
REGIME_KR = {'bull': '강세장', 'sideways': '횡보장', 'bear': '약세장'}

COLUMN_KR = {
    'quarter':           '분기말',
    'sig_date':          '매수일',
    'next_sig_date':     '매도일',
    'regime':            '시장국면',
    'kospi_return':      '코스피_수익률(%)',
    '밸런스형_return':   '밸런스형_수익률(%)',
    '밸런스형_winrate':  '밸런스형_승률(%)',
    '저평가형_return':   '저평가형_수익률(%)',
    '저평가형_winrate':  '저평가형_승률(%)',
    '모멘텀형_return':   '모멘텀형_수익률(%)',
    '모멘텀형_winrate':  '모멘텀형_승률(%)',
    '국면대응형_return': '국면대응형_수익률(%)',
    '국면대응형_winrate':'국면대응형_승률(%)',
}

detail_df = result_df.copy()
detail_df['regime'] = detail_df['regime'].map(REGIME_KR).fillna(detail_df['regime'])

# 숫자 컬럼 소수점 1자리 반올림
for col in detail_df.select_dtypes(include='number').columns:
    detail_df[col] = detail_df[col].round(1)

detail_df = detail_df.rename(columns=COLUMN_KR)

detail_path = os.path.join(RESULT_DIR, f'backtest_detail_{now}.csv')
detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')

# ----------------------------------------------------------
# 10-2. 국면별 요약: 강세/횡보/약세 × 전략별 성과
# ----------------------------------------------------------
regime_order    = ['강세장', '횡보장', '약세장']
strategy_labels = ['밸런스형', '저평가형', '모멘텀형', '국면대응형']

summary_rows = []

for regime_kr in regime_order:

    regime_data = detail_df[detail_df['시장국면'] == regime_kr]

    if regime_data.empty:
        continue

    for strat in strategy_labels:

        ret_col  = f'{strat}_수익률(%)'
        win_col  = f'{strat}_승률(%)'

        if ret_col not in regime_data.columns:
            continue

        rets = regime_data[ret_col].dropna()

        if rets.empty:
            continue

        kospi_rets = regime_data['코스피_수익률(%)'].dropna()
        kospi_avg  = kospi_rets.mean() if not kospi_rets.empty else 0

        # 샤프 비율 (국면 내)
        rf_qtr = RISK_FREE_RATE_QTR * 100
        excess_rf_r = rets - rf_qtr
        sharpe_r = (excess_rf_r.mean() / excess_rf_r.std() * np.sqrt(4)) if excess_rf_r.std() > 0 else 0

        # IR (국면 내)
        qtr_exc_r = rets - kospi_rets.reindex(rets.index).fillna(kospi_avg)
        ir_r = (qtr_exc_r.mean() / qtr_exc_r.std() * np.sqrt(4)) if qtr_exc_r.std() > 0 else 0

        summary_rows.append({
            '시장국면':      regime_kr,
            '전략':          strat,
            '분기수':        len(rets),
            '평균수익률(%)': round(rets.mean(), 1),
            '평균승률(%)':   round(regime_data[win_col].dropna().mean(), 1),
            '코스피초과(%p)': round(rets.mean() - kospi_avg, 1),
            '최고분기(%)':   round(rets.max(), 1),
            '최저분기(%)':   round(rets.min(), 1),
            '샤프':          round(sharpe_r, 2),
            'IR':            round(ir_r, 2),
        })

summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(RESULT_DIR, f'backtest_regime_summary_{now}.csv')
summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

# 연도별 성과 저장
yearly_path = os.path.join(RESULT_DIR, f'backtest_yearly_{now}.csv')
yearly_df.to_csv(yearly_path, index=False, encoding='utf-8-sig')

# ----------------------------------------------------------
# 출력
# ----------------------------------------------------------
print(f"\n{'='*60}")
print("📁 저장 완료")
print(f"  상세 결과  : backtest_detail_{now}.csv")
print(f"  국면별 요약: backtest_regime_summary_{now}.csv")
print(f"  연도별 성과: backtest_yearly_{now}.csv")

print(f"\n{'='*60}")
print("📊 국면별 전략 성과 요약")
print(f"{'='*60}")
print(summary_df.to_string(index=False))