import glob
import os
import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import plotly.graph_objects as go

st.set_page_config(page_title="퀀트 스코어링", page_icon="📊", layout="wide")

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
FACTOR_DB_PATH = os.path.join(BASE_DIR, 'db', 'factor_db.parquet')
RESULT_DIR     = os.path.join(BASE_DIR, 'results')
LIVE_DB_PATH   = os.path.join(BASE_DIR, 'db', 'live_positions.parquet')

# ──────────────────────────────────────────
# 전략 설정
# ──────────────────────────────────────────
STRATEGY_TOP_N = {'밸런스형': 20, '저평가형': 20, '모멘텀형': 15, '국면대응형': 20, '촉매형': 20}
MAX_PER_SECTOR = 4  # 섭터별 최대 편입 종목 수 (모멘텀형 제외)

FACTOR_COLS   = ['growth_score', 'value_score', 'quality_score',
                 'profitability_score', 'momentum_score', 'lowvol_score',
                 'garp_score', 'catalyst_score']
FACTOR_LABELS = {
    'growth_score': '성장', 'value_score': '가치', 'quality_score': '퀄리티',
    'profitability_score': '수익성', 'momentum_score': '모멘텀', 'lowvol_score': '저변동성',
    'garp_score': 'GARP', 'catalyst_score': '촉매',
}
_SHORT_DESC = {
    'growth_score': '매출·영업이익·순이익 YoY', 'value_score': 'PER·POR·ROE 저평가',
    'quality_score': 'ROE·부채비율·흑자여부', 'profitability_score': 'ROA·영업이익률(OPM)',
    'momentum_score': '6개월 수익률·52주 최고가', 'lowvol_score': '연간 변동성',
    'garp_score': '이익가속·PEG·ROE 종합', 'catalyst_score': '어닝서프라이즈·마진변곡·턴어라운드',
}
FACTOR_DESC = {
    'growth_score': (
        "**성장성** — 기업이 얼마나 빠르게 커지고 있는가\n\n"
        "- 매출 YoY 성장률 (30%)\n- 영업이익 YoY 성장률 (40%)\n- 당기순이익 YoY 성장률 (30%)\n\n"
        "📌 매출 +150% 이상이면 100점, -30% 이하이면 0점"
    ),
    'value_score': (
        "**가치** — 현재 주가가 싼가 비싼가 (GARP 방식)\n\n"
        "- PER: 시총/순이익×4 (35%) — 낮을수록 좋음\n"
        "- POR: 시총/영업이익×4 (30%) — 낮을수록 좋음\n"
        "- ROE 기반 수익성 필터 (35%)\n\n"
        "📌 섹터 PER의 0.4배면 100점, 3배면 0점\n"
        "📌 적자 기업은 PER/POR 자동 0점 패널티"
    ),
    'quality_score': (
        "**퀄리티** — 재무 건전성과 수익 안정성\n\n"
        "- ROE (60%) — 자기자본 대비 순이익률\n"
        "- 부채비율 (25%) — 낮을수록 좋음\n"
        "- 영업이익 흑자 여부 (15%)\n\n"
        "📌 ROE 섹터 벤치마크의 3배면 100점, 0%면 0점"
    ),
    'profitability_score': (
        "**수익성** — 자산·매출을 얼마나 효율적으로 활용하는가\n\n"
        "- ROA: 총자산 대비 영업이익률 (60%)\n"
        "- OPM: 영업이익률 (40%)\n\n"
        "📌 ROA 10% 이상이면 100점, 0%면 0점\n"
        "📌 OPM은 섹터 벤치마크 기준"
    ),
    'momentum_score': (
        "**모멘텀** — 주가 상승 추세의 강도\n\n"
        "- 6개월 수익률 (70%)\n"
        "- 52주 최고가 대비 현재가 비율 (30%)\n\n"
        "📌 6개월 +60% 이상이면 100점, -30% 이하면 0점"
    ),
    'lowvol_score': (
        "**저변동성** — 주가가 얼마나 안정적인가\n\n"
        "- 252일 연간 변동성 기준, 분기 내 유니버스 퍼센타일 순위\n\n"
        "📌 100점 = 이번 분기 가장 안정적인 종목\n"
        "📌 약세장 방어 역할"
    ),
    'garp_score': (
        "**GARP** — 이익 가속 + 합리적 밸류 종합 점수\n\n"
        "- 최근 4분기 영업이익 연속 흑자 (+1)\n"
        "- 이익 가속: 현재 YoY > 직전분기 YoY (+1)\n"
        "- 매출 YoY 성장 (+1)\n"
        "- PEG ≤ 1.5 (+1), PEG ≤ 1.0 보너스 (+1)\n"
        "- ROE 대리: 4Q 순이익 합계 양수 (+1)\n\n"
        "📌 6/6 기준 100점 → 0점 (선형 정규화)"
    ),
    'catalyst_score': (
        "**촉매** — 단기 재평가를 유발하는 이벤트 드리븐 신호\n\n"
        "- 어닝 서프라이즈: 현재분기 영업이익 > 직전 4분기 평균 +30% (+1)\n"
        "- 마진 변곡점: OPM QoQ·YoY 동시 개선 (+1)\n"
        "- 턴어라운드: 직전분기 적자 → 현재 흑자 (+1)\n\n"
        "📌 국면 무관 — 실적 촉매는 기관 강제 매수 유발\n"
        "📌 3/3 기준 100점 → 0점"
    ),
}

BALANCED_WEIGHTS = {'growth_score': 0.20, 'value_score': 0.20, 'quality_score': 0.20,
                    'profitability_score': 0.15, 'momentum_score': 0.15, 'lowvol_score': 0.10,
                    'garp_score': 0.00, 'catalyst_score': 0.00}
VALUE_WEIGHTS    = {'growth_score': 0.10, 'value_score': 0.35, 'quality_score': 0.20,
                    'profitability_score': 0.25, 'momentum_score': 0.00, 'lowvol_score': 0.10,
                    'garp_score': 0.00, 'catalyst_score': 0.00}
MOMENTUM_WEIGHTS = {'growth_score': 0.25, 'value_score': 0.00, 'quality_score': 0.10,
                    'profitability_score': 0.10, 'momentum_score': 0.45, 'lowvol_score': 0.10,
                    'garp_score': 0.00, 'catalyst_score': 0.00}
BULL_WEIGHTS     = {'growth_score': 0.25, 'value_score': 0.05, 'quality_score': 0.15,
                    'profitability_score': 0.15, 'momentum_score': 0.30, 'lowvol_score': 0.10,
                    'garp_score': 0.00, 'catalyst_score': 0.00}
SIDEWAYS_WEIGHTS = {'growth_score': 0.15, 'value_score': 0.20, 'quality_score': 0.25,
                    'profitability_score': 0.15, 'momentum_score': 0.15, 'lowvol_score': 0.10,
                    'garp_score': 0.00, 'catalyst_score': 0.00}
BEAR_WEIGHTS     = {'growth_score': 0.05, 'value_score': 0.40, 'quality_score': 0.25,
                    'profitability_score': 0.20, 'momentum_score': 0.00, 'lowvol_score': 0.10,
                    'garp_score': 0.00, 'catalyst_score': 0.00}
CATALYST_WEIGHTS = {'growth_score': 0.05, 'value_score': 0.05, 'quality_score': 0.05,
                    'profitability_score': 0.05, 'momentum_score': 0.15, 'lowvol_score': 0.05,
                    'garp_score': 0.30, 'catalyst_score': 0.30}

REGIME_WEIGHTS = {'bull': BULL_WEIGHTS, 'sideways': SIDEWAYS_WEIGHTS, 'bear': BEAR_WEIGHTS}
REGIME_KR      = {'bull': '강세장', 'sideways': '횡보장', 'bear': '약세장'}
REGIME_EMOJI   = {'bull': '📈', 'sideways': '➡️', 'bear': '📉'}

SECTOR_BENCHMARKS = {
    '반도체':         {'PER': 25, 'POR': 18, 'ROE': 15, 'OPM': 20},
    'IT':             {'PER': 20, 'POR': 14, 'ROE': 12, 'OPM': 15},
    '자동차':         {'PER':  8, 'POR':  6, 'ROE': 10, 'OPM':  6},
    '금융':           {'PER':  6, 'POR':  5, 'ROE':  8, 'OPM': 20},
    '바이오':         {'PER': 50, 'POR': 35, 'ROE':  5, 'OPM':  5},
    '산업재':         {'PER': 10, 'POR':  7, 'ROE':  8, 'OPM':  6},
    '소비재':         {'PER': 15, 'POR': 10, 'ROE': 12, 'OPM':  8},
    '엔터/미디어':    {'PER': 25, 'POR': 15, 'ROE': 10, 'OPM': 10},
    '에너지/유틸리티':{'PER': 10, 'POR':  8, 'ROE':  8, 'OPM':  8},
    '기본':           {'PER': 12, 'POR':  8, 'ROE': 10, 'OPM':  8},
}

REGIME_ADVICE = {
    'bull':     {'추천': ['모멘텀형', '국면대응형'], '설명': '상승 추세 강함 — 모멘텀·성장 팩터 집중 활용',
                 'color': '#E8F5E9', 'border': '#2E7D32'},
    'sideways': {'추천': ['촉매형', '밸런스형'],     '설명': '방향성 불명확 — 실적 촉매·균형 분산이 유리',
                 'color': '#FFF3E0', 'border': '#E65100'},
    'bear':     {'추천': ['촉매형', '저평가형'],     '설명': '하락 국면 — 어닝 서프라이즈 기업은 역방향 매수 유발',
                 'color': '#FFEBEE', 'border': '#B71C1C'},
}

# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_factor_db():
    df = pd.read_parquet(FACTOR_DB_PATH)
    df['ticker']      = df['ticker'].astype(str).str.zfill(6)
    df['signal_date'] = pd.to_datetime(df['signal_date'])
    df['quarter_end'] = pd.to_datetime(df['quarter_end'])
    # 저변동성: 분기 내 유니버스 퍼센타일로 재정규화 (절대값 → 상대 순위)
    df['lowvol_score'] = (
        df.groupby('signal_date')['lowvol_score']
        .rank(pct=True, method='average') * 100
    ).round(1)
    return df

@st.cache_data(ttl=1800)
def load_kospi():
    kospi = fdr.DataReader('KS200', '2023-01-01')
    kospi = kospi[['Close']].copy()
    kospi['MA200'] = kospi['Close'].rolling(200).mean()
    return kospi

@st.cache_data(ttl=3600)
def get_name_dict():
    import json
    # 로컬 파일 우선 (클라우드 환경 대비)
    local = os.path.join(BASE_DIR, 'db', 'ticker_names.json')
    if os.path.exists(local):
        with open(local, encoding='utf-8') as f:
            return json.load(f)
    try:
        krx = fdr.StockListing('KRX')
        krx['Code'] = krx['Code'].astype(str).str.zfill(6)
        return dict(zip(krx['Code'], krx['Name']))
    except Exception:
        return {}

@st.cache_data(ttl=3600)
def load_backtest():
    files = sorted(glob.glob(os.path.join(RESULT_DIR, 'backtest_detail_*.csv')))
    if not files:
        return None
    return pd.read_csv(files[-1])

@st.cache_data(ttl=1800)
def load_kospi_full():
    """KS11 2년치 — 차트 및 오늘의 시황 위젯용"""
    start = (pd.Timestamp.today() - pd.Timedelta(days=760)).strftime('%Y-%m-%d')
    k = fdr.DataReader('KS11', start)
    k.index = pd.to_datetime(k.index)
    k['MA200'] = k['Close'].rolling(200).mean()
    return k

def _map_sector(raw: str, name: str = '') -> str:
    r = raw if raw else ''
    n = name if name else ''
    if '반도체' in r or '전자부품' in r or '집적회로' in r or '디스플레이' in r or '반도체' in n or '솔브레인' in n:
        return '반도체'
    if ('소프트웨어' in r or '컴퓨터' in r or '정보통신' in r or '통신업' in r
            or '인터넷' in r or '자료처리' in r or '포털' in r
            or '통신 및 방송 장비' in r or '통신장비' in r or '전기통신' in r
            or '정보 서비스' in r or '측정, 시험' in r):
        return 'IT'
    if '자동차' in r or '운송장비' in r or '모터' in r:
        return '자동차'
    if ('은행' in r or '보험' in r or '증권' in r or '금융업' in r
            or '금융 지원 서비스' in r or '신탁업' in r or '집합투자' in r
            or '여신' in r or '대부업' in r):
        return '금융'
    if ('의약' in r or '의료' in r or '바이오' in r or '제약' in r
            or '연구개발' in r or '기초 의약물질' in r or '의약품 제조' in r
            or '녹십자' in n):
        return '바이오'
    # 에너지/유틸리티: 정유·가스 생산·전기 공급
    # 이름 override: SK가스·E1 등 Industry가 "기타 전문 도매업"인 LPG/LNG 도매 커버
    # 메가스터디처럼 "메가" 안에 "가스"가 섞이는 오매칭 방지
    if ('석유 정제' in r or '원유' in r or '정유' in r
            or '연료용 가스' in r or '가스 배관' in r
            or '전기업' in r or '수력 발전' in r or '원자력' in r
            or ('가스' in n and '메가' not in n and '이차전지' not in r and '배터리' not in r)
            or '정유' in n or n in ('E1', 'INVENI')
            or '증기, 냉' in r or '에너비스' in n):
        return '에너지/유틸리티'
    if ('게임' in r or '엔터' in r or '방송프로그램' in r or '비디오물' in r
            or '미디어' in r or '콘텐츠' in r or '광고' in r
            or '영화' in r or '음악' in r or '공연' in r
            or '텔레비전 방송' in r or '창작 및 예술' in r or '오디오물' in r):
        return '엔터/미디어'
    if ('음식' in r or '식품' in r or '음료' in r or '유통' in r
            or '의류' in r or '의복' in r or '봉제' in r
            or '섬유' in r or '화장품' in r or '생활용품' in r
            or '도소매' in r or '패션' in r or '소매업' in r or '종합 소매' in r
            or '가정용 기기' in r or '영상 및 음향기기' in r or '가전' in r
            or '가구' in r or '출판' in r
        or '낙농' in r or '수산물' in r or '곰물' in r
        or '가죽' in r or '담배' in r):
        return '소비재'
    if ('조선' in r or '선박' in r or '보트 건조' in r
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
        return '산업재'
    return '기본'

@st.cache_data(ttl=3600)
def get_sector_dict() -> dict:
    try:
        krx = fdr.StockListing('KRX')
        krx['Code'] = krx['Code'].astype(str).str.zfill(6)
        kd  = fdr.StockListing('KRX-DESC')
        kd['Code'] = kd['Code'].astype(str).str.zfill(6)
        if 'Sector' in krx.columns:
            krx = krx.drop(columns=['Sector'])
        cols = ['Code', 'Sector', 'Industry'] if 'Industry' in kd.columns else ['Code', 'Sector']
        krx = krx.merge(kd[cols], on='Code', how='left')
        result = {}
        for _, row in krx.iterrows():
            sec  = str(row.get('Sector', '')   or '')
            ind  = str(row.get('Industry', '') or '')
            name = str(row.get('Name', '')     or '')
            result[row['Code']] = _map_sector((sec + ' ' + ind).strip(), name)
        return result
    except Exception:
        return {}

# ──────────────────────────────────────────
# 지표 계산
# ──────────────────────────────────────────
def get_regime(kospi):
    row = kospi.dropna(subset=['MA200']).iloc[-1]
    curr, ma200 = row['Close'], row['MA200']
    ratio = (curr / ma200 - 1) * 100
    if ratio > 3:    regime = 'bull'
    elif ratio < -3: regime = 'bear'
    else:            regime = 'sideways'
    return regime, curr, ma200, ratio

def next_rebalancing(quarter_end: pd.Timestamp) -> pd.Timestamp:
    m, y = quarter_end.month, quarter_end.year
    nq_map = {3: pd.Timestamp(y, 6, 30), 6: pd.Timestamp(y, 9, 30),
              9: pd.Timestamp(y, 12, 31), 12: pd.Timestamp(y + 1, 3, 31)}
    return nq_map[m] + pd.Timedelta(days=45)

def select_with_sector_cap(df: pd.DataFrame, score_col: str,
                           top_n: int, max_per_sector: int) -> pd.DataFrame:
    sorted_df = df.sort_values(score_col, ascending=False)
    selected, counts = [], {}
    for idx, row in sorted_df.iterrows():
        s = row['sector']
        if counts.get(s, 0) < max_per_sector:
            selected.append(idx)
            counts[s] = counts.get(s, 0) + 1
        if len(selected) >= top_n:
            break
    return df.loc[selected]

def compute_scores(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    df = df.copy()
    # 새 팩터가 DB에 없으면 중립값 50 삽입 (factor_db 재빌드 전 과도기 대응)
    for col in ['garp_score', 'catalyst_score']:
        if col not in df.columns:
            df[col] = 50.0
    avail_cols = [c for c in FACTOR_COLS if c in df.columns]
    matrix = df[avail_cols].fillna(50).values
    for name, weights in [('밸런스형', BALANCED_WEIGHTS), ('저평가형', VALUE_WEIGHTS),
                          ('모멘텀형', MOMENTUM_WEIGHTS), ('국면대응형', REGIME_WEIGHTS[regime]),
                          ('촉매형', CATALYST_WEIGHTS)]:
        w = np.array([weights[c] for c in avail_cols])
        df[name] = np.round(matrix @ w, 2)
    for name in ['밸런스형', '저평가형', '모멘텀형', '국면대응형', '촉매형']:
        df[f'{name}_sn'] = df[name]
    return df

def calc_strat_metrics(bt_df: pd.DataFrame, strat: str) -> dict:
    RISK_FREE_QTR = 0.035 / 4
    ret_col, kospi_col = f'{strat}_수익률(%)', '코스피_수익률(%)'
    if ret_col not in bt_df.columns or bt_df.empty:
        return {}
    rets   = bt_df[ret_col].values
    kospi  = bt_df[kospi_col].values
    excess = rets - kospi
    sharpe = (rets.mean() - RISK_FREE_QTR * 100) / (rets.std() + 1e-9)
    ir     = excess.mean() / (excess.std() + 1e-9)
    win    = (excess > 0).mean() * 100
    n      = len(rets)
    cagr   = (np.prod(1 + rets / 100) ** (4 / n) - 1) * 100
    return {'sharpe': round(sharpe, 2), 'ir': round(ir, 2),
            'win': round(win, 1), 'cagr': round(cagr, 1)}

# ──────────────────────────────────────────
# 추천 이유 생성
# ──────────────────────────────────────────
_RAW_COLS = ['sales_yoy', 'op_yoy', 'ni_yoy', 'per', 'por',
             'roe', 'debt_ratio', 'roa', 'opm', 'mom_6m', 'volatility']

def _fmt_yoy(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    if v > 999:  return '+999%↑'
    if v < -99:  return '-99%↓'
    return f'{v:+.0f}%'

def _pct_cmp(val, bench, higher_is_better=True):
    if val is None or bench == 0: return ''
    ratio = (val - bench) / bench * 100
    if higher_is_better:
        if ratio > 30:  return f' ↑섹터평균({bench:.0f}) 대비 {ratio:+.0f}%'
        if ratio < -20: return f' ↓섹터평균({bench:.0f}) 대비 {ratio:+.0f}%'
    else:
        if ratio < -20: return f' ↓섹터평균({bench:.0f}배) 대비 {abs(ratio):.0f}% 저평가'
        if ratio > 30:  return f' ↑섹터평균({bench:.0f}배) 대비 {ratio:.0f}% 고평가'
    return ''

def generate_reason(row: pd.Series) -> str:
    has_raw = any(col in row.index and pd.notna(row.get(col)) for col in _RAW_COLS)
    if not has_raw:
        return '_상세 수치 없음 — DB 재빌드 후 표시됩니다._'

    bench = SECTOR_BENCHMARKS.get(str(row.get('sector', '기본')), SECTOR_BENCHMARKS['기본'])
    lines = []

    def icon(score):
        if score >= 70: return '✅'
        if score < 40:  return '⚠️'
        return '▶'

    def _v(col): return row.get(col) if col in row.index and pd.notna(row.get(col)) else None

    # 성장성
    sy, oy, ny = _v('sales_yoy'), _v('op_yoy'), _v('ni_yoy')
    g_parts = [x for x in [
        f'매출 {_fmt_yoy(sy)}'    if sy is not None else None,
        f'영업이익 {_fmt_yoy(oy)}' if oy is not None else None,
        f'순이익 {_fmt_yoy(ny)}'   if ny is not None else None,
    ] if x]
    if g_parts:
        lines.append(f'{icon(row["growth_score"])} **성장성 {row["growth_score"]:.0f}점** — {" · ".join(g_parts)} (전년 동기 대비)')

    # 가치
    per_v, por_v, roe_v = _v('per'), _v('por'), _v('roe')
    v_parts = []
    if per_v: v_parts.append(f'PER {per_v:.1f}배{_pct_cmp(per_v, bench["PER"], False)}')
    if por_v: v_parts.append(f'POR {por_v:.1f}배{_pct_cmp(por_v, bench["POR"], False)}')
    if roe_v: v_parts.append(f'ROE {roe_v:.1f}%{_pct_cmp(roe_v, bench["ROE"])}')
    if v_parts:
        lines.append(f'{icon(row["value_score"])} **가치 {row["value_score"]:.0f}점** — {" · ".join(v_parts)}')

    # 퀄리티
    debt_v = _v('debt_ratio')
    q_parts = []
    if roe_v:  q_parts.append(f'ROE {roe_v:.1f}%{_pct_cmp(roe_v, bench["ROE"])}')
    if debt_v:
        d_desc = '안정' if debt_v < 100 else ('주의' if debt_v < 200 else '위험')
        q_parts.append(f'부채비율 {debt_v:.0f}% ({d_desc})')
    if q_parts:
        lines.append(f'{icon(row["quality_score"])} **퀄리티 {row["quality_score"]:.0f}점** — {" · ".join(q_parts)}')

    # 수익성
    roa_v, opm_v = _v('roa'), _v('opm')
    p_parts = []
    if roa_v: p_parts.append(f'ROA {roa_v:.1f}%')
    if opm_v: p_parts.append(f'영업이익률 {opm_v:.1f}%{_pct_cmp(opm_v, bench["OPM"])}')
    if p_parts:
        lines.append(f'{icon(row["profitability_score"])} **수익성 {row["profitability_score"]:.0f}점** — {" · ".join(p_parts)}')

    # 모멘텀
    m6 = _v('mom_6m')
    if m6 is not None:
        trend = '상승 추세' if m6 > 10 else ('약보합' if m6 > -10 else '하락 추세')
        lines.append(f'{icon(row["momentum_score"])} **모멘텀 {row["momentum_score"]:.0f}점** — 최근 6개월 {m6:+.1f}% ({trend})')

    # 저변동성
    vol = _v('volatility')
    if vol is not None:
        stab = '매우 안정' if vol < 20 else ('안정' if vol < 30 else ('보통' if vol < 45 else '높은 변동성'))
        lines.append(f'{icon(row["lowvol_score"])} **저변동성 {row["lowvol_score"]:.0f}점** — 연간 변동성 {vol:.1f}% ({stab})')

    # GARP
    gs = row.get('garp_score') if 'garp_score' in row.index else None
    if gs is not None and not (isinstance(gs, float) and np.isnan(gs)) and gs != 50:
        lines.append(f'{icon(gs)} **GARP {gs:.0f}점** — 이익가속·PEG·ROE 종합 (50점=데이터부족)')

    # 촉매
    cs = row.get('catalyst_score') if 'catalyst_score' in row.index else None
    if cs is not None and not (isinstance(cs, float) and np.isnan(cs)) and cs != 50:
        lines.append(f'{icon(cs)} **촉매 {cs:.0f}점** — 어닝서프라이즈·마진변곡·턴어라운드')

    return '\n\n'.join(lines) if lines else '_데이터 없음_'

# ──────────────────────────────────────────
# 차트
# ──────────────────────────────────────────
def history_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = {'성장': '#FF6B6B', '가치': '#4ECDC4', '퀄리티': '#45B7D1',
              '수익성': '#96CEB4', '모멘텀': '#FFEAA7', '저변동성': '#DDA0DD',
              'GARP': '#FF9800', '촉매': '#E91E63'}
    col_map = dict(zip(FACTOR_COLS, colors.keys()))
    for col, label in FACTOR_LABELS.items():
        if col not in hist.columns:
            continue
        fig.add_trace(go.Scatter(
            x=hist['signal_date'], y=hist[col], name=label, mode='lines+markers',
            line=dict(color=colors.get(col_map.get(col, ''), '#888'), width=2), marker=dict(size=6),
        ))
    fig.update_layout(
        height=230, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(range=[0, 100], title='점수'),
        legend=dict(orientation='h', y=-0.25, font=dict(size=10)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='#E0E0E0'), yaxis_gridcolor='#E0E0E0',
    )
    return fig

def radar_chart(row: pd.Series, name: str) -> go.Figure:
    labels = [FACTOR_LABELS[c] for c in FACTOR_COLS if c in row.index]
    vals   = [row[c] for c in FACTOR_COLS if c in row.index]
    vals_c = vals + [vals[0]]; lbls_c = labels + [labels[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=lbls_c, fill='toself',
        fillcolor='rgba(33,150,243,0.15)', line=dict(color='#2196F3', width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10))),
        showlegend=False, height=280, margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def sector_pie(top_df: pd.DataFrame) -> go.Figure:
    counts  = top_df['sector'].value_counts()
    palette = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7',
               '#DDA0DD','#F0A500','#82C341','#5B8DB8','#E8927C']
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.4,
        marker=dict(colors=palette[:len(counts)]),
        textinfo='label+percent', textfont=dict(size=10),
    ))
    fig.update_layout(height=200, margin=dict(l=0,r=0,t=0,b=0),
                      showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
    return fig

def make_kospi_chart(kf: pd.DataFrame) -> go.Figure:
    """KOSPI + 200일 MA + 국면 배경 (최근 12개월)"""
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)
    df = kf[kf.index >= cutoff].copy()
    if df.empty or df['MA200'].isna().all():
        df = kf.dropna(subset=['MA200']).copy()
    dates  = df.index.tolist()
    ratios = ((df['Close'] / df['MA200']) - 1).tolist()
    def _rcolor(r):
        if r > 0.03:  return 'rgba(200,240,210,0.45)'
        elif r < -0.03: return 'rgba(255,200,200,0.45)'
        else:           return 'rgba(255,235,185,0.4)'
    shapes, seg_start, prev_r = [], dates[0] if dates else None, None
    for i, (dt, r) in enumerate(zip(dates, ratios)):
        cr = 'bull' if r > 0.03 else ('bear' if r < -0.03 else 'sw')
        if cr != prev_r:
            if prev_r is not None and seg_start:
                shapes.append(dict(type='rect', xref='x', yref='paper',
                    x0=seg_start, x1=dt, y0=0, y1=1,
                    fillcolor=_rcolor(ratios[dates.index(seg_start)]),
                    layer='below', line_width=0))
            seg_start, prev_r = dt, cr
    if seg_start and dates:
        shapes.append(dict(type='rect', xref='x', yref='paper',
            x0=seg_start, x1=dates[-1], y0=0, y1=1,
            fillcolor=_rcolor(ratios[-1]), layer='below', line_width=0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200일 MA',
        line=dict(color='#FF7043', width=1.5, dash='dash'), opacity=0.9))
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='KOSPI',
        line=dict(color='#1565C0', width=2.5)))
    fig.update_layout(
        shapes=shapes, height=265,
        margin=dict(l=0, r=0, t=38, b=0),
        xaxis=dict(showgrid=False, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor='rgba(200,200,200,0.4)', tickformat=','),
        legend=dict(orientation='h', y=1.12, x=0, font=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,249,250,0.8)',
        title=dict(text='KOSPI vs 200일 이동평균', font=dict(size=12)),
        hovermode='x unified')
    return fig

def make_backtest_chart(bt_df: pd.DataFrame) -> go.Figure:
    """5개 전략 누적수익률 vs 코스피"""
    if bt_df is None:
        return None
    df = bt_df.copy()
    date_col   = '매수일' if '매수일' in df.columns else 'sig_date'
    kospi_col  = '코스피_수익률(%)' if '코스피_수익률(%)' in df.columns else 'kospi_return'
    strat_map  = {'밸런스형': '밸런스형_수익률(%)', '저평가형': '저평가형_수익률(%)',
                  '모멘텀형': '모멘텀형_수익률(%)', '국면대응형': '국면대응형_수익률(%)',
                  '촉매형': '촉매형_수익률(%)'}
    palette    = {'밸런스형': '#2196F3', '저평가형': '#4CAF50',
                  '모멘텀형': '#FF5722', '국면대응형': '#9C27B0',
                  '촉매형': '#FF9800', '코스피': '#9E9E9E'}
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    fig = go.Figure()
    if kospi_col in df.columns:
        kc = (1 + df[kospi_col].fillna(0) / 100).cumprod()
        fig.add_trace(go.Scatter(x=df[date_col], y=(kc - 1) * 100, name='코스피',
            line=dict(color=palette['코스피'], width=1.5, dash='dot')))
    for strat, col in strat_map.items():
        if col not in df.columns: continue
        sc = (1 + df[col].fillna(0) / 100).cumprod()
        fig.add_trace(go.Scatter(x=df[date_col], y=(sc - 1) * 100, name=strat,
            line=dict(color=palette[strat], width=2)))
    fig.update_layout(
        height=265, margin=dict(l=0, r=0, t=38, b=0),
        yaxis=dict(gridcolor='rgba(200,200,200,0.4)', ticksuffix='%', title='누적수익률'),
        xaxis=dict(gridcolor='rgba(200,200,200,0.4)'),
        legend=dict(orientation='h', y=1.12, x=0, font=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,249,250,0.8)',
        title=dict(text='전략별 누적수익률 (백테스트 2015~)', font=dict(size=12)),
        hovermode='x unified')
    return fig

def score_color(val):
    if val >= 70:  return 'background-color:#E8F5E9; color:#1B5E20'
    elif val >= 40: return 'background-color:#FFF8E1; color:#BF360C'
    else:           return 'background-color:#FFEBEE; color:#B71C1C'

# ──────────────────────────────────────────
# 전략 비교 탭
# ──────────────────────────────────────────
def render_overview_tab(q_df: pd.DataFrame, name_dict: dict,
                        regime: str, bt_df=None):
    """5개 전략 상위 종목 한눈에 비교"""
    advice = REGIME_ADVICE[regime]
    st.caption(f"{REGIME_EMOJI[regime]} **{REGIME_KR[regime]}** — {advice['설명']}")
    st.subheader('전략별 상위 5종목')
    cols = st.columns(5)
    ticker_strats: dict = {}
    for i, strat in enumerate(['밸런스형', '저평가형', '모멘텀형', '국면대응형', '촉매형']):
        sn_col = f'{strat}_sn'
        is_rec = strat in advice['추천']
        border = advice['border'] if is_rec else '#9E9E9E'
        badge  = '✅ 추천' if is_rec else '보조'
        if strat == '모멘텀형':
            top5 = q_df.nlargest(5, sn_col)
        else:
            top5 = select_with_sector_cap(q_df, sn_col, 5, MAX_PER_SECTOR)
        with cols[i]:
            st.markdown(
                f'<div style="border:1px solid {border};border-radius:8px;padding:10px 12px;">'
                f'<div style="font-weight:bold;margin-bottom:6px;">{strat} '
                f'<span style="font-size:11px;color:{border};">{badge}</span></div>',
                unsafe_allow_html=True)
            for rank, (_, row) in enumerate(top5.iterrows(), 1):
                name  = name_dict.get(row['ticker'], row['ticker'])
                score = row[sn_col]
                ticker_strats.setdefault(row['ticker'], []).append(strat)
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="color:#888;font-size:12px;">#{rank}</span> '
                    f'<b style="font-size:13px;">{name}</b><br>'
                    f'<span style="color:#aaa;font-size:11px;">{row["sector"]} · {score:.0f}점</span>'
                    f'</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    multi = {t: s for t, s in ticker_strats.items() if len(s) >= 2}
    if multi:
        st.divider()
        st.subheader('2개 이상 전략 동시 편입')
        rows = []
        for ticker, strats in sorted(multi.items(), key=lambda x: -len(x[1])):
            rq = q_df[q_df['ticker'] == ticker]
            if rq.empty: continue
            r = rq.iloc[0]
            rows.append({'종목명': name_dict.get(ticker, ticker), '티커': ticker,
                         '섹터': r['sector'], '편입 전략': ' / '.join(strats),
                         '전략수': len(strats)})
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values('전략수', ascending=False),
                         use_container_width=True, hide_index=True)
    if bt_df is not None:
        st.divider()
        fig = make_backtest_chart(bt_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key='overview_bt')

# ──────────────────────────────────────────
# 전략 탭 렌더링
# ──────────────────────────────────────────
def render_strategy_tab(strat: str, df: pd.DataFrame, name_dict: dict,
                        bt_df=None, prev_tickers: set = None,
                        regime: str = 'sideways',
                        full_factor_df: pd.DataFrame = None,
                        quarter_label: str = ''):

    if strat == '모멘텀형':
        top = df.nlargest(STRATEGY_TOP_N[strat], f'{strat}_sn').copy()
    else:
        top = select_with_sector_cap(
            df, f'{strat}_sn', STRATEGY_TOP_N[strat], MAX_PER_SECTOR).copy()
    top['종목명'] = top['ticker'].map(name_dict).fillna(top['ticker'])

    current_tickers = set(top['ticker'].tolist())
    new_in  = current_tickers - prev_tickers if prev_tickers else set()
    removed = prev_tickers - current_tickers  if prev_tickers else set()

    # 종합점수 내림차순 정렬 (높은 점수 = 1위가 맨 위)
    top = top.sort_values(f'{strat}_sn', ascending=False).reset_index(drop=True)

    # ── 1. 한 줄 헤더 ──────────────────────────────────────
    advice         = REGIME_ADVICE[regime]
    is_recommended = strat in advice['추천']
    badge_color    = advice['border'] if is_recommended else '#9E9E9E'
    badge_text     = '✅ 추천' if is_recommended else '⬜ 보조'

    m = calc_strat_metrics(bt_df, strat) if bt_df is not None else {}

    hdr_parts = [
        f'<span style="border:1px solid {badge_color};color:{badge_color};'
        f'border-radius:4px;padding:2px 8px;font-size:12px;font-weight:bold;">{badge_text}</span>'
    ]
    if m:
        hdr_parts += [
            f'<span style="font-size:12px;color:#555;margin-left:12px;">CAGR <b>{m["cagr"]:+.1f}%</b></span>',
            f'<span style="font-size:12px;color:#555;margin-left:10px;">IR <b>{m["ir"]:.2f}</b></span>',
            f'<span style="font-size:12px;color:#555;margin-left:10px;">샤프 <b>{m["sharpe"]:.2f}</b></span>',
            f'<span style="font-size:12px;color:#555;margin-left:10px;">승률 <b>{m["win"]:.0f}%</b></span>',
        ]
    st.markdown('<div style="padding:6px 0;">' + ''.join(hdr_parts) + '</div>',
                unsafe_allow_html=True)

    # ── 2. 신규/제외 한 줄 요약 ──────────────────────────
    if prev_tickers:
        hold_n = len(current_tickers & prev_tickers)
        parts  = [f"🔄 연속 **{hold_n}**"]
        if new_in:
            names = sorted([name_dict.get(t, t) for t in new_in])
            parts.append(f"🆕 신규 **{len(new_in)}** — " + ' '.join(f'`{n}`' for n in names[:8]))
        if removed:
            names = sorted([name_dict.get(t, t) for t in removed])
            parts.append(f"🔴 제외 **{len(removed)}** — " + ' '.join(f'`{n}`' for n in names[:8]))
        st.caption('  　'.join(parts))

    # ── 3. 테이블 + 상세 패널 ────────────────────────────
    col_tbl, col_detail = st.columns([3, 2])

    with col_tbl:
        if prev_tickers:
            top['구분'] = top['ticker'].apply(lambda t: '🔄' if t in prev_tickers else '🆕')
            disp = top[['구분', '종목명', 'ticker', 'sector', strat] + FACTOR_COLS].copy()
            disp.columns = ['구분', '종목명', '티커', '섹터', '종합점수',
                            '성장', '가치', '퀄리티', '수익성', '모멘텀', '저변동성']
        else:
            disp = top[['종목명', 'ticker', 'sector', strat] + FACTOR_COLS].copy()
            disp.columns = ['종목명', '티커', '섹터', '종합점수',
                            '성장', '가치', '퀄리티', '수익성', '모멘텀', '저변동성']

        disp = disp.reset_index(drop=True)
        disp.index = [f'#{i+1}' for i in range(len(disp))]
        score_cols = ['성장', '가치', '퀄리티', '수익성', '모멘텀', '저변동성']
        styled = (
            disp.style
            .map(score_color, subset=score_cols)
            .format({'종합점수': '{:.1f}', '성장': '{:.0f}', '가치': '{:.0f}',
                     '퀄리티': '{:.0f}', '수익성': '{:.0f}', '모멘텀': '{:.0f}', '저변동성': '{:.0f}'})
        )
        event = st.dataframe(
            styled, use_container_width=True, height=530,
            on_select="rerun", selection_mode="single-row", key=f"tbl_{strat}",
        )

        dl_col, pie_col = st.columns([1, 2])
        with dl_col:
            csv_bytes = disp.to_csv(index=True, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button("⬇️ CSV", data=csv_bytes,
                               file_name=f"{strat}_{quarter_label}.csv",
                               mime='text/csv', key=f"dl_{strat}")
        with pie_col:
            st.caption("섹터 분포")
            st.plotly_chart(sector_pie(top), use_container_width=True, key=f"pie_{strat}")

    with col_detail:
        sel        = event.selection.rows if event.selection.rows else [0]
        row        = top.iloc[sel[0]]
        stock_name = row['종목명']
        ticker     = row['ticker']

        st.markdown(f"#### {stock_name} `{ticker}`")
        st.plotly_chart(make_stock_chart(ticker, stock_name), use_container_width=True,
                        key=f"price_{strat}_{sel[0]}")

        for col, label in FACTOR_LABELS.items():
            score = row[col]
            color = "#4CAF50" if score >= 70 else "#FF9800" if score >= 40 else "#F44336"
            st.markdown(
                f'<div style="margin-bottom:7px;">'
                f'<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px;">'
                f'<span><b>{label}</b></span>'
                f'<span style="color:{color};font-weight:bold;">{score:.0f}</span>'
                f'</div>'
                f'<div style="background:#E0E0E0;border-radius:4px;height:9px;">'
                f'<div style="width:{score:.0f}%;background:{color};border-radius:4px;height:9px;"></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        sector     = row['sector']
        sector_sub = df[df['sector'] == sector]
        sec_rank   = int((sector_sub[strat] > row[strat]).sum()) + 1
        st.caption(f"📌 {sector}  |  섹터 {sec_rank}/{len(sector_sub)}위")
        st.markdown(
            f"[🔗 네이버금융](https://finance.naver.com/item/main.naver?code={ticker})  "
            f"| [📄 공시](https://finance.naver.com/item/news.naver?code={ticker})"
        )

        with st.expander("📝 편입 이유"):
            st.markdown(generate_reason(row))

        if full_factor_df is not None:
            hist = (
                full_factor_df[full_factor_df['ticker'] == ticker]
                .sort_values('signal_date').tail(8)[['signal_date'] + FACTOR_COLS]
            )
            if len(hist) >= 2:
                st.markdown("---"); st.caption("팩터 추이 (최근 8분기)")
                hist = hist.copy()
                hist['signal_date'] = hist['signal_date'].dt.strftime('%Y-%m')
                st.plotly_chart(history_chart(hist), use_container_width=True,
                               key=f"hist_{strat}_{sel[0]}")


# ──────────────────────────────────────────
# 실거래 추적
# ──────────────────────────────────────────
def quarter_str(qe):
    return f"{qe.year}Q{(qe.month - 1) // 3 + 1}"

def _sb_headers():
    key = st.secrets.get('SUPABASE_KEY', '')
    return {'apikey': key, 'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'}

def _sb_url():
    return st.secrets.get('SUPABASE_URL', '')

def load_live_positions():
    import requests as _req
    _cols = ['quarter', 'snapshot_date', 'strategy', 'ticker',
             'name', 'sector', 'entry_price', 'exit_price', 'exit_date', 'status',
             'last_price', 'last_price_date']
    url = _sb_url()
    if url:
        try:
            import urllib.request as _ur, json as _js
            key = st.secrets.get('SUPABASE_KEY', '')
            req = _ur.Request(f'{url}/rest/v1/live_positions?select=*')
            req.add_header('apikey', key)
            req.add_header('Authorization', f'Bearer {key}')
            with _ur.urlopen(req, timeout=10) as resp:
                data = _js.loads(resp.read().decode('utf-8'))
            if data:
                df = pd.DataFrame(data)
                for c in _cols:
                    if c not in df.columns:
                        df[c] = None
                return df
        except Exception:
            pass
    if os.path.exists(LIVE_DB_PATH):
        df = pd.read_parquet(LIVE_DB_PATH)
        for c in _cols:
            if c not in df.columns:
                df[c] = None
        return df
    return pd.DataFrame(columns=_cols)

def save_live_positions(df):
    import requests as _req
    url = _sb_url()
    if url and not df.empty:
        try:
            records = []
            for _, row in df.iterrows():
                rec = {}
                for col in df.columns:
                    if col == 'id':
                        continue
                    val = row[col]
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        rec[col] = None
                    elif hasattr(val, 'item'):
                        rec[col] = val.item()
                    else:
                        rec[col] = str(val) if not isinstance(val, (int, float, str, bool)) else val
                records.append(rec)
            import urllib.request as _ur, json as _js
            key = st.secrets.get('SUPABASE_KEY', '')
            body = _js.dumps(records, ensure_ascii=False).encode('utf-8')
            req = _ur.Request(f'{url}/rest/v1/live_positions',
                              data=body, method='POST')
            req.add_header('apikey', key)
            req.add_header('Authorization', f'Bearer {key}')
            req.add_header('Content-Type', 'application/json')
            req.add_header('Prefer', 'resolution=merge-duplicates')
            with _ur.urlopen(req, timeout=15):
                pass
            return
        except Exception as e:
            st.toast(f'Supabase 오류: {e}', icon='⚠️')
    df.to_parquet(LIVE_DB_PATH, index=False)

def fetch_prices_now(tickers, progress_bar=None):
    end   = pd.Timestamp.today()
    start = (end - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end_s = end.strftime('%Y-%m-%d')
    out   = {}
    for i, t in enumerate(tickers):
        try:
            df = fdr.DataReader(t, start, end_s)
            out[t] = float(df['Close'].iloc[-1]) if not df.empty else None
        except Exception:
            out[t] = None
        if progress_bar:
            progress_bar.progress((i + 1) / max(len(tickers), 1))
    return out

@st.cache_data(ttl=3600)
def get_stock_price(ticker: str) -> pd.DataFrame:
    try:
        start = (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        df = fdr.DataReader(ticker, start)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df[['Close']].dropna()
    except Exception:
        return pd.DataFrame()

def make_stock_chart(ticker: str, name: str) -> go.Figure:
    df = get_stock_price(ticker)
    fig = go.Figure()
    if df.empty:
        fig.add_annotation(text='주가 데이터 없음', xref='paper', yref='paper',
                           x=0.5, y=0.5, showarrow=False)
    else:
        mn, mx = df['Close'].min(), df['Close'].max()
        first, last = df['Close'].iloc[0], df['Close'].iloc[-1]
        color = '#D32F2F' if last >= first else '#1565C0'
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], name=name,
            line=dict(color=color, width=2), fill='tozeroy',
            fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba') if 'rgb' in color
                       else ('rgba(211,47,47,0.08)' if color == '#D32F2F' else 'rgba(21,101,192,0.08)'),
        ))
        chg = (last / first - 1) * 100
        fig.update_layout(
            title=dict(text=f'{name}  <span style="font-size:13px;color:{color}">{last:,.0f}원  {chg:+.1f}%</span>',
                       font=dict(size=14)),
        )
    fig.update_layout(
        height=265, margin=dict(l=0, r=0, t=38, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,249,250,0.8)',
        xaxis=dict(showgrid=False, tickformat='%m/%d'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.07)',
                   tickformat=',.0f', side='right'),
        showlegend=False, hovermode='x unified',
    )
    return fig

def snapshot_quarter(live_df, q_df, current_q, name_dict):
    if current_q in live_df['quarter'].values:
        return live_df
    rows = []
    for strat, top_n in STRATEGY_TOP_N.items():
        sn = f'{strat}_sn'
        top = (q_df.nlargest(top_n, sn) if strat == '모멘텀형'
               else select_with_sector_cap(q_df, sn, top_n, MAX_PER_SECTOR))
        for _, r in top.iterrows():
            rows.append({
                'quarter': current_q,
                'snapshot_date': pd.Timestamp.today().date(),
                'strategy': strat, 'ticker': r['ticker'],
                'name': name_dict.get(r['ticker'], r['ticker']),
                'sector': r.get('sector', '기본'),
                'entry_price': None, 'exit_price': None,
                'exit_date': None, 'status': 'pending'})
    return pd.concat([live_df, pd.DataFrame(rows)], ignore_index=True)

def close_old_positions(live_df, current_q, progress_bar=None):
    mask    = (live_df['status'] == 'open') & (live_df['quarter'] != current_q)
    tickers = live_df.loc[mask, 'ticker'].unique().tolist()
    if not tickers:
        return live_df
    prices = fetch_prices_now(tickers, progress_bar)
    today  = pd.Timestamp.today().date()
    for idx in live_df[mask].index:
        p = prices.get(live_df.loc[idx, 'ticker'])
        if p:
            live_df.loc[idx, 'exit_price'] = p
            live_df.loc[idx, 'exit_date']  = today
            live_df.loc[idx, 'status']     = 'closed'
    return live_df

def render_tracking_tab(live_df, current_q, q_df, name_dict, bt_df=None):
    st.subheader('실거래 추적')
    st.caption('분기 추천 종목을 가상 매수매도로 기록해 백테스트와 비교합니다.')

    cur      = live_df[live_df['quarter'] == current_q]
    pend     = cur[cur['status'] == 'pending']
    open_pos = cur[cur['status'] == 'open']

    c_stat, c_btn = st.columns([3, 1])
    with c_stat:
        if cur.empty:
            st.info(f'**{current_q}** 스냅샷 없음 — 버튼으로 저장하세요.')
        elif not pend.empty:
            st.warning(f'**{current_q}** 진입가 미수집 ({len(pend)}개 종목)')
        else:
            st.success(f'**{current_q}** 추적 중  |  {len(open_pos)}개 종목')
    with c_btn:
        if cur.empty:
            if st.button('📸 스냅샷 저장', use_container_width=True):
                live_df = snapshot_quarter(live_df, q_df, current_q, name_dict)
                save_live_positions(live_df)
                st.rerun()
        elif not pend.empty:
            if st.button('💹 진입가 수집', use_container_width=True):
                tickers = pend['ticker'].unique().tolist()
                pb = st.progress(0, text='현재가 조회 중...')
                prices = fetch_prices_now(tickers, pb)
                mask2 = ((live_df['quarter'] == current_q) &
                          (live_df['status'] == 'pending'))
                for idx in live_df[mask2].index:
                    p = prices.get(live_df.loc[idx, 'ticker'])
                    if p:
                        live_df.loc[idx, 'entry_price'] = p
                        live_df.loc[idx, 'status'] = 'open'
                save_live_positions(live_df)
                st.rerun()

    old_open = live_df[(live_df['status'] == 'open') & (live_df['quarter'] != current_q)]
    if not old_open.empty:
        st.divider()
        qlist = ', '.join(sorted(old_open['quarter'].unique()))
        st.warning(f'이전 분기 미청산 포지션: **{qlist}**')
        if st.button('📤 이전 분기 자동 청산 (현재가)'):
            pb = st.progress(0, text='청산 중...')
            live_df = close_old_positions(live_df, current_q, pb)
            save_live_positions(live_df)
            st.rerun()

    if not open_pos.empty:
        st.divider()
        c_tit, c_upd = st.columns([4, 1])
        with c_tit:
            st.markdown('**현재 보유 종목**')
        with c_upd:
            if st.button('📊 현재가 갱신', use_container_width=True):
                tickers = open_pos['ticker'].unique().tolist()
                pb = st.progress(0, text='조회 중...')
                prices = fetch_prices_now(tickers, pb)
                today  = pd.Timestamp.today().date()
                for idx in live_df[live_df['quarter'] == current_q].index:
                    t = live_df.loc[idx, 'ticker']
                    p = prices.get(t)
                    if p:
                        live_df.loc[idx, 'last_price']      = p
                        live_df.loc[idx, 'last_price_date'] = str(today)
                save_live_positions(live_df)
                st.rerun()
        rows = []
        for strat in STRATEGY_TOP_N:
            sub = open_pos[open_pos['strategy'] == strat]
            for _, r in sub.iterrows():
                entry = r.get('entry_price')
                curr  = r.get('last_price')
                ldate = r.get('last_price_date', '')
                ret   = (curr / entry - 1) * 100 if (curr and entry) else None
                rows.append({
                    '전략': strat, '종목명': r['name'],
                    '티커': r['ticker'], '섹터': r['sector'],
                    '진입가': f"{int(entry):,}" if (pd.notna(entry) and entry) else '-',
                    '현재가': f"{int(curr):,}" if curr else '-',
                    '조회일': str(ldate) if ldate else '-',
                    '미실현(%)': f"{ret:+.1f}%" if ret is not None else '-',
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if not live_df.empty:
        csv_bytes = live_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        st.download_button(
            '📥 CSV 내보내기 (Supabase 업로드용)',
            data=csv_bytes,
            file_name=f'live_positions_{pd.Timestamp.today().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )

    closed = live_df[live_df['status'] == 'closed'].copy()
    if closed.empty:
        st.caption('아직 청산된 분기가 없습니다. 다음 리밸런싱 후 성과가 집계됩니다.')
        return

    closed['entry_price'] = pd.to_numeric(closed['entry_price'], errors='coerce')
    closed['exit_price']  = pd.to_numeric(closed['exit_price'],  errors='coerce')
    closed['ret']         = (closed['exit_price'] / closed['entry_price'] - 1) * 100
    perf = (closed.groupby(['quarter', 'strategy'])['ret']
            .mean().reset_index()
            .rename(columns={'ret': '실제(%)'}))

    if bt_df is not None:
        date_col = '매수일' if '매수일' in bt_df.columns else 'sig_date'
        bdf = bt_df.copy()
        bdf[date_col] = pd.to_datetime(bdf[date_col])
        bdf['quarter'] = bdf[date_col].apply(
            lambda d: f"{d.year}Q{(d.month - 1) // 3 + 1}")
        for strat in STRATEGY_TOP_N:
            col = f'{strat}_수익률(%)'
            if col not in bdf.columns:
                continue
            bt_q = bdf.groupby('quarter')[col].mean()
            m = perf['strategy'] == strat
            perf.loc[m, '백테스트(%)'] = perf.loc[m, 'quarter'].map(bt_q)
        perf['차이(%p)'] = (perf['실제(%)'] - perf['백테스트(%)']).round(1)

    st.divider()
    st.markdown('**분기별 실제 vs 백테스트**')
    st.dataframe(perf.sort_values(['quarter', 'strategy']).round(2),
                 use_container_width=True, hide_index=True)

    if len(closed['quarter'].unique()) >= 2:
        qret = closed.groupby(['quarter', 'strategy'])['ret'].mean().reset_index()
        pal  = {'밸런스형': '#2196F3', '저평가형': '#4CAF50',
                '모멘텀형': '#FF5722', '국면대응형': '#9C27B0'}
        fig  = go.Figure()
        for strat in STRATEGY_TOP_N:
            sub = qret[qret['strategy'] == strat].sort_values('quarter')
            if sub.empty:
                continue
            cum = (1 + sub['ret'] / 100).cumprod()
            fig.add_trace(go.Scatter(
                x=sub['quarter'], y=(cum - 1) * 100, name=strat,
                line=dict(color=pal.get(strat, '#888'), width=2), mode='lines+markers'))
        fig.update_layout(
            height=280, yaxis_ticksuffix='%', yaxis_title='누적수익률',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(248,249,250,0.8)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation='h', y=1.12, font=dict(size=11)),
            hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True, key='live_cumret')

# ══════════════════════════════════════════
# 메인
# ══════════════════════════════════════════
st.title("📊 퀀트 스코어링")

with st.spinner("데이터 로딩 중..."):
    factor_df  = load_factor_db()
    kospi_df   = load_kospi()
    kospi_full = load_kospi_full()
    name_dict  = get_name_dict()
    sector_fix = get_sector_dict()
    bt_df      = load_backtest()
    live_df    = load_live_positions()

if sector_fix:
    factor_df['sector'] = factor_df['ticker'].map(sector_fix).fillna(factor_df['sector'])

latest_signal  = factor_df['signal_date'].max()
latest_quarter = factor_df[factor_df['signal_date'] == latest_signal]['quarter_end'].iloc[0]
q_df      = factor_df[factor_df['signal_date'] == latest_signal].copy()
current_q = quarter_str(latest_quarter)

prev_signal = factor_df[factor_df['signal_date'] < latest_signal]['signal_date'].max()
prev_q_df   = factor_df[factor_df['signal_date'] == prev_signal].copy() if pd.notna(prev_signal) else None

regime, curr_price, ma200, ratio = get_regime(kospi_df)

# 오늘의 KOSPI 지표
kf_last  = kospi_full.iloc[-1] if not kospi_full.empty else None
kf_prev  = kospi_full.iloc[-2] if len(kospi_full) >= 2 else kf_last
ks_close = float(kf_last['Close']) if kf_last is not None else 0
ks_chg   = ((kf_last['Close'] / kf_prev['Close']) - 1) * 100 if (kf_last is not None and kf_prev is not None) else 0
ks_hi    = float(kf_last.get('High', ks_close)) if kf_last is not None else 0
ks_lo    = float(kf_last.get('Low',  ks_close)) if kf_last is not None else 0
ks_date  = kospi_full.index[-1].strftime('%m/%d') if not kospi_full.empty else ''
next_reb  = next_rebalancing(latest_quarter)
days_left = (next_reb - pd.Timestamp.today()).days

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric('시장 국면', f'{REGIME_EMOJI[regime]} {REGIME_KR[regime]}')
c2.metric(f'KOSPI ({ks_date})', f'{ks_close:,.0f}', delta=f'{ks_chg:+.2f}%')
c3.metric('일중 고가', f'{ks_hi:,.0f}')
c4.metric('일중 저가', f'{ks_lo:,.0f}')
c5.metric('200MA 대비', f'{ratio:+.1f}%')
c6.metric('다음 리밸런싱', next_reb.strftime('%m/%d'), delta=f'{max(days_left,0)}일 후')

# 시황 차트 (KOSPI + 백테스트)
chart_c1, chart_c2 = st.columns(2)
with chart_c1:
    st.plotly_chart(make_kospi_chart(kospi_full), use_container_width=True, key='main_kospi')
with chart_c2:
    bt_fig = make_backtest_chart(bt_df)
    if bt_fig:
        st.plotly_chart(bt_fig, use_container_width=True, key='main_bt')
    else:
        st.info('백테스트 결과 CSV가 없습니다.')

st.caption(
    f"📅 데이터 기준: **{latest_quarter.strftime('%Y년 %m월')} 실적**  "
    f"(signal date {latest_signal.strftime('%Y-%m-%d')})  |  유니버스 **{len(q_df):,}개 종목**"
)

with st.expander("📖 팩터 점수 설명"):
    cols_desc = st.columns(3)
    for i, (col, desc) in enumerate(FACTOR_DESC.items()):
        with cols_desc[i % 3]:
            st.markdown(f"#### {FACTOR_LABELS[col]}")
            st.markdown(desc)

st.divider()

q_df = compute_scores(q_df, regime)

prev_top: dict[str, set] = {}
if prev_q_df is not None:
    prev_q_df = compute_scores(prev_q_df, regime)
    for strat, n in STRATEGY_TOP_N.items():
        if strat == '모멘텀형':
            prev_top[strat] = set(prev_q_df.nlargest(n, f'{strat}_sn')['ticker'].tolist())
        else:
            prev_top[strat] = set(
                select_with_sector_cap(prev_q_df, f'{strat}_sn', n, MAX_PER_SECTOR)['ticker'].tolist())

tab_labels = {
    '밸런스형':   "⚖️ 밸런스형",
    '저평가형':   "💰 저평가형",
    '모멘텀형':   "🚀 모멘텀형",
    '국면대응형': f"🧭 국면대응형  ({REGIME_KR[regime]})",
    '촉매형':     "⚡ 촉매형",
}
tab_overview_label = '📋 전략 비교'
tab_search_label   = '🔍 종목 검색'
tab_track_label    = '📈 실거래 추적'
all_tabs      = st.tabs(list(tab_labels.values()) + [tab_overview_label, tab_search_label, tab_track_label])
strategy_tabs = all_tabs[:5]
overview_tab  = all_tabs[5]
search_tab    = all_tabs[6]
track_tab     = all_tabs[7]

q_label = latest_quarter.strftime('%Y%m')
for (strat, _), tab in zip(tab_labels.items(), strategy_tabs):
    with tab:
        render_strategy_tab(
            strat, q_df, name_dict,
            bt_df=bt_df, prev_tickers=prev_top.get(strat),
            regime=regime, full_factor_df=factor_df, quarter_label=q_label,
        )

with overview_tab:
    render_overview_tab(q_df, name_dict, regime, bt_df)

# ══════════════════════════════════════════
# 종목 검색 탭
# ══════════════════════════════════════════
with search_tab:
    st.subheader("🔍 종목 검색")
    st.caption("종목명 또는 티커로 검색하면 팩터 점수와 전략별 순위를 확인할 수 있습니다.")

    query = st.text_input("종목명 또는 티커 입력 (예: 삼성전자, 005930)", placeholder="검색어 입력...")

    if query:
        query        = query.strip()
        reverse_dict = {v: k for k, v in name_dict.items()}
        matched_ticker = None

        if query in q_df['ticker'].values:
            matched_ticker = query
        elif query in reverse_dict:
            matched_ticker = reverse_dict[query]
        else:
            name_matches = [t for name, t in reverse_dict.items() if query in name]
            tick_matches = [t for t in name_dict.keys() if query in t]
            candidates   = list(dict.fromkeys(name_matches + tick_matches))
            if len(candidates) == 1:
                matched_ticker = candidates[0]
            elif len(candidates) > 1:
                labels = [f"{name_dict.get(t, t)} ({t})" for t in candidates[:20]]
                choice = st.selectbox("검색 결과", labels)
                matched_ticker = candidates[labels.index(choice)]
            else:
                st.warning("일치하는 종목이 없습니다.")

        if matched_ticker:
            row_curr = q_df[q_df['ticker'] == matched_ticker]
            _data_note = ''
            if row_curr.empty:
                _hist = factor_df[factor_df['ticker'] == matched_ticker].sort_values('signal_date')
                if not _hist.empty:
                    _ls      = _hist['signal_date'].max()
                    _hq      = _hist[_hist['signal_date'] == _ls].copy()
                    if sector_fix:
                        _hq['sector'] = _hq['ticker'].map(sector_fix).fillna(_hq['sector'])
                    row_curr   = compute_scores(_hq, regime)
                    _data_note = f"현재 유니버스 외 종목 \u2014 마지막 데이터: {_ls.strftime('%Y년 %m월')}"
                else:
                    st.info(
                        f"**{name_dict.get(matched_ticker, matched_ticker)}** ({matched_ticker}) \u2014 "
                        "팩터 DB에 데이터가 없습니다. (신규 상장 또는 상장 요건 미충족)"
                    )
            if not row_curr.empty:
                row        = row_curr.iloc[0]
                stock_name = name_dict.get(matched_ticker, matched_ticker)

                st.markdown(f"### {stock_name} `{matched_ticker}`")
                if _data_note:
                    st.warning(_data_note)
                _sig = pd.to_datetime(row.get('signal_date', latest_signal)).strftime('%Y-%m-%d')
                st.caption(f"섹터: **{row.get('sector', '-')}**  |  기준: {_sig}")

                st.markdown("**전략별 순위**")
                rank_cols = st.columns(4)
                for i, (strat, n) in enumerate(STRATEGY_TOP_N.items()):
                    sn_col = f'{strat}_sn'
                    total  = len(q_df)
                    ticker_sn = q_df.loc[q_df['ticker'] == matched_ticker, sn_col]
                    if ticker_sn.empty: continue
                    rank   = int((q_df[sn_col] > ticker_sn.values[0]).sum()) + 1
                    in_top = rank <= n
                    rank_cols[i].metric(strat, f"#{rank}" if in_top else f"{rank:,}/{total:,}위",
                                        "✅ 편입" if in_top else "")

                st.divider()

                col_radar, col_bars = st.columns([1, 1])
                with col_radar:
                    st.plotly_chart(make_stock_chart(matched_ticker, stock_name),
                                   use_container_width=True, key="search_price")

                with col_bars:
                    st.markdown("**팩터 점수 상세**")
                    universe_means = q_df[FACTOR_COLS].mean()
                    for col, label in FACTOR_LABELS.items():
                        score   = row[col]
                        avg_val = universe_means[col]
                        color   = "#4CAF50" if score >= 70 else "#FF9800" if score >= 40 else "#F44336"
                        pct     = int((q_df[col] < score).mean() * 100)
                        hint    = _SHORT_DESC.get(col, '')
                        st.markdown(
                            f'<div style="margin-bottom:10px;">'
                            f'<div style="display:flex;justify-content:space-between;font-size:13px;">'
                            f'<span><b>{label}</b> <span style="color:#999;font-size:11px;">{hint}</span></span>'
                            f'<span style="color:{color};font-weight:bold;">{score:.0f}점</span>'
                            f'<span style="color:#888;font-size:11px;">상위 {100-pct}%</span>'
                            f'</div>'
                            f'<div style="background:#E0E0E0;border-radius:4px;height:10px;position:relative;">'
                            f'<div style="width:{score:.0f}%;background:{color};border-radius:4px;height:10px;"></div>'
                            f'<div style="position:absolute;top:0;left:{avg_val:.0f}%;width:2px;height:10px;background:#333;"></div>'
                            f'</div>'
                            f'<div style="font-size:10px;color:#888;">유니버스 평균: {avg_val:.0f}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.divider()
                with st.expander("📝 팩터 분석 요약"):
                    st.markdown(generate_reason(row))

                st.markdown("**팩터 점수 추이 (최근 8분기)**")
                hist = (
                    factor_df[factor_df['ticker'] == matched_ticker]
                    .sort_values('signal_date').tail(8)[['signal_date'] + FACTOR_COLS]
                )
                if len(hist) >= 2:
                    hist = hist.copy()
                    hist['signal_date'] = hist['signal_date'].dt.strftime('%Y-%m')
                    st.plotly_chart(history_chart(hist), use_container_width=True, key="search_hist")
                else:
                    st.caption("히스토리 데이터가 부족합니다.")

with track_tab:
    render_tracking_tab(live_df, current_q, q_df, name_dict, bt_df)
