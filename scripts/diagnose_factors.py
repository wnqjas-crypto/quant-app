"""
팩터 진단 스크립트
- 국면별 팩터 점수 분포 분석
- 전략별 포트폴리오 구성 추적
- 문제 원인 파악
"""
import os
import warnings
import numpy as np
import pandas as pd
import FinanceDataReader as fdr

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR   = os.path.join(BASE_DIR, 'db')
FACTOR_DB = os.path.join(DB_DIR, 'factor_db.parquet')
PRICE_DB  = os.path.join(DB_DIR, 'price_db.parquet')

# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────
print("📦 팩터 DB 로드 중...")
factor_df = pd.read_parquet(FACTOR_DB)
factor_df['ticker'] = factor_df['ticker'].astype(str).str.zfill(6)
factor_df['signal_date'] = pd.to_datetime(factor_df['signal_date'])
print(f"✅ {len(factor_df):,}건, {factor_df['signal_date'].nunique()}분기")

FACTOR_COLS = ['growth_score', 'value_score', 'quality_score',
               'profitability_score', 'momentum_score', 'lowvol_score']

# ──────────────────────────────────────────────
# 2. 시장 국면 판별
# ──────────────────────────────────────────────
print("\n📈 KOSPI 로드 중...")
kospi = fdr.DataReader('KS11', '2014-01-01', '2026-12-31')
kospi['MA200'] = kospi['Close'].rolling(200).mean()

def get_regime(date):
    dt = pd.to_datetime(date)
    row = kospi.loc[kospi.index <= dt].iloc[-1] if any(kospi.index <= dt) else None
    if row is None or pd.isna(row['MA200']):
        return 'unknown'
    ratio = row['Close'] / row['MA200']
    if ratio >= 1.03:
        return 'bull'
    elif ratio <= 0.97:
        return 'bear'
    return 'sideways'

signal_dates = sorted(factor_df['signal_date'].unique())
regime_map = {d: get_regime(d) for d in signal_dates}
factor_df['regime'] = factor_df['signal_date'].map(regime_map)
print(f"✅ 국면 판별 완료")

# ──────────────────────────────────────────────
# 3. 전략 가중치
# ──────────────────────────────────────────────
WEIGHTS = {
    '밸런스형':   {'growth_score': 0.20, 'value_score': 0.20, 'quality_score': 0.20,
                   'profitability_score': 0.15, 'momentum_score': 0.15, 'lowvol_score': 0.10},
    '저평가형':   {'growth_score': 0.10, 'value_score': 0.35, 'quality_score': 0.20,
                   'profitability_score': 0.25, 'momentum_score': 0.00, 'lowvol_score': 0.10},
    '모멘텀형':   {'growth_score': 0.25, 'value_score': 0.00, 'quality_score': 0.10,
                   'profitability_score': 0.10, 'momentum_score': 0.45, 'lowvol_score': 0.10},
    '국면대응형_bull':     {'growth_score': 0.25, 'value_score': 0.05, 'quality_score': 0.15,
                            'profitability_score': 0.15, 'momentum_score': 0.30, 'lowvol_score': 0.10},
    '국면대응형_sideways': {'growth_score': 0.15, 'value_score': 0.20, 'quality_score': 0.25,
                            'profitability_score': 0.15, 'momentum_score': 0.15, 'lowvol_score': 0.10},
    '국면대응형_bear':     {'growth_score': 0.05, 'value_score': 0.40, 'quality_score': 0.25,
                            'profitability_score': 0.20, 'momentum_score': 0.00, 'lowvol_score': 0.10},
}
TOP_N = {'밸런스형': 20, '저평가형': 20, '모멘텀형': 15}

# ──────────────────────────────────────────────
# 4. 섹터 중립화 함수 (백테스트와 동일)
# ──────────────────────────────────────────────
def sector_neutralize(df, score_col):
    df = df.copy()
    df['sn'] = df.groupby('sector')[score_col].rank(pct=True) * 100
    return df

# ──────────────────────────────────────────────
# 5. 국면별 팩터 점수 분포
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("【국면별 팩터 점수 분포】")
print("="*60)

regime_labels = {'bull': '강세장', 'sideways': '횡보장', 'bear': '약세장'}
for regime, label in regime_labels.items():
    sub = factor_df[factor_df['regime'] == regime]
    n_quarters = sub['signal_date'].nunique()
    n_stocks   = len(sub)
    print(f"\n▶ {label} ({n_quarters}분기, 총 {n_stocks:,}종목-분기)")
    print(f"  {'팩터':<20} {'평균':>6} {'표준편차':>8} {'100점비율':>9} {'0점비율':>8} {'중앙값':>7}")
    print(f"  {'-'*62}")
    for col in FACTOR_COLS:
        vals = sub[col].dropna()
        mean_v  = vals.mean()
        std_v   = vals.std()
        pct100  = (vals >= 99.9).mean() * 100
        pct0    = (vals <= 0.1).mean()  * 100
        median_v = vals.median()
        print(f"  {col:<20} {mean_v:>6.1f} {std_v:>8.1f} {pct100:>8.1f}% {pct0:>7.1f}% {median_v:>7.1f}")

# ──────────────────────────────────────────────
# 6. 전략 × 국면별 포트폴리오 팩터 기여 분석
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("【전략 × 국면별 포트폴리오 팩터 프로파일】")
print("  (선택된 Top-N 종목들의 평균 팩터 점수)")
print("="*60)

results = []
for sd in sorted(signal_dates):
    regime = regime_map[sd]
    q_df = factor_df[factor_df['signal_date'] == sd].copy()

    for strat, w in [('밸런스형', WEIGHTS['밸런스형']),
                     ('저평가형', WEIGHTS['저평가형']),
                     ('모멘텀형', WEIGHTS['모멘텀형'])]:

        # 섹터 중립화 (모멘텀형 제외)
        raw_score = sum(q_df[f] * w[f] for f in FACTOR_COLS)
        q_df[f'{strat}_raw'] = raw_score

        if strat != '모멘텀형':
            q_df = sector_neutralize(q_df, f'{strat}_raw')
            score_col = 'sn'
        else:
            q_df['sn'] = raw_score
            score_col = 'sn'

        n = TOP_N[strat]
        top = q_df.nlargest(n, score_col)

        row = {'signal_date': sd, 'regime': regime, 'strategy': strat}
        for f in FACTOR_COLS:
            row[f'{f}_mean'] = top[f].mean()
        row['composite_mean'] = top[score_col].mean()
        results.append(row)

res_df = pd.DataFrame(results)

for strat in ['밸런스형', '저평가형', '모멘텀형']:
    print(f"\n▶ {strat}")
    print(f"  {'국면':<8} {'분기수':>5} {'성장':>6} {'가치':>6} {'퀄리티':>7} {'수익성':>7} {'모멘텀':>7} {'저변동':>7}")
    print(f"  {'-'*60}")
    for regime, label in regime_labels.items():
        sub = res_df[(res_df['strategy'] == strat) & (res_df['regime'] == regime)]
        if sub.empty:
            continue
        nq = len(sub)
        means = sub[[f'{f}_mean' for f in FACTOR_COLS]].mean()
        vals = [f"{means[f'{f}_mean']:>6.1f}" for f in FACTOR_COLS]
        print(f"  {label:<8} {nq:>5}   {'  '.join(vals)}")

# ──────────────────────────────────────────────
# 7. 약세장 밸런스형 vs 전체 분포 비교 (핵심 문제 진단)
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("【핵심 진단: 약세장 밸런스형 포트폴리오 vs 전체】")
print("="*60)

bear_dates = [d for d, r in regime_map.items() if r == 'bear']
bull_dates  = [d for d, r in regime_map.items() if r == 'bull']

if bear_dates:
    bear_all   = factor_df[factor_df['signal_date'].isin(bear_dates)]
    bear_res   = res_df[(res_df['strategy'] == '밸런스형') & (res_df['regime'] == 'bear')]
    bull_res   = res_df[(res_df['strategy'] == '밸런스형') & (res_df['regime'] == 'bull')]

    print(f"\n  약세장 전체 유니버스 팩터 평균:")
    for f in FACTOR_COLS:
        mean_all  = bear_all[f].mean()
        mean_port = bear_res[f'{f}_mean'].mean() if not bear_res.empty else float('nan')
        diff      = mean_port - mean_all
        arrow     = '▲' if diff > 3 else ('▼' if diff < -3 else ' ')
        print(f"    {f:<22} 유니버스:{mean_all:5.1f}  포트폴리오:{mean_port:5.1f}  차이:{arrow}{diff:+.1f}")

    print(f"\n  강세장 밸런스형 포트폴리오 팩터 평균 (비교용):")
    for f in FACTOR_COLS:
        mean_bull = bull_res[f'{f}_mean'].mean() if not bull_res.empty else float('nan')
        mean_bear = bear_res[f'{f}_mean'].mean() if not bear_res.empty else float('nan')
        diff      = mean_bear - mean_bull
        print(f"    {f:<22} 강세:{mean_bull:5.1f}  약세:{mean_bear:5.1f}  차이:{diff:+.1f}")

# ──────────────────────────────────────────────
# 8. 점수 분포: 팩터별 100점 종목 수 추이 (분기별)
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("【분기별 100점(≥99.9) 종목 수 — 점수 인플레이션 체크】")
print("="*60)

quarterly_inflation = []
for sd in sorted(signal_dates):
    q = factor_df[factor_df['signal_date'] == sd]
    row = {'quarter': str(sd)[:10], 'total': len(q), 'regime': regime_map[sd]}
    for f in FACTOR_COLS:
        row[f] = (q[f] >= 99.9).sum()
    quarterly_inflation.append(row)

inf_df = pd.DataFrame(quarterly_inflation)
print(f"\n  {'분기':<12} {'국면':6} {'전체':5} {'성장':5} {'가치':5} {'퀄리티':6} {'수익성':6} {'모멘텀':6} {'저변동':6}")
print(f"  {'-'*65}")
for _, row in inf_df.iterrows():
    r_label = {'bull':'강세','sideways':'횡보','bear':'약세'}.get(row['regime'], row['regime'])
    print(f"  {row['quarter']:<12} {r_label:<6} {row['total']:5.0f} "
          f"{row['growth_score']:5.0f} {row['value_score']:5.0f} "
          f"{row['quality_score']:6.0f} {row['profitability_score']:6.0f} "
          f"{row['momentum_score']:6.0f} {row['lowvol_score']:6.0f}")

# 요약
print(f"\n  ▶ 분기 평균 100점 종목 수:")
for f in FACTOR_COLS:
    avg = inf_df[f].mean()
    max_v = inf_df[f].max()
    max_q = inf_df.loc[inf_df[f].idxmax(), 'quarter']
    print(f"    {f:<22} 평균 {avg:5.1f}개  최대 {max_v:5.0f}개 ({max_q})")

# ──────────────────────────────────────────────
# 9. 횡보장 모멘텀형 분석
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("【핵심 진단: 횡보장 모멘텀형 포트폴리오 vs 강세장 비교】")
print("="*60)

sideways_res = res_df[(res_df['strategy'] == '모멘텀형') & (res_df['regime'] == 'sideways')]
bull_mom_res = res_df[(res_df['strategy'] == '모멘텀형') & (res_df['regime'] == 'bull')]

print(f"\n  {'팩터':<22} {'강세장':>8} {'횡보장':>8} {'차이':>7}")
print(f"  {'-'*48}")
for f in FACTOR_COLS:
    b_mean = bull_mom_res[f'{f}_mean'].mean() if not bull_mom_res.empty else float('nan')
    s_mean = sideways_res[f'{f}_mean'].mean() if not sideways_res.empty else float('nan')
    diff   = s_mean - b_mean
    print(f"  {f:<22} {b_mean:>8.1f} {s_mean:>8.1f} {diff:>+7.1f}")

print("\n✅ 진단 완료")
