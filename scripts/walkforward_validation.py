"""
워크포워드 과적합 검증 스크립트
- 전반기(2016-2020) vs 후반기(2021-2025) 성과 비교
- 연도별 IR 일관성 분석
- 국면별 서브기간 비교
"""
import os
import glob
import numpy as np
import pandas as pd

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
RISK_FREE_QTR = 0.035 / 4

# ── 최신 백테스트 결과 로드 ──────────────────────────────────
files = sorted(glob.glob(os.path.join(RESULT_DIR, 'backtest_detail_*.csv')))
if not files:
    raise FileNotFoundError("backtest_detail_*.csv 없음")

latest = files[-1]
print(f"📂 로드: {os.path.basename(latest)}\n")
df = pd.read_csv(latest)

STRATEGIES = ['밸런스형', '저평가형', '모멘텀형', '국면대응형']
REGIMES    = {'강세장': 'bull', '횡보장': 'sideways', '약세장': 'bear'}

# ── 지표 계산 함수 ───────────────────────────────────────────
def calc_metrics(sub: pd.DataFrame, strat: str) -> dict:
    ret_col   = f'{strat}_수익률(%)'
    kospi_col = '코스피_수익률(%)'
    if ret_col not in sub.columns or sub.empty:
        return {}

    rets   = sub[ret_col].values
    kospi  = sub[kospi_col].values
    excess = rets - kospi
    n      = len(rets)

    sharpe = (rets.mean() - RISK_FREE_QTR * 100) / (rets.std() + 1e-9)
    ir     = excess.mean() / (excess.std() + 1e-9)
    win    = (excess > 0).mean() * 100
    cagr   = (np.prod(1 + rets / 100) ** (4 / n) - 1) * 100

    return {
        'n':        n,
        'avg_ret':  round(rets.mean(), 2),
        'avg_exc':  round(excess.mean(), 2),
        'cagr':     round(cagr, 1),
        'sharpe':   round(sharpe, 2),
        'ir':       round(ir, 2),
        'win_pct':  round(win, 1),
    }

# ── 기간 정의 ────────────────────────────────────────────────
PERIODS = {
    '전체 (2016-2025)':    (2016, 2025),
    '전반기 (2016-2020)':  (2016, 2020),
    '후반기 (2021-2025)':  (2021, 2025),
}

# ═══════════════════════════════════════════════════════════
# 1. 기간별 전략 성과 비교
# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("【기간별 전략 성과 — 전반기 vs 후반기】")
print("=" * 65)

period_results = {}
for label, (y_start, y_end) in PERIODS.items():
    sub = df[(df['year'] >= y_start) & (df['year'] <= y_end)]
    period_results[label] = {}
    for strat in STRATEGIES:
        period_results[label][strat] = calc_metrics(sub, strat)

# 출력
for strat in STRATEGIES:
    print(f"\n▶ {strat}")
    print(f"  {'기간':<22} {'분기':>4} {'평균수익':>8} {'코스피초과':>9} {'샤프':>6} {'IR':>6} {'승률':>7}")
    print(f"  {'-'*62}")
    for label, m_dict in period_results.items():
        m = m_dict[strat]
        if not m:
            continue
        print(f"  {label:<22} {m['n']:>4}  {m['avg_ret']:>7.1f}%  {m['avg_exc']:>8.1f}%p  "
              f"{m['sharpe']:>5.2f}  {m['ir']:>5.2f}  {m['win_pct']:>6.1f}%")

# ═══════════════════════════════════════════════════════════
# 2. 국면 × 기간 교차 분석
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("【국면 × 기간 IR 비교 (과적합 핵심 지표)】")
print("=" * 65)
print("  IR이 전반기와 후반기에서 비슷하면 → 과적합 아님 ✅")
print("  전반기만 높고 후반기에서 급락하면 → 과적합 의심 ⚠️\n")

regime_labels = list(REGIMES.keys())
early  = df[(df['year'] >= 2016) & (df['year'] <= 2020)]
recent = df[(df['year'] >= 2021) & (df['year'] <= 2025)]

for strat in STRATEGIES:
    print(f"▶ {strat}")
    print(f"  {'국면':<8} {'전체 IR':>8} {'전반 IR':>8} {'후반 IR':>8}  판정")
    print(f"  {'-'*50}")
    for regime in regime_labels:
        full_m   = calc_metrics(df[df['시장국면'] == regime], strat)
        early_m  = calc_metrics(early[early['시장국면'] == regime], strat)
        recent_m = calc_metrics(recent[recent['시장국면'] == regime], strat)

        if not full_m:
            continue

        ir_full   = full_m['ir']
        ir_early  = early_m.get('ir', float('nan'))
        ir_recent = recent_m.get('ir', float('nan'))

        # 판정: 후반기 IR이 전반기의 50% 이상이면 안정적
        if np.isnan(ir_early) or np.isnan(ir_recent):
            verdict = '데이터 부족'
        elif ir_early <= 0 and ir_recent <= 0:
            verdict = '양쪽 다 부진'
        elif ir_early > 0 and ir_recent < ir_early * 0.5:
            verdict = '⚠️ 과적합 의심'
        elif ir_early > 0 and ir_recent >= ir_early * 0.5:
            verdict = '✅ 안정적'
        elif ir_early <= 0 and ir_recent > 0:
            verdict = '후반 개선'
        else:
            verdict = '확인 필요'

        print(f"  {regime:<8} {ir_full:>8.2f} {ir_early:>8.2f} {ir_recent:>8.2f}  {verdict}")
    print()

# ═══════════════════════════════════════════════════════════
# 3. 연도별 초과수익 일관성
# ═══════════════════════════════════════════════════════════
print("=" * 65)
print("【연도별 코스피 초과수익률 (%p) — 일관성 확인】")
print("=" * 65)

years = sorted(df['year'].unique())
header = f"  {'연도':>4} {'국면':>5}"
for s in STRATEGIES:
    header += f"  {s[:4]:>6}"
print(header)
print(f"  {'-'*55}")

for yr in years:
    yr_df   = df[df['year'] == yr]
    regime_counts = yr_df['시장국면'].value_counts()
    dominant_regime = regime_counts.index[0] if len(regime_counts) > 0 else '-'
    regime_short = {'강세장': '강세', '횡보장': '횡보', '약세장': '약세'}.get(dominant_regime, '-')

    row = f"  {yr:>4} {regime_short:>5}"
    for strat in STRATEGIES:
        ret_col = f'{strat}_수익률(%)'
        exc = (yr_df[ret_col] - yr_df['코스피_수익률(%)']).mean()
        flag = ' ' if abs(exc) < 10 else ('↑' if exc > 0 else '↓')
        row += f"  {exc:>+5.1f}{flag}"
    print(row)

# ═══════════════════════════════════════════════════════════
# 4. 과적합 종합 판정
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("【종합 판정】")
print("=" * 65)

for strat in STRATEGIES:
    full_all   = calc_metrics(df, strat)
    early_all  = calc_metrics(early, strat)
    recent_all = calc_metrics(recent, strat)

    ir_f = full_all['ir']
    ir_e = early_all['ir']
    ir_r = recent_all['ir']

    # 후반기 IR이 전반기 대비 얼마나 유지됐는지
    if ir_e > 0:
        retention = ir_r / ir_e
        if retention >= 0.8:
            verdict = "✅ 양호 (후반기 IR 80%↑ 유지)"
        elif retention >= 0.5:
            verdict = "△ 보통 (후반기 IR 50~80% 유지)"
        else:
            verdict = "⚠️ 과적합 의심 (후반기 IR 50% 미만)"
    else:
        if ir_r > 0:
            verdict = "△ 후반기에 개선됨"
        else:
            verdict = "❌ 양쪽 모두 부진"

    print(f"  {strat:<8}  전체 IR {ir_f:>5.2f}  |  전반 {ir_e:>5.2f} → 후반 {ir_r:>5.2f}  |  {verdict}")

print("\n✅ 검증 완료")
