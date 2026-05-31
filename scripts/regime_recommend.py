import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

RF     = 0.035 / 4
STRATS = ['밸런스형', '저평가형', '모멘텀형', '국면대응형']

df = pd.read_csv('results/backtest_detail_20260530_023533.csv')

def calc(sub, s):
    col = f'{s}_수익률(%)'
    if sub.empty or col not in sub.columns:
        return {}
    r   = sub[col].values
    k   = sub['코스피_수익률(%)'].values
    exc = r - k
    return {
        'n':       len(r),
        'cagr':    round(((1 + r.mean() / 100) ** 4 - 1) * 100, 1),
        'avg_exc': round(exc.mean(), 2),
        'sharpe':  round((r.mean() - RF * 100) / (r.std() + 1e-9), 2),
        'ir':      round(exc.mean() / (exc.std() + 1e-9), 2),
        'win':     round((exc > 0).mean() * 100, 1),
        'avg_ret': round(r.mean(), 2),
    }

# ── 국면별 상세 ──────────────────────────────────────────────
print('=' * 68)
print('【국면별 전략 성과 (섹터 수정 후 최신 백테스트)】')
print('=' * 68)

regime_winner = {}  # 국면 → 추천 전략
for reg in ['강세장', '횡보장', '약세장']:
    sub = df[df['시장국면'] == reg]
    print(f'\n▶ {reg}  ({len(sub)}분기)')
    print(f"  {'전략':<8} {'연환산수익':>9} {'코스피초과':>9} {'샤프':>6} {'IR':>6} {'승률':>7}")
    print('  ' + '-' * 52)
    results = []
    for s in STRATS:
        mm = calc(sub, s)
        if not mm:
            continue
        results.append((s, mm))
        print(f"  {s:<8} {mm['cagr']:>+8.1f}%  {mm['avg_exc']:>+8.2f}%p"
              f"  {mm['sharpe']:>5.2f}  {mm['ir']:>5.2f}  {mm['win']:>6.1f}%")

    # IR 기준 1·2위
    ranked = sorted(results, key=lambda x: x[1]['ir'], reverse=True)
    regime_winner[reg] = ranked
    medals = ['🥇', '🥈', '🥉', '  ']
    print()
    for i, (s, mm) in enumerate(ranked):
        m = medals[i] if i < 3 else '  '
        print(f"  {m} {s}  IR {mm['ir']:+.2f}  CAGR {mm['cagr']:+.1f}%  승률 {mm['win']:.0f}%")

# ── 종합 추천 ─────────────────────────────────────────────────
print()
print('=' * 68)
print('【국면별 전략 추천 요약】')
print('=' * 68)

advice = {
    '강세장': {
        'desc': '200MA +3% 이상 상승 지속',
        'why':  '추세 추종 팩터가 정확히 작동. 모멘텀·성장 비중 높은 전략 유리.',
        'caution': '과열 신호 시 국면대응형으로 전환 대비',
    },
    '횡보장': {
        'desc': '200MA ±3% 범위 내 등락',
        'why':  '방향성 없는 장에서는 개별 종목 선별력이 승부처. 균형·가치 유리.',
        'caution': '모멘텀 전략은 횡보장에서 상대적으로 불리',
    },
    '약세장': {
        'desc': '200MA -3% 이하 하락 지속',
        'why':  '절대적 손실 방어가 우선. 낮은 PER·부채비율 종목이 낙폭 방어.',
        'caution': '모멘텀 전략은 하락장에서 손실 위험 높음',
    },
}

for reg in ['강세장', '횡보장', '약세장']:
    ranked = regime_winner[reg]
    top2   = [s for s, _ in ranked[:2]]
    worst  = ranked[-1][0]
    print(f"\n  {'📈' if reg=='강세장' else '➡️' if reg=='횡보장' else '📉'} {reg}  ({advice[reg]['desc']})")
    print(f"  ┌─ 추천 전략: {' > '.join(top2)}")
    print(f"  ├─ 이유: {advice[reg]['why']}")
    print(f"  ├─ 주의: {advice[reg]['caution']}")
    print(f"  └─ 비추천: {worst}  (이 국면에서 IR 최하위)")

# ── 현재 국면 판단 참고 ───────────────────────────────────────
print()
print('=' * 68)
print('【전략 선택 가이드】')
print('=' * 68)
print("""
  어떤 전략을 써야 할지 모르겠을 때:
  → 국면대응형을 기본으로, 보조로 해당 국면 추천 전략 병행

  단일 전략만 쓴다면:
  → 강세장: 모멘텀형 (CAGR·IR 모두 최고)
  → 횡보장: 밸런스형 (안정적인 초과수익)
  → 약세장: 저평가형 (낙폭 방어 + 반등 시 수익)

  국면 전환 시:
  → 강세→횡보: 모멘텀형 → 밸런스형으로 비중 이동
  → 횡보→약세: 밸런스형 → 저평가형으로 방어 강화
  → 약세→강세: 저평가형 → 모멘텀형으로 공격 전환
""")
