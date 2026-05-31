"""
섹터 분류 전체 진단
- KRX Industry 원문과 매핑 결과를 모두 출력
- 의심 종목 (이름과 섹터가 안 맞는 경우) 자동 탐지
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import FinanceDataReader as fdr
import pandas as pd

# ── KRX 데이터 로드 ──
krx = fdr.StockListing('KRX')
krx['Code'] = krx['Code'].astype(str).str.zfill(6)

kd = fdr.StockListing('KRX-DESC')
kd['Code'] = kd['Code'].astype(str).str.zfill(6)

if 'Sector' in krx.columns:
    krx = krx.drop(columns=['Sector'])
cols = ['Code', 'Sector', 'Industry'] if 'Industry' in kd.columns else ['Code', 'Sector']
krx = krx.merge(kd[cols], on='Code', how='left')

# ── 섹터 매핑 함수 (app.py와 동일) ──
def map_sector(raw: str) -> str:
    r = raw if raw else ''
    if '반도체' in r or '전자부품' in r or '집적회로' in r or '디스플레이' in r:
        return '반도체'
    if ('소프트웨어' in r or '컴퓨터' in r or '정보통신' in r or '통신업' in r
            or '인터넷' in r or '자료처리' in r or '포털' in r
            or '통신 및 방송 장비' in r or '통신장비' in r or '전기통신' in r):
        return 'IT'
    if '자동차' in r or '운송장비' in r or '모터' in r:
        return '자동차'
    if '금융' in r or '은행' in r or '보험' in r or '지주' in r or '증권' in r or '투자' in r:
        return '금융'
    if '의약' in r or '의료' in r or '바이오' in r or '제약' in r:
        return '바이오'
    if ('게임' in r or '엔터' in r or '방송프로그램' in r or '비디오물' in r
            or '미디어' in r or '콘텐츠' in r or '광고' in r
            or '영화' in r or '음악' in r or '공연' in r):
        return '엔터/미디어'
    if ('음식' in r or '식품' in r or '음료' in r or '유통' in r or '의류' in r
            or '섬유' in r or '화장품' in r or '생활용품' in r
            or '도소매' in r or '패션' in r):
        return '소비재'
    if ('조선' in r or '건설' in r or '철강' in r or '기계' in r or '화학' in r
            or '에너지' in r or '전력' in r or '가스' in r or '금속' in r
            or '운송' in r or '물류' in r or '항공' in r):
        return '산업재'
    return '기본'

krx['raw_sector'] = (
    krx['Sector'].fillna('').astype(str) + ' ' +
    krx['Industry'].fillna('').astype(str)
).str.strip()

krx['mapped'] = krx['raw_sector'].apply(map_sector)

# ── 특정 종목 확인 ──
print("=" * 70)
print("【특정 종목 섹터 확인】")
print("=" * 70)
targets = ['대덕', '미래반도체', '삼성전자', 'SK하이닉스', '현대차', 'NAVER',
           '카카오', '셀트리온', 'LG에너지솔루션', '포스코', '한화']
for keyword in targets:
    rows = krx[krx['Name'].str.contains(keyword, na=False)]
    for _, r in rows.iterrows():
        print(f"  [{r['Name']} {r['Code']}]")
        print(f"    Industry  : {r.get('Industry','')}")
        print(f"    매핑 결과 : {r['mapped']}")
    if rows.empty:
        print(f"  [{keyword}] — 없음")

# ── 섹터별 종목 수 ──
print("\n" + "=" * 70)
print("【섹터별 매핑 종목 수】")
print("=" * 70)
print(krx['mapped'].value_counts().to_string())

# ── '기본' 섹터 상위 Industry ──
print("\n" + "=" * 70)
print("【'기본' 섹터 Industry 분포 Top 40 (미매핑 원인 분석)】")
print("=" * 70)
basic = krx[krx['mapped'] == '기본']
print(basic['Industry'].value_counts().head(40).to_string())

# ── 이름과 섹터 불일치 의심 종목 ──
print("\n" + "=" * 70)
print("【이름-섹터 불일치 의심 종목】")
print("=" * 70)

# 이름에 키워드가 있지만 해당 섹터로 매핑 안 된 경우
checks = [
    ('반도체', '반도체',   ['반도체']),
    ('바이오', '바이오',   ['바이오']),
    ('금융',   '금융',     ['금융']),
    ('자동차', '자동차',   ['자동차']),
    ('건설',   '산업재',   ['산업재']),
]
for keyword, label, expected_sectors in checks:
    suspects = krx[
        krx['Name'].str.contains(keyword, na=False) &
        ~krx['mapped'].isin(expected_sectors)
    ]
    if not suspects.empty:
        print(f"\n  이름에 '{keyword}' 포함이지만 {label} 섹터 아닌 종목:")
        for _, r in suspects.iterrows():
            print(f"    {r['Name']} ({r['Code']}) → {r['mapped']} | Industry: {r.get('Industry','')}")

# ── 금융으로 분류된 종목 중 이름이 이상한 경우 ──
print("\n" + "=" * 70)
print("【금융 분류 종목 중 비금융 의심 (Industry에 '투자' 포함)】")
print("=" * 70)
fin_suspicious = krx[
    (krx['mapped'] == '금융') &
    (krx['Industry'].fillna('').str.contains('투자|지주', na=False))
]
for _, r in fin_suspicious.head(30).iterrows():
    print(f"  {r['Name']} ({r['Code']}) | {r.get('Industry','')}")

print("\n✅ 진단 완료")
