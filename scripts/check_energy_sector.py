import sys
sys.stdout.reconfigure(encoding='utf-8')
import FinanceDataReader as fdr
import pandas as pd

# ── KRX-DESC Industry 확인 ───────────────────────────────────
krx = fdr.StockListing('KRX')
krx['Code'] = krx['Code'].astype(str).str.zfill(6)
kd  = fdr.StockListing('KRX-DESC')
kd['Code'] = kd['Code'].astype(str).str.zfill(6)
if 'Sector' in krx.columns:
    krx = krx.drop(columns=['Sector'])
krx = krx.merge(kd[['Code','Sector','Industry']], on='Code', how='left')

print("=" * 65)
print("【에너지·자원 관련 종목 Industry 확인】")
print("=" * 65)
targets = ['에스오일', 'S-Oil', 'SK이노베이션', 'GS', '현대오일뱅크',
           'Inveni', '인베니', '한국가스공사', '도시가스', '한국전력',
           'SK가스', 'E1', '부산가스', '서울가스', '삼천리',
           'HD현대오일뱅크', '오일뱅크', '한화토탈', '롯데케미칼',
           '포스코인터']
for kw in targets:
    rows = krx[krx['Name'].str.contains(kw, na=False)]
    for _, r in rows.head(2).iterrows():
        print(f"  [{r['Name']} {r['Code']}]  Industry: {r.get('Industry','없음')}")

print()
print("=" * 65)
print("【KOSPI 업종(Sector) 컬럼 확인 — FDR 지원 여부】")
print("=" * 65)
try:
    kospi = fdr.StockListing('KOSPI')
    print(f"  컬럼: {list(kospi.columns)}")
    # 업종 관련 컬럼 찾기
    for col in kospi.columns:
        if any(kw in col for kw in ['Sector','Industry','업종','sector']):
            print(f"  ★ 업종 관련 컬럼: {col}")
            print(kospi[col].value_counts().head(10).to_string())
except Exception as e:
    print(f"  오류: {e}")

print()
print("=" * 65)
print("【기본 섹터 중 에너지·자원 의심 Industry 목록】")
print("=" * 65)
energy_keywords = ['원유', '석유', '정유', '가스', '광업', '광산',
                   '채굴', '유틸리티', '연료', 'LNG', 'LPG', '발전']
for kw in energy_keywords:
    hits = krx[krx['Industry'].fillna('').str.contains(kw, na=False)]
    if not hits.empty:
        print(f"\n  '{kw}' 포함 Industry ({len(hits)}개):")
        for ind, cnt in hits['Industry'].value_counts().head(5).items():
            print(f"    {ind}  ({cnt}개)")
