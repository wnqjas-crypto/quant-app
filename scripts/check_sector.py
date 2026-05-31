import FinanceDataReader as fdr
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

krx = fdr.StockListing('KRX')
krx['Code'] = krx['Code'].astype(str).str.zfill(6)

try:
    kd = fdr.StockListing('KRX-DESC')
    kd['Code'] = kd['Code'].astype(str).str.zfill(6)
    print('KRX-DESC 컬럼:', list(kd.columns))
    cols = ['Code', 'Sector', 'Industry'] if 'Industry' in kd.columns else ['Code', 'Sector']
    if 'Sector' in krx.columns:
        krx = krx.drop(columns=['Sector'])
    krx = krx.merge(kd[cols], on='Code', how='left')
except Exception as e:
    print('DESC 실패:', e)

print()
print('=== 특정 종목 확인 ===')
for code, name in [('005930', '삼성전자'), ('037460', '삼지전자'), ('000660', 'SK하이닉스'), ('035420', 'NAVER')]:
    r = krx[krx['Code'] == code]
    if not r.empty:
        row = r.iloc[0]
        print(f'[{name}] Sector={row.get("Sector")} | Industry={row.get("Industry")}')
    else:
        print(f'[{name}] 없음')

print()
print('=== Sector 고유값 ===')
if 'Sector' in krx.columns:
    print(krx['Sector'].value_counts().head(20).to_string())

print()
print('=== Industry 고유값 top30 ===')
if 'Industry' in krx.columns:
    print(krx['Industry'].value_counts().head(30).to_string())
