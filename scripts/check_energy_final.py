import sys; sys.stdout.reconfigure(encoding='utf-8')
import FinanceDataReader as fdr, pandas as pd

krx = fdr.StockListing('KRX'); krx['Code'] = krx['Code'].astype(str).str.zfill(6)
kd = fdr.StockListing('KRX-DESC'); kd['Code'] = kd['Code'].astype(str).str.zfill(6)
if 'Sector' in krx.columns: krx = krx.drop(columns=['Sector'])
krx = krx.merge(kd[['Code','Sector','Industry']], on='Code', how='left')

def map_sector(raw, name=''):
    r = raw if raw else ''; n = name if name else ''
    if '반도체' in r or '전자부품' in r or '집적회로' in r or '디스플레이' in r or '반도체' in n: return '반도체'
    if '소프트웨어' in r or '컴퓨터' in r or '정보통신' in r or '통신업' in r or '인터넷' in r or '자료처리' in r or '포털' in r or '통신 및 방송 장비' in r or '통신장비' in r or '전기통신' in r or '정보 서비스' in r or '측정, 시험' in r: return 'IT'
    if '자동차' in r or '운송장비' in r or '모터' in r: return '자동차'
    if '은행' in r or '보험' in r or '증권' in r or '금융업' in r or '금융 지원 서비스' in r or '신탁업' in r or '집합투자' in r or '여신' in r or '대부업' in r: return '금융'
    if '의약' in r or '의료' in r or '바이오' in r or '제약' in r or '연구개발' in r or '기초 의약물질' in r or '의약품 제조' in r: return '바이오'
    if '석유 정제' in r or '원유' in r or '정유' in r or '연료용 가스' in r or '가스 배관' in r or '전기업' in r or '수력 발전' in r or '원자력' in r or ('가스' in n and '이차전지' not in r and '배터리' not in r) or '정유' in n: return '에너지/유틸리티'
    if '게임' in r or '엔터' in r or '방송프로그램' in r or '비디오물' in r or '미디어' in r or '콘텐츠' in r or '광고' in r or '영화' in r or '음악' in r or '공연' in r or '텔레비전 방송' in r or '창작 및 예술' in r or '오디오물' in r: return '엔터/미디어'
    if '음식' in r or '식품' in r or '음료' in r or '유통' in r or '의류' in r or '의복' in r or '봉제' in r or '섬유' in r or '화장품' in r or '생활용품' in r or '도소매' in r or '패션' in r or '소매업' in r or '종합 소매' in r or '가정용 기기' in r or '영상 및 음향기기' in r or '가전' in r or '가구' in r or '출판' in r: return '소비재'
    if '조선' in r or '선박' in r or '건설' in r or '철강' in r or '기계' in r or '화학' in r or '에너지' in r or '전력' in r or '가스' in r or '금속' in r or '운송' in r or '물류' in r or '항공' in r or '전동기' in r or '발전기' in r or '전기 변환' in r or '절연선' in r or '케이블' in r or '이차전지' in r or '일차전지' in r or '플라스틱' in r or '고무' in r or '비료' in r or '농약' in r or '시멘트' in r or '유리' in r or '세라믹' in r or '펄프' in r or '종이' in r or '엔지니어링' in r or '전기장비' in r or '광학기기' in r: return '산업재'
    return '기본'

krx['raw'] = (krx['Sector'].fillna('').astype(str)+' '+krx['Industry'].fillna('').astype(str)).str.strip()
krx['mapped'] = krx.apply(lambda row: map_sector(row['raw'], str(row.get('Name','') or '')), axis=1)

print("=== 에너지 관련 종목 분류 확인 ===")
for kw in ['에스오일','S-Oil','SK이노베이션','한국가스공사','SK가스','E1','삼천리','한국전력','서울가스','LG에너지']:
    rows = krx[krx['Name'].str.contains(kw, na=False)].head(1)
    for _, r in rows.iterrows():
        print(f"  [{r['Name']}] -> {r['mapped']}  ({r.get('Industry','')})")

print()
print("=== 에너지/유틸리티 전체 목록 ===")
energy = krx[krx['mapped'] == '에너지/유틸리티']
for _, r in energy.iterrows():
    print(f"  {r['Name']} ({r['Code']}) | {r.get('Industry','')}")

print()
print("=== 섹터 분포 ===")
print(krx['mapped'].value_counts().to_string())
