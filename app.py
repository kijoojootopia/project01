import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# 1. 페이지 설정 및 노란색 테마 강제 주입
# ---------------------------------------------------------
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# 나눔고딕 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 기본 primaryColor(빨간색 등)를 #FFFF00 계열로 덮어쓰기
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'NanumGothic', sans-serif;
    }
    /* 사이드바 multiselect 태그 및 포커스 색상 */
    span[data-baseweb="tag"] {
        background-color: #FFFF00 !important;
        color: #000000 !important;
    }
    /* 체크박스, 라디오, 슬라이더 기본 테마 색상 덮어쓰기 */
    :root {
        --primary-color: #FFFF00;
    }
    div[data-baseweb="select"] * {
        border-color: #FFFF00 !important;
    }
    div[data-baseweb="slider"] div {
        color: #FFFF00 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# 2. 데이터 불러오기 및 전처리 (국가 코드 매핑 보정)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    trade_df = pd.read_csv("baci_85_sample.csv")
    country_df = pd.read_csv("country_codes_sample.csv")
    
    # 1) country_df의 컬럼명 공백 제거 및 소문자 통일
    country_df.columns = [c.strip().lower() for c in country_df.columns]
    
    # 2) 코드 컬럼 찾기 (code, id, i, c 중 포함된 컬럼 자동 탐색)
    code_candidates = [c for c in country_df.columns if any(k in c for k in ['code', 'id', 'i', 'num'])]
    name_candidates = [c for c in country_df.columns if any(k in c for k in ['name', 'country', 'desc'])]
    
    code_col = code_candidates[0] if code_candidates else country_df.columns[0]
    name_col = name_candidates[0] if name_candidates else country_df.columns[1]
    
    # 3) 데이터 타입 통일 (정수형 변환 후 문자열 변환하여 공백/소수점 오차 방지)
    country_df['clean_code'] = pd.to_numeric(country_df[code_col], errors='coerce').fillna(-1).astype(int).astype(str)
    trade_df['clean_i'] = pd.to_numeric(trade_df['i'], errors='coerce').fillna(-1).astype(int).astype(str)
    
    # 4) 매핑 딕셔너리 생성
    code_to_name = dict(zip(country_df['clean_code'], country_df[name_col]))
    
    # 5) 국가명 치환 (매핑 실패 시에만 원본 코드 유지)
    trade_df['country_name'] = trade_df['clean_i'].map(code_to_name).fillna(trade_df['i'].astype(str))
    
    # 무역액 등급 (대, 중, 소)
    trade_df['무역액등급'] = pd.qcut(
        trade_df['v'], 
        q=[0, 0.33, 0.66, 1.0], 
        labels=['소', '중', '대']
    )
    return trade_df

trade_df = load_data()

# ---------------------------------------------------------
# 3. 사이드바 필터
# ---------------------------------------------------------
st.sidebar.title("필터 옵션")

all_countries = sorted(trade_df['country_name'].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "국가 선택", 
    options=all_countries, 
    default=all_countries[:min(5, len(all_countries))]
)

tier_options = ['소', '중', '대']
selected_tiers = st.sidebar.multiselect("무역액 등급 선택", options=tier_options, default=tier_options)

filtered_df = trade_df[
    (trade_df['country_name'].isin(selected_countries)) &
    (trade_df['무역액등급'].isin(selected_tiers))
]

# ---------------------------------------------------------
# 4. 메인 화면
# ---------------------------------------------------------
st.title("무역 분석 대시보드")

col1, col2 = st.columns(2)
col1.metric("전체 결측치 수", f"{trade_df.isnull().sum().sum():,} 개")
col2.metric("전체 거래 건수", f"{len(trade_df):,} 건")

st.divider()

col3, col4 = st.columns(2)
col3.metric("총 거래 건수", f"{len(filtered_df):,} 건")
col4.metric("총 수출액(달러)", f"${filtered_df['v'].sum():,.2f}")

st.divider()

chart_col1, chart_col2 = st.columns(2)
yellow_cmap = sns.light_palette("#FFFF00", as_cmap=True)

with chart_col1:
    st.subheader("상위 8개국 국가*연도 수출액 히트맵")
    top_8 = trade_df.groupby('country_name')['v'].sum().nlargest(8).index
    heatmap_data = trade_df[trade_df['country_name'].isin(top_8)].pivot_table(
        index='country_name', 
        columns='t', 
        values='v', 
        aggfunc='sum', 
        fill_value=0
    )
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.heatmap(heatmap_data, cmap=yellow_cmap, ax=ax1, cbar_kws={'label': '수출액'})
    ax1.set_xlabel("연도")
    ax1.set_ylabel("국가")
    st.pyplot(fig1)

with chart_col2:
    st.subheader("무역액 등급 분포")
    tier_counts = filtered_df['무역액등급'].value_counts().reindex(['소', '중', '대']).fillna(0)
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(tier_counts.index, tier_counts.values, color="#FFFF00", edgecolor="#888800")
    ax2.set_xlabel("무역액 등급")
    ax2.set_ylabel("건수")
    st.pyplot(fig2)

st.divider()

st.subheader("상위 5개국 * 무역액 등급 교차표")
top_5 = trade_df.groupby('country_name')['v'].sum().nlargest(5).index
top_5_df = trade_df[trade_df['country_name'].isin(top_5)]

raw_ct = pd.crosstab(top_5_df['country_name'], top_5_df['무역액등급'])
norm_ct = pd.crosstab(top_5_df['country_name'], top_5_df['무역액등급'], normalize='index').round(4)

table_col1, table_col2 = st.columns(2)
with table_col1:
    st.write("**원본건수**")
    st.dataframe(raw_ct, use_container_width=True)

with table_col2:
    st.write("**정규화비율 (행 기준)**")
    st.dataframe(norm_ct, use_container_width=True)