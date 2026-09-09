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

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'NanumGothic', sans-serif;
    }
    /* 사이드바 multiselect 선택 태그 노란색 적용 */
    span[data-baseweb="tag"] {
        background-color: #FFFF00 !important;
        color: #000000 !important;
    }
    /* 슬라이더 색상 #FFFF00 */
    div[data-baseweb="slider"] div {
        color: #FFFF00 !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# 2. 데이터 불러오기 및 결합 (j 기준 merge)
# ---------------------------------------------------------
@st.cache_data
def load_data():
    trade_df = pd.read_csv("baci_85_sample.csv")
    country_df = pd.read_csv("country_codes_sample.csv")
    
    # 공백 제거
    trade_df.columns = trade_df.columns.str.strip()
    country_df.columns = country_df.columns.str.strip()
    
    # j 컬럼 기준으로 국가명 병합
    merged_df = pd.merge(trade_df, country_df[['j', 'country_name']], on='j', how='left')
    merged_df['country_name'] = merged_df['country_name'].fillna(merged_df['j'].astype(str))
    
    # 무역액 등급 (대, 중, 소) 범주화
    merged_df['무역액등급'] = pd.qcut(
        merged_df['v'], 
        q=[0, 0.33, 0.66, 1.0], 
        labels=['소', '중', '대']
    )
    return merged_df

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
selected_tiers = st.sidebar.multiselect(
    "무역액 등급 선택", 
    options=tier_options, 
    default=tier_options
)

# 필터 적용
filtered_df = trade_df[
    (trade_df['country_name'].isin(selected_countries)) &
    (trade_df['무역액등급'].isin(selected_tiers))
]

# ---------------------------------------------------------
# 4. 메인 화면
# ---------------------------------------------------------
# 1. 타이틀
st.title("무역 분석 대시보드")

# 2. 결측치 및 전체 거래 건수
col1, col2 = st.columns(2)
col1.metric("전체 결측치 수", f"{trade_df.isnull().sum().sum():,} 개")
col2.metric("전체 거래 건수", f"{len(trade_df):,} 건")

st.divider()

# 3. 필터 적용 건수 및 총 수출액
col3, col4 = st.columns(2)
col3.metric("총 거래 건수", f"{len(filtered_df):,} 건")
col4.metric("총 수출액(달러)", f"${filtered_df['v'].sum():,.2f}")

st.divider()

# 4. 국가*연도 히트맵 & 등급 분포
chart_col1, chart_col2 = st.columns(2)


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

# 5. 상위 5개국 * 무역액 등급 교차표
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