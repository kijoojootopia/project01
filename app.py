import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# ---------------------------------------------------------
# 1. 페이지 설정 및 스타일 (나눔고딕 및 테마 색상 #FFFF00)
# ---------------------------------------------------------
st.set_page_config(page_title="무역 분석 대시보드", layout="wide")

# 나눔고딕 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 슬라이더 및 위젯 강조색(#FFFF00) 스타일 주입
st.markdown(
    """
    <style>
    /* 전체 폰트 나눔고딕 적용 */
    html, body, [class*="css"] {
        font-family: 'NanumGothic', sans-serif;
    }
    /* 슬라이더 트랙 및 강조색 #FFFF00 */
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
# 2. 데이터 불러오기 및 전처리
# ---------------------------------------------------------
@st.cache_data
def load_data():
    trade_df = pd.read_csv("baci_85_sample.csv")
    country_df = pd.read_csv("country_codes_sample.csv")
    
    # 국가 코드 매핑 (수출국 i 기준)
    code_col = 'country_code' if 'country_code' in country_df.columns else country_df.columns[0]
    name_col = 'country_name' if 'country_name' in country_df.columns else country_df.columns[1]
    
    code_to_name = dict(zip(country_df[code_col], country_df[name_col]))
    trade_df['country_name'] = trade_df['i'].map(code_to_name).fillna(trade_df['i'].astype(str))
    
    # 무역액 등급 (대, 중, 소) 범주화 (3분위수 기준)
    trade_df['무역액등급'] = pd.qcut(
        trade_df['v'], 
        q=[0, 0.33, 0.66, 1.0], 
        labels=['소', '중', '대']
    )
    return trade_df

trade_df = load_data()

# ---------------------------------------------------------
# 3. 사이드바 필터 구성
# ---------------------------------------------------------
st.sidebar.title("필터 옵션")

all_countries = sorted(trade_df['country_name'].unique().tolist())
selected_countries = st.sidebar.multiselect("국가 선택", options=all_countries, default=all_countries[:5])

tier_options = ['소', '중', '대']
selected_tiers = st.sidebar.multiselect("무역액 등급 선택", options=tier_options, default=tier_options)

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

# 2. 결측치 및 총 거래 건수
missing_count = trade_df.isnull().sum().sum()
total_raw_count = len(trade_df)

col1, col2 = st.columns(2)
col1.metric(label="전체 결측치 수", value=f"{missing_count:,} 개")
col2.metric(label="전체 거래 건수", value=f"{total_raw_count:,} 건")

st.divider()

# 3. 필터링된 총 거래 건수 및 총 수출액(달러)
filtered_trade_count = len(filtered_df)
filtered_total_val = filtered_df['v'].sum()

col3, col4 = st.columns(2)
col3.metric(label="총 거래 건수", value=f"{filtered_trade_count:,} 건")
col4.metric(label="총 수출액(달러)", value=f"${filtered_total_val:,.2f}")

st.divider()

# 4. 국가*연도 수출액 히트맵(상위 8개국) & 무역액 등급 분포
chart_col1, chart_col2 = st.columns(2)

# 노란색 단색 계열 컬러맵 생성
yellow_cmap = sns.light_palette("#FFFF00", as_cmap=True)

with chart_col1:
    st.subheader("상위 8개국 국가*연도 수출액 히트맵")
    top_8_countries = trade_df.groupby('country_name')['v'].sum().nlargest(8).index
    heatmap_data = trade_df[trade_df['country_name'].isin(top_8_countries)].pivot_table(
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

# 5. 상위 5개국 * 무역액 등급 교차표 (원본건수, 정규화비율)
st.subheader("상위 5개국 * 무역액 등급 교차표")

top_5_countries = trade_df.groupby('country_name')['v'].sum().nlargest(5).index
top_5_df = trade_df[trade_df['country_name'].isin(top_5_countries)]

# 교차표 생성
raw_ct = pd.crosstab(top_5_df['country_name'], top_5_df['무역액등급'])
norm_ct = pd.crosstab(top_5_df['country_name'], top_5_df['무역액등급'], normalize='index').round(4)

table_col1, table_col2 = st.columns(2)

with table_col1:
    st.write("**원본건수**")
    st.dataframe(raw_ct, use_container_width=True)

with table_col2:
    st.write("**정규화비율 (행 기준)**")
    st.dataframe(norm_ct, use_container_width=True)