import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os

# ----------------------------------------------------
# 1. 을유1945 (Eulyoo1945) OTF 폰트 설정
# ----------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))

# 폰트 파일 후보 경로 설정
font_regular_path = os.path.join(current_dir, 'Eulyoo1945-Regular.otf')
font_semibold_path = os.path.join(current_dir, 'Eulyoo1945-SemiBold.otf')

# 폰트 적용 로직
selected_font_path = None
if os.path.exists(font_regular_path):
    selected_font_path = font_regular_path
elif os.path.exists(font_semibold_path):
    selected_font_path = font_semibold_path

if selected_font_path:
    font_prop = fm.FontProperties(fname=selected_font_path)
    fm.fontManager.addfont(selected_font_path)
    plt.rc('font', family=font_prop.get_name())
else:
    # 폰트 파일이 같은 폴더에 없을 경우 시스템 기본 한글 폰트 예비 적용
    plt.rc('font', family='Malgun Gothic')

plt.rcParams['axes.unicode_minus'] = False

# ----------------------------------------------------
# 2. 페이지 기본 설정
# ----------------------------------------------------
st.set_page_config(
    page_title="무역 분석 대시보드",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------------------------
# 3. 데이터 로드 및 전처리
# ----------------------------------------------------
@st.cache_data
def load_data():
    baci_path = os.path.join(current_dir, 'baci_85_sample.csv')
    country_path = os.path.join(current_dir, 'country_codes_sample.csv')

    baci_df = pd.read_csv(baci_path)
    country_df = pd.read_csv(country_path)

    # 컬럼명 앞뒤 공백 제거
    baci_df.columns = baci_df.columns.str.strip()
    country_df.columns = country_df.columns.str.strip()

    # 국가 코드 파일 매핑 (첫 번째 열: 코드, 두 번째 열: 국가명)
    code_col = country_df.columns[0]
    name_col = country_df.columns[1] if len(country_df.columns) > 1 else country_df.columns[0]
    code_to_name = dict(zip(country_df[code_col], country_df[name_col]))

    # 컬럼명 소문자 매핑 사전
    baci_cols_lower = {str(c).lower(): c for c in baci_df.columns}

    # 수출국 컬럼 식별
    exp_col = None
    for cand in ['i', 'exporter', 'country', '국가코드', '수출국']:
        if cand in baci_cols_lower:
            exp_col = baci_cols_lower[cand]
            break
    if exp_col is None:
        exp_col = baci_df.columns[1] if len(baci_df.columns) > 1 else baci_df.columns[0]

    baci_df['country_name'] = baci_df[exp_col].map(code_to_name).fillna(baci_df[exp_col].astype(str))

    # 연도 컬럼 식별
    year_col = None
    for cand in ['t', 'year', '연도']:
        if cand in baci_cols_lower:
            year_col = baci_cols_lower[cand]
            break
    baci_df['year'] = baci_df[year_col] if year_col else 2020

    # 무역액 컬럼 식별 (v, value, trade_value 등)
    val_col = None
    for cand in ['v', 'value', 'trade_value', '수출금액', '금액']:
        if cand in baci_cols_lower:
            val_col = baci_cols_lower[cand]
            break
    if val_col is None:
        val_col = baci_df.columns[-1]

    # 무역액 환산 (BACI 데이터 기준 1,000 USD 곱하기)
    baci_df['raw_value'] = pd.to_numeric(baci_df[val_col], errors='coerce')
    baci_df['trade_value_usd'] = baci_df['raw_value'] * 1000

    # 무역액 등급 구분 (대, 중, 소 3분위수)
    valid_vals = baci_df['trade_value_usd'].dropna()
    if not valid_vals.empty:
        q33 = valid_vals.quantile(0.33)
        q66 = valid_vals.quantile(0.66)
    else:
        q33, q66 = 0, 0

    def classify_grade(val):
        if pd.isna(val):
            return '결측'
        if val <= q33:
            return '소'
        elif val <= q66:
            return '중'
        else:
            return '대'

    baci_df['trade_grade'] = baci_df['trade_value_usd'].apply(classify_grade)

    return baci_df

try:
    df = load_data()
except Exception as e:
    st.error(f"파일을 읽어오는 중 오류가 발생했습니다: {e}")
    st.stop()

# ----------------------------------------------------
# 4. 사이드바 필터
# ----------------------------------------------------
st.sidebar.header("조회 필터")

# 국가 선택
all_countries = sorted([str(x) for x in df['country_name'].dropna().unique()])
default_countries = all_countries[:8] if len(all_countries) >= 8 else all_countries
selected_countries = st.sidebar.multiselect(
    "국가 선택",
    options=all_countries,
    default=default_countries
)

# 무역액 등급 선택 (대, 중, 소)
grade_options = ['대', '중', '소']
selected_grades = st.sidebar.multiselect(
    "무역액 등급 선택",
    options=grade_options,
    default=grade_options
)

# 데이터 필터링 적용
filtered_df = df[
    (df['country_name'].isin(selected_countries)) &
    (df['trade_grade'].isin(selected_grades))
]

# ----------------------------------------------------
# 5. 메인 화면 구성
# ----------------------------------------------------

# 1. 타이틀
st.title("무역 분석 대시보드")
st.markdown("---")

# 2. 총 0액 및 결측치 현황
st.subheader("데이터 결측치 및 0액 거래 현황")

zero_count = (df['trade_value_usd'] == 0).sum()
null_count = df['raw_value'].isna().sum()
total_raw_records = len(df)

kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.metric("총 0액 건수 (전체)", f"{zero_count:,} 건")
with kpi_col2:
    st.metric("baci_85_sample 결측치", f"{null_count:,} 건")
with kpi_col3:
    st.metric("원본 총 행수", f"{total_raw_records:,} 건")

st.markdown("---")

# 3. 총 거래 건수 및 총 수출액(달러)
st.subheader("필터링 기준 종합 실적")

filtered_total_count = len(filtered_df)
filtered_total_export = filtered_df['trade_value_usd'].sum()

stat_col1, stat_col2 = st.columns(2)
with stat_col1:
    st.metric("총 거래 건수", f"{filtered_total_count:,} 건")
with stat_col2:
    st.metric("총 수출액(달러)", f"${filtered_total_export:,.0f}")

st.markdown("---")

# 4. 국가*연도 수출액 히트맵(상위 8개국), 무역액 등급 분포
st.subheader("수출액 히트맵 및 무역액 등급 분포")

chart_col1, chart_col2 = st.columns([1.2, 0.8])

# 4-1. 상위 8개국 x 연도 히트맵
with chart_col1:
    st.write("**국가 x 연도 수출액 히트맵 (상위 8개국)**")
    top_8_countries = (
        filtered_df.groupby('country_name')['trade_value_usd']
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )

    if top_8_countries:
        heatmap_data = filtered_df[filtered_df['country_name'].isin(top_8_countries)].pivot_table(
            index='country_name',
            columns='year',
            values='trade_value_usd',
            aggfunc='sum',
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            heatmap_data / 1e6,
            cmap='Blues',
            annot=True,
            fmt=',.0f',
            ax=ax,
            cbar_kws={'label': '백만 달러 (M USD)'}
        )
        ax.set_ylabel("국가명", fontsize=11)
        ax.set_xlabel("연도", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("선택된 조건에 해당하는 데이터가 없습니다.")

# 4-2. 무역액 등급 분포
with chart_col2:
    st.write("**무역액 등급 분포 (건수)**")
    if not filtered_df.empty:
        grade_dist = filtered_df['trade_grade'].value_counts().reindex(['대', '중', '소']).fillna(0)

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        colors = ['#2563EB', '#60A5FA', '#93C5FD']
        bars = ax2.bar(grade_dist.index, grade_dist.values, color=colors)
        ax2.set_ylabel("거래 건수", fontsize=11)
        ax2.set_xlabel("무역액 등급", fontsize=11)
        ax2.grid(axis='y', linestyle='--', alpha=0.5)

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + (max(grade_dist.values) * 0.01),
                     f"{int(height):,}", ha='center', va='bottom')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("선택된 조건에 해당하는 데이터가 없습니다.")

st.markdown("---")

# 5. 상위 5개국 * 무역액 등급 교차표 (원본건수 / 정규화비율)
st.subheader("상위 5개국 x 무역액 등급 교차표")

top_5_countries = (
    filtered_df.groupby('country_name')['trade_value_usd']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .index.tolist()
)

if top_5_countries:
    sub_top5 = filtered_df[filtered_df['country_name'].isin(top_5_countries)]

    # 원본 건수 교차표
    raw_ct = pd.crosstab(
        sub_top5['country_name'],
        sub_top5['trade_grade'],
        margins=True,
        margins_name="합계"
    )
    cols_order = [c for c in ['대', '중', '소', '합계'] if c in raw_ct.columns]
    raw_ct = raw_ct[cols_order]

    # 행 기준 정규화 비율 교차표 (%)
    norm_ct = pd.crosstab(
        sub_top5['country_name'],
        sub_top5['trade_grade'],
        normalize='index'
    ) * 100
    norm_cols_order = [c for c in ['대', '중', '소'] if c in norm_ct.columns]
    norm_ct = norm_ct[norm_cols_order]

    tbl_col1, tbl_col2 = st.columns(2)
    with tbl_col1:
        st.write("**1) 원본 건수 교차표**")
        st.dataframe(raw_ct, use_container_width=True)

    with tbl_col2:
        st.write("**2) 정규화 비율 교차표 (행 기준 %)**")
        st.dataframe(norm_ct.style.format("{:.2f}%"), use_container_width=True)
else:
    st.info("교차표를 구성하기 위한 데이터가 부족합니다.")