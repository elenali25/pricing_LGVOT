import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from pathlib import Path

# 确保导入 get_clean_region_name
from utils import load_data, REGION_LEVELS, NEW_ISSUE_MONTHS, get_clean_region_name 

# --- 配置参数 ---
POLYNOMIAL_ORDER = 3 # 曲线拟合使用三阶多项式

# --- B 区域子分组加载 (保持不变) ---
CLASSIFICATION_FILE = 'b_region_classification_recent.csv'
B_CLASSIFICATION_MAP = {}
try:
    script_dir = Path(__file__).resolve().parent
    df_b_groups = pd.read_csv(script_dir / CLASSIFICATION_FILE, index_col='Province', encoding='utf-8')
    B_CLASSIFICATION_MAP = df_b_groups['B_SubGroup'].to_dict()
    print(f"✅ 成功加载 B 区子分组结果，包含 {len(B_CLASSIFICATION_MAP)} 个省份。")
except FileNotFoundError:
    print(f"⚠️  未找到 B 区子分组文件 ({CLASSIFICATION_FILE})，所有 B 区省份将作为统一的 'B' 区域处理。")
except Exception as e:
    print(f"❌ 加载 B 区子分组失败: {e}")

# --- B 区域子分组应用函数 (保持不变) ---

def apply_b_subgroups(df, classification_map):
    """
    根据 B_CLASSIFICATION_MAP，将 df 中区域等级为 'B' 的省份细分为 'B1', 'B2', 'B3'。
    """
    if not classification_map or df.empty:
        return df

    # 1. 确保区域名称在 df 中被清理
    df['区域_Clean'] = df['区域'].apply(get_clean_region_name)

    # 2. 创建一个新的区域等级列
    df['区域等级_New'] = df['区域等级'].copy()

    # 3. 应用子分组
    for clean_name, new_group in classification_map.items():
        # 找到原始区域等级为 'B' 且清洗后名称匹配的行
        df.loc[
            (df['区域等级'] == 'B') & (df['区域_Clean'] == clean_name),
            '区域等级_New'
        ] = new_group
        
    df['区域等级'] = df['区域等级_New']
    # 注意：此处暂时保留 '区域_Clean'，稍后在 main_app 中用于查找
    df.drop(columns=['区域等级_New'], inplace=True) 
    return df

# --- 模型训练和曲线拟合 (保持不变) ---
@st.cache_data(show_spinner="正在训练双基准曲线模型...")
def train_yield_curve_model(df_subset, order):
    # ... (函数体保持不变) ...
    if df_subset.empty or len(df_subset) < order + 1:
        return None, None
        
    X = df_subset['剩余年限'].values.reshape(-1, 1)
    Y = df_subset['收盘收益率'].values

    poly_features = PolynomialFeatures(degree=order, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, Y)
    
    return model, poly_features

# --- 阶段二：增强版区域信用溢价模型 (保持不变) ---
@st.cache_data(show_spinner="正在训练增强版区域利差模型...")
def train_spread_regression_model(df_latest, _base_yield_func):
    # ... (函数体保持不变) ...
    # 核心：确保 OLS 模型中的 '区域等级' 已被 B1/B2/B3/C 替换
    
    df_model = df_latest.copy()
    df_model['基准收益率'] = df_model.apply(
        lambda row: _base_yield_func(row['剩余年限'], row['是否交税']), axis=1
    )
    df_model['利差'] = df_model['收盘收益率'] - df_model['基准收益率'] 

    df_model['C_Spread'] = df_model['票面'] - df_model['基准收益率']
    
    df_model = df_model.dropna(subset=['利差', '基准收益率', 'C_Spread']).copy()
    
    min_required_points = 11
    if df_model.empty or len(df_model) < min_required_points: 
        st.error(f"❗ 样本量过少：计算利差后，用于回归的有效数据点少于 {min_required_points} 个 (目前有 {len(df_model)} 个)，无法进行 OLS 回归。")
        return None
    
    df_model = pd.get_dummies(df_model, columns=['区域等级'], prefix='区域等级', drop_first=True)
    
    regression_cols = []
    
    if '区域等级_B1' in df_model.columns: regression_cols.append('区域等级_B1')
    if '区域等级_B2' in df_model.columns: regression_cols.append('区域等级_B2')
    if '区域等级_B3' in df_model.columns: regression_cols.append('区域等级_B3')
    if '区域等级_C' in df_model.columns: regression_cols.append('区域等级_C')
    
    regression_cols.append('余额_ln') 
    regression_cols.append('Is_Special') 
    regression_cols.append('Is_New') 
    regression_cols.append('Is_Taxable') 
    regression_cols.append('C_Spread') 

    df_model['C_Spread_Taxable_Int'] = df_model['C_Spread'] * df_model['Is_Taxable']
    regression_cols.append('C_Spread_Taxable_Int')

    X = df_model.loc[:, [col for col in regression_cols if col in df_model.columns]] 
    Y = df_model['利差'].astype(float) 

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    X = sm.add_constant(X, has_constant='add') 

    try:
        spread_model = sm.OLS(Y, X).fit()
        return spread_model
    except Exception as e:
        st.error(f"❗ OLS 回归运行时发生错误，请检查数据和样本量：{e}")
        return None

# --- 辅助函数 (保持不变) ---
def get_fitted_curve_data(df, model, poly_features, label):
    # ... (函数体保持不变) ...
    if model is None or df.empty:
        return pd.DataFrame()

    min_term = df['剩余年限'].min()
    max_term = df['剩余年限'].max()
    term_range = np.linspace(min_term, max_term, 100).reshape(-1, 1)

    term_range_poly = poly_features.transform(term_range)
    predicted_yield = model.predict(term_range_poly)

    return pd.DataFrame({
        '剩余年限': term_range.flatten(),
        '拟合收益率': predicted_yield,
        '是否交税': label
    })



# --- Streamlit 主应用函数 (核心修改区域) ---

def main_app():
    
    st.set_page_config(page_title="地方债双曲线利差定价模型", layout="wide")
    st.header("⚖️ 地方债双曲线利差定价模型")
    st.sidebar.title("数据来源")
    data_source = st.sidebar.radio("选择数据来源", ["仓库底表", "手动上传"], index=0)
    repo_file_name = st.sidebar.text_input("仓库底表文件名", value="样本数据.xlsx")
    def _refresh_repo_data():
        st.cache_data.clear()
        st.experimental_rerun()
    if data_source == "仓库底表":
        st.sidebar.button("刷新仓库数据", on_click=_refresh_repo_data)

    uploaded_file = None
    if data_source == "手动上传":
        uploaded_file = st.sidebar.file_uploader(
            "请上传地方债数据文件 (.xlsx 或 .csv)", 
            type=["xlsx", "csv"]
        )

    if data_source == "手动上传" and uploaded_file is None:
        st.info("👈 请在左侧边栏上传您的数据文件开始模型分析。")
        return

    if data_source == "手动上传":
        df_full, latest_date = load_data(uploaded_file)
        loaded_file_label = uploaded_file.name if uploaded_file else ""
    else:
        script_dir = Path(__file__).resolve().parent
        default_path = script_dir / repo_file_name
        df_full, latest_date = load_data(default_path)
        loaded_file_label = str(default_path.name)

    if df_full is None or df_full.empty:
        st.warning("数据文件加载成功，但筛选后没有足够有效数据点。")
        return

    st.success(f"已加载数据文件：{loaded_file_label}")
        
    # 提取最新的交易日数据进行模型训练
    df_latest_for_model = df_full[df_full['当前日期'] == latest_date].copy()
    
    # **【核心步骤 1】应用 B 区子分组**
    df_latest_for_model = apply_b_subgroups(df_latest_for_model, B_CLASSIFICATION_MAP)
    
    st.info(f"模型训练基于最新交易日数据：**{latest_date.strftime('%Y-%m-%d')}**")
    
    # --- 阶段一/二 模型训练 ---
    
    taxable_df = df_latest_for_model[df_latest_for_model['是否交税'] == '是']
    taxfree_df = df_latest_for_model[df_latest_for_model['是否交税'] == '否']
    
    taxable_model, taxable_poly = train_yield_curve_model(taxable_df, POLYNOMIAL_ORDER)
    taxfree_model, taxfree_poly = train_yield_curve_model(taxfree_df, POLYNOMIAL_ORDER)
    
    # 辅助函数：根据期限和是否交税获取基准收益率 (保持不变)
    def get_base_yield(term, tax_status):
        if tax_status == '是' and taxable_model:
            X_poly = taxable_poly.transform(np.array([[term]]))
            return taxable_model.predict(X_poly)[0]
        elif tax_status == '否' and taxfree_model:
            X_poly = taxfree_poly.transform(np.array([[term]]))
            return taxfree_model.predict(X_poly)[0]
        return np.nan
        
    st.subheader("基准曲线拟合")
    
    # =================================================================
    # 【修复：新增曲线拟合图表可视化】
    # =================================================================
    curve_data_tax = get_fitted_curve_data(taxable_df, taxable_model, taxable_poly, '应税曲线')
    curve_data_free = get_fitted_curve_data(taxfree_df, taxfree_model, taxfree_poly, '免税曲线')
    
    if not curve_data_tax.empty and not curve_data_free.empty:
        full_curve_data = pd.concat([curve_data_tax, curve_data_free])
        
        # 准备散点数据，注意使用包含 B 细分后的 df_latest_for_model
        df_latest_for_model_plot = df_latest_for_model.copy() 
        df_latest_for_model_plot['类型'] = df_latest_for_model_plot['是否交税'].apply(lambda x: '应税成交点' if x == '是' else '免税成交点') 
        
        # 计算坐标轴范围
        min_yield = df_latest_for_model_plot['收盘收益率'].min()
        y_min_start = max(0.0, min_yield - 0.1)
        y_max_end = df_latest_for_model_plot['收盘收益率'].max() * 1.05
        x_scale = alt.Scale(domain=[0.0, 30.0])
        y_scale = alt.Scale(domain=[y_min_start, y_max_end], reverse=False) 

        # 绘制散点图
        scatter = alt.Chart(df_latest_for_model_plot).mark_point(size=50).encode(
            x=alt.X('剩余年限', title='剩余年限 (年)', scale=x_scale),
            y=alt.Y('收盘收益率', title='收盘收益率 (%)', scale=y_scale),
            color=alt.Color('类型', scale=alt.Scale(domain=['应税成交点', '免税成交点'], range=['red', 'blue'])),
            tooltip=['债券名称', '剩余年限', alt.Tooltip('收盘收益率', format='.4f'), '区域等级']
        )
        
        # 绘制拟合曲线
        line = alt.Chart(full_curve_data).mark_line(strokeWidth=3).encode(
            x=alt.X('剩余年限', scale=x_scale),
            y=alt.Y('拟合收益率', scale=y_scale),
            color=alt.Color('是否交税', scale=alt.Scale(domain=['应税曲线', '免税曲线'], range=['red', 'blue'])),
            tooltip=['是否交税', '剩余年限', alt.Tooltip('拟合收益率', format='.4f')]
        )
        
        # 组合图表并显示
        st.altair_chart((scatter + line).interactive(), use_container_width=True)
        st.caption("图中展示了应税和免税两组数据的成交点及其拟合的**三阶多项式曲线**。图表支持鼠标缩放和平移。")
    else:
        st.error("数据点不足，无法拟合双基准曲线。请检查数据中是否包含足够的应税和免税债券的最新成交数据。")
        return # 如果曲线拟合失败，后续的 OLS 也会失败
    # =================================================================
    
    # OLS 模型训练
    spread_model = train_spread_regression_model(df_latest_for_model.copy(), get_base_yield)
    
    if spread_model is None:
        return 

    st.subheader("目标券合理收益率")

    col_r1_1, col_r1_2, col_r1_3, _ = st.columns(4)
    col_r2_1, col_r2_2, col_r2_3, col_r2_4 = st.columns(4)

    all_unique_provinces = sorted(df_latest_for_model['区域_Clean'].unique().tolist())
    
    min_term = df_latest_for_model['剩余年限'].min()
    max_term = df_latest_for_model['剩余年限'].max()
    target_term = col_r1_1.number_input("剩余年限 (年)", min_value=min_term, max_value=max_term, value=5.0, step=0.1, format='%.2f')
    target_tax = col_r1_2.selectbox("是否交税", options=['是', '否'])
    target_province_clean = col_r1_3.selectbox("目标省份", options=all_unique_provinces)

    target_special = col_r2_1.selectbox("专项/一般类型", options=['一般', '专项'])
    target_balance_yi = col_r2_2.number_input("余额 (亿元)", min_value=0.01, value=10.0, step=0.1, format='%.2f')
    target_coupon = col_r2_3.number_input("票面利率 (%)", min_value=0.01, value=3.20, step=0.01, format='%.2f')
    
    if target_balance_yi <= 0:
        col_r2_4.warning("余额必须大于 0 亿元。")
        return

    target_row = df_latest_for_model[df_latest_for_model['区域_Clean'] == target_province_clean].iloc[0]
    target_region = target_row['区域等级']
    st.caption(f"系统确定 **{target_province_clean}** 属于 **{target_region}** 区域等级进行预测。")

    params = spread_model.params * 10000 
    gamma_0 = params.get('const', 0)
    gamma_B1 = params.get('区域等级_B1', 0) 
    gamma_B2 = params.get('区域等级_B2', 0) 
    gamma_B3 = params.get('区域等级_B3', 0) 
    gamma_C = params.get('区域等级_C', 0)
    gamma_ln_balance = params.get('余额_ln', 0)
    gamma_special = params.get('Is_Special', 0)
    gamma_new = params.get('Is_New', 0)
    gamma_taxable = params.get('Is_Taxable', 0)
    gamma_C_Spread = params.get('C_Spread', 0)
    gamma_C_Spread_Int = params.get('C_Spread_Taxable_Int', 0)

    base_yield = get_base_yield(target_term, target_tax)
    
    if np.isnan(base_yield):
        col_r2_4.warning("无法计算基准收益率，请检查期限是否在样本范围内。")
        return
        
    spread_pred_decimal = 0.0
    spread_pred_decimal += gamma_0 / 10000 

    if target_region == 'B1':
        spread_pred_decimal += gamma_B1 / 10000
    elif target_region == 'B2':
        spread_pred_decimal += gamma_B2 / 10000
    elif target_region == 'B3':
        spread_pred_decimal += gamma_B3 / 10000
    elif target_region == 'C':
        spread_pred_decimal += gamma_C / 10000
        
    ln_balance = np.log(target_balance_yi)
    spread_pred_decimal += (gamma_ln_balance / 10000) * ln_balance
    
    is_special = 1 if target_special == '专项' else 0
    if is_special == 1:
        spread_pred_decimal += gamma_special / 10000

    is_taxable = 1 if target_tax == '是' else 0
    if is_taxable == 1:
        spread_pred_decimal += gamma_taxable / 10000
        
    C_Spread = target_coupon - base_yield
    
    spread_pred_decimal += (gamma_C_Spread / 10000) * C_Spread
    
    if is_taxable == 1:
        spread_pred_decimal += (gamma_C_Spread_Int / 10000) * C_Spread
    
    final_yield = base_yield + spread_pred_decimal
    
    col_r2_4.metric(
        "📈 合理收益率定价结果", 
        f"{final_yield:.4f}%",
        delta=f"基准收益率: {base_yield:.4f}%"
    )

    st.subheader("溢价模型（OLS 利差回归）")
    st.markdown("回归公式")
    st.latex(r"""
        \text{利差} = \gamma_0 + \gamma_{B1}\,\mathbb{1}_{B1} + \gamma_{B2}\,\mathbb{1}_{B2} + \gamma_{B3}\,\mathbb{1}_{B3} + \gamma_{C}\,\mathbb{1}_{C} + \gamma_{\ln \text{余额}}\cdot\ln(\text{余额}) + \gamma_{\text{专项}}\,\mathbb{1}_{\text{专项}} + \gamma_{\text{新券}}\,\mathbb{1}_{\text{新券}} + \gamma_{\text{应税}}\,\mathbb{1}_{\text{应税}} + \gamma_{\text{票面}}\cdot(\text{票面} - \text{基准收益率}) + \gamma_{\text{票面×应税}}\cdot\mathbb{1}_{\text{应税}}\cdot(\text{票面} - \text{基准收益率}) + \varepsilon
    """)
    st.caption("OLS 回归结果概览")

    results_df = pd.DataFrame({
        '系数 (BP)': spread_model.params * 10000,
        'P 值 (P>|t|)': spread_model.pvalues,
    })

    sig_marks = []
    for p in results_df['P 值 (P>|t|)']:
        if p < 0.01:
            sig_marks.append('***')
        elif p < 0.05:
            sig_marks.append('**')
        elif p < 0.1:
            sig_marks.append('*')
        else:
            sig_marks.append('')
    results_df['显著性'] = sig_marks

    results_df.rename(index={
        'const': '截距项 (A级基础利差)',
        '区域等级_B1': '区域等级_B1 (相对A级的溢价)',
        '区域等级_B2': '区域等级_B2 (相对A级的溢价)',
        '区域等级_B3': '区域等级_B3 (相对A级的溢价)',
        '区域等级_C': '区域等级_C (相对A级的溢价)',
        '余额_ln': 'ln(余额)',
        'Is_Special': '专项债哑变量',
        'Is_New': '新发行券哑变量',
        'Is_Taxable': '是否交税哑变量',
        'C_Spread': '票面利差主效应 (Coupon - Base_Yield)',
        'C_Spread_Taxable_Int': '票面利差*应税交互项',
    }, inplace=True)

    display_order = [
        '截距项 (A级基础利差)',
        '区域等级_B1 (相对A级的溢价)',
        '区域等级_B2 (相对A级的溢价)',
        '区域等级_B3 (相对A级的溢价)',
        '区域等级_C (相对A级的溢价)',
        'ln(余额)',
        '专项债哑变量',
        '新发行券哑变量',
        '是否交税哑变量',
        '票面利差主效应 (Coupon - Base_Yield)',
        '票面利差*应税交互项',
    ]
    results_df = results_df.loc[[idx for idx in display_order if idx in results_df.index]]

    st.dataframe(results_df.style.format({
        '系数 (BP)': "{:.2f}",
        'P 值 (P>|t|)': "{:.4f}",
    }), use_container_width=True)

    coef_plot_df = results_df.reset_index().rename(columns={'index': '变量'})
    coef_chart = alt.Chart(coef_plot_df).mark_bar().encode(
        x=alt.X('系数 (BP):Q', title='系数 (BP)'),
        y=alt.Y('变量:N', sort='-x'),
        color=alt.Color('显著性:N', scale=alt.Scale(domain=['***','**','*',''], range=['#2b8a3e','#4dabf7','#fab005','#adb5bd']))
    )
    st.altair_chart(coef_chart, use_container_width=True)
    
    # 单独展示 R^2
    r2 = spread_model.rsquared * 100
    st.markdown(f"**模型解释度 ($R^2$)**: **{r2:.2f}%**")

    st.markdown("---")
    # =================================================================

    # --- 阶段三：交互式预测器 (保持不变) ---

    # --- 阶段四：税收利差曲线表格输出 (保持不变) ---

if __name__ == '__main__':
    main_app()
