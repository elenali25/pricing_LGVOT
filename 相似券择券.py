import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt 
from pathlib import Path

# ***************************************************************
# 1. 区域等级定义与查找 (保持不变)
# ***************************************************************

A_REGIONS_CORE = ['广东', '浙江', '北京', '上海', '深圳', '江苏', '宁波', '厦门', '广州']
C_REGIONS_CORE = ['云南', '贵州', '内蒙古', '黑龙江', '吉林', '辽宁', '天津', '西藏', '海南', '广西壮族', '青海']

def create_region_level(region):
    """ 定义区域信用等级：A (好), B (中), C (差) """
    clean_region = region.replace('省', '').replace('市', '').replace('自治区', '').strip()
    
    if clean_region in A_REGIONS_CORE: 
        return 'A'
    elif clean_region in C_REGIONS_CORE:
        return 'C'
    else:
        return 'B'

def find_region_and_level(user_input_region, all_regions_data):
    """ 根据用户输入模糊匹配省份，并返回该省份的全名和区域等级。"""
    if not user_input_region:
        return None, None
        
    clean_input = user_input_region.replace('省', '').replace('市', '').replace('自治区', '').strip()
    
    for full_region in all_regions_data:
        if full_region.replace('省', '').replace('市', '').replace('自治区', '').strip() == clean_input:
            level = create_region_level(full_region)
            return full_region, level
            
    for full_region in all_regions_data:
        clean_full_region = full_region.replace('自治区', '').replace('省', '').replace('市', '')
        if clean_input in clean_full_region:
            level = create_region_level(full_region)
            return full_region, level
            
    return None, None 

# ***************************************************************
# 2. 数据加载与预处理 (保持不变)
# ***************************************************************

@st.cache_data
-def load_data(uploaded_file_or_path):
+def load_data(uploaded_file_or_path, version_key=None):
    """
    加载用户上传的文件，并进行日期筛选和数据清洗。
    """
    # 兼容 UploadedFile 与本地文件路径两种来源
    filename = uploaded_file_or_path.name if hasattr(uploaded_file_or_path, 'name') else str(uploaded_file_or_path)
    st.info(f"正在加载文件: {Path(filename).name}...")

    try:
        file_src = uploaded_file_or_path if hasattr(uploaded_file_or_path, 'name') else filename
        if filename.endswith('.csv'):
            df = pd.read_csv(file_src)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_src, sheet_name=0)
        else:
            st.error("错误：文件格式不支持。")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"加载数据时发生错误: {e}")
        return pd.DataFrame()
        
    df.columns = df.columns.astype(str).str.strip() 
    if '是否免税' in df.columns:
        df.rename(columns={'是否免税': '是否交税'}, inplace=True)
    
    for col in ['发行日期', '当前日期']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
    for col in ['剩余年限', '收盘收益率', '估值', '票面', '余额', '成交量']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 确保关键分类字段是字符串类型
    for col in ['债券代码', '是否交税', '专项一般', '区域']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if '当前日期' not in df.columns:
        st.error("错误：数据中缺少 '当前日期' 列，无法进行最新日期筛选。")
        return pd.DataFrame()
        
    latest_date = df['当前日期'].max()
    unique_dates = df['当前日期'].dropna().unique()
    # 筛选最近 5 个交易日的数据，作为筛选基础数据源
    recent_dates = pd.Series(unique_dates).sort_values(ascending=False).iloc[:5] 

    df_filtered = df[df['当前日期'].isin(recent_dates)].copy() 
    
    if df_filtered.empty:
        return pd.DataFrame()

    if '区域' in df_filtered.columns:
        df_filtered['区域等级'] = df_filtered['区域'].apply(create_region_level)
    if '发行日期' in df_filtered.columns:
        df_filtered['发行年份'] = df_filtered['发行日期'].dt.year
    
    st.sidebar.info(f"数据已按最近5个交易日过滤。最新日期：{latest_date.strftime('%Y-%m-%d')}")
    
    # 返回用于筛选和可视化的基础数据框
    return df_filtered

# ***************************************************************
# 3. 筛选核心函数 (保持不变)
# ***************************************************************

TOLERANCE_LEVELS = {
    0: {'name': '最严格档', 'term_tol': 0.3, 'coupon_tol': 0.3, 'type_match': True},
    1: {'name': '放松一档', 'term_tol': 0.5, 'coupon_tol': 0.5, 'type_match': False},
    2: {'name': '放松二档', 'term_tol': 0.7, 'coupon_tol': 0.7, 'type_match': False}
}

def find_matching_bonds_by_level(df: pd.DataFrame, target: dict, region_level: str, level: int):
    """ 根据用户输入的单个目标属性和指定的公差等级进行筛选 """
    
    config = TOLERANCE_LEVELS.get(level)
    if not config:
        return pd.DataFrame()

    df_filtered = df.copy()
    
    # 1. 剩余年限
    df_filtered = df_filtered[
        (df_filtered['剩余年限'] >= (target['term'] - config['term_tol'])) & 
        (df_filtered['剩余年限'] <= (target['term'] + config['term_tol']))
    ]

    # 2. 票面利率
    df_filtered = df_filtered[
        (df_filtered['票面'] >= (target['coupon'] - config['coupon_tol'])) & 
        (df_filtered['票面'] <= (target['coupon'] + config['coupon_tol']))
    ]
    
    # 3. 专项一般
    if config['type_match']:
        df_filtered = df_filtered[df_filtered['专项一般'] == target['bond_type']]
    
    # 4. 区域维度
    if region_level:
        df_filtered = df_filtered[df_filtered['区域等级'] == region_level]
    
    # 5. 发行年份
    issue_year_tol = 1
    df_filtered = df_filtered[
        (df_filtered['发行年份'] >= (target['issue_year'] - issue_year_tol)) & 
        (df_filtered['发行年份'] <= (target['issue_year'] + issue_year_tol))
    ]
    
    # 6. 是否交税
    df_filtered = df_filtered[df_filtered['是否交税'] == target['tax_status']]
    
    if df_filtered.empty:
        return pd.DataFrame()
        
    return df_filtered


def find_matching_bonds_with_fallback(df: pd.DataFrame, target: dict, region_level: str):
    """ 自动尝试 3 个不同级别的条件，直到找到相似债券 """
    
    for level in range(3): 
        results = find_matching_bonds_by_level(df, target, region_level, level)
        if not results.empty:
            return results, level 
            
    return pd.DataFrame(), 3 

# ***************************************************************
# 4. Streamlit 主应用 
# ***************************************************************

def main():
    st.set_page_config(page_title="地方债属性匹配筛选工具", layout="wide")
    st.title("地方债相似属性分级匹配筛选工具")
    
    uploaded_file = st.sidebar.file_uploader(
        "请在左侧上传您的地方债数据文件 (.xlsx 或 .csv)", 
        type=["xlsx", "csv"]
    )
    
    df = pd.DataFrame()
    
    if uploaded_file is None:
        # 尝试使用仓库中的默认样本数据
        sample_path = Path(__file__).resolve().parent / "样本数据.xlsx"
+        sample_path = Path(__file__).resolve().parent / "样本数据.xlsx"
         if sample_path.exists():
             st.info("未上传文件，已自动加载仓库中的默认样本数据。")
-            df = load_data(str(sample_path))
+            # 将文件修改时间作为额外的缓存键，确保样本更新能触发重新加载
+            df = load_data(str(sample_path), version_key=sample_path.stat().st_mtime)
         else:
             st.info("👈 请在左侧边栏上传您的数据文件开始属性筛选。")
             st.sidebar.header("输入目标属性")
             st.sidebar.warning("数据未加载")
             return
    
    # df 是包含最近 5 个交易日数据的基础数据框
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    if df.empty:
        return
    
    # --- 侧边栏：目标属性输入 ---
    st.sidebar.header("🎯 输入目标券的 6 个筛选属性")
    
    target_term = st.sidebar.number_input("📅 剩余年限 (年)", min_value=0.0, value=5.0, step=0.1, format='%.2f')
    target_coupon = st.sidebar.number_input("🏷️ 票面利率 (%)", min_value=0.0, value=3.20, step=0.01, format='%.2f')
    target_type = st.sidebar.selectbox("📄 专项/一般类型", options=df['专项一般'].unique().tolist(), index=0)

    all_regions = df['区域'].unique().tolist()
    input_region = st.sidebar.text_input("🌍 输入省份关键词 (如: 安徽)", value="安徽").strip()
    
    matched_region, region_level = find_region_and_level(input_region, all_regions)
    
    if matched_region:
        st.sidebar.success(f"匹配到：{matched_region} (等级: {region_level})")
    else:
        st.sidebar.error("未找到匹配的省份，请检查关键词。")
    
    target_issue_year = st.sidebar.slider("🗓️ 目标发行年份 (公差±1年)", min_value=int(df['发行年份'].min()), max_value=int(df['发行年份'].max()), value=2023)
    target_tax = st.sidebar.selectbox("🧾 是否交税", options=df['是否交税'].unique().tolist(), index=0)
    
    target_attributes = {
        'term': target_term,
        'coupon': target_coupon,
        'bond_type': target_type,
        'issue_year': target_issue_year,
        'tax_status': target_tax,
    }
    
    # --- 核心筛选逻辑 (生成相似券列表) ---
    st.header("匹配结果：与目标属性相近的债券")
    
    if not matched_region:
        st.warning("请在侧边栏输入有效的省份关键词。")
        return

    with st.spinner('正在根据分级放松机制筛选匹配券...'):
        # filtered_df: 这是需要展示在表格中的相似券
        filtered_df, relaxation_level = find_matching_bonds_with_fallback(
            df, target_attributes, region_level
        )
        
    st.metric("🎉 匹配到的债券数量", filtered_df.shape[0])
    
    # 状态和公差显示 (保持不变)
    if filtered_df.empty:
        st.error("🛑 无法找到相似券，已尝试所有放松档位。")
        st.warning(
            f"请检查核心条件 (区域等级、发行年份±1Y、是否交税) 或尝试调整目标属性的输入值。"
        )
        return
        
    config = TOLERANCE_LEVELS.get(relaxation_level)
    
    if relaxation_level == 0:
        st.success(f"✅ 相似券已通过 **{config['name']}** 筛选找到。")
    elif relaxation_level == 1:
        st.warning(f"⚠️ 相似券已通过 **{config['name']}** 找到 (已放松专项/一般限制)。")
    elif relaxation_level == 2:
        st.error(f"🛑 相似券已通过 **{config['name']}** 找到 (已启用最大放松限制)。")

    st.markdown(f"""
    **使用的筛选公差范围:**
    * **剩余年限:** $\pm {config['term_tol']:.2f}$ 年
    * **票面利率:** $\pm {config['coupon_tol']:.2f}\%$
    * **专项/一般匹配:** {'是 (严格匹配)' if config['type_match'] else '否 (不要求匹配)'}
    * **发行年份:** $\pm 1$ 年 (固定)
    * **区域等级/是否交税:** 必须严格匹配 (固定)
    """)
    
    # --- 定价趋势可视化分析 (使用全市场最新数据) ---
    
    st.subheader("全市场定价趋势可视化分析 (最新日期，按区域和交税状态区分)")
    
    # 提取最新的日期数据，用于绘制背景图
    latest_date = df['当前日期'].max()
    latest_date_df = df[df['当前日期'] == latest_date].copy()
    
    if latest_date_df.empty:
        st.warning("无法找到最新交易日期的全市场数据，图表无法绘制。")
        return

    # 定义颜色映射：A(绿), B(蓝), C(红)
    color_scale = alt.Scale(
        domain=['A', 'B', 'C'], 
        range=['green', 'blue', 'red']
    )
    
    # 定义形状映射：'是' -> 三角形, '否' -> 圆形 (更通用的形状)
    shape_map = {'是': 'triangle', '否': 'circle'}
    
    # 散点图使用 mark_point 以确保形状编码可以正确覆盖，而不是被 mark_circle 限制为圆形
    scatter = alt.Chart(latest_date_df).mark_point(size=60).encode( # <<< 关键修正：mark_point
        x=alt.X('剩余年限', title='剩余年限 (年)', axis=alt.Axis(format='.2f')),
        y=alt.Y('收盘收益率', title='收盘收益率 (%)', axis=alt.Axis(format='.4f')),
        
        # 应用定制化的颜色映射 (区域等级)
        color=alt.Color('区域等级', scale=color_scale, title='区域等级 (绿/蓝/红)'),
        
        # 应用定制化的形状映射 (是否交税)
        shape=alt.Shape('是否交税', 
             scale=alt.Scale(domain=list(shape_map.keys()), range=list(shape_map.values())),
             title='是否交税 (三角/圆)'
        ),
        
        tooltip=['债券名称', alt.Tooltip('剩余年限', format='.2f'), alt.Tooltip('收盘收益率', format='.4f'), '区域等级', '是否交税']
    ).properties(
        title=f'最新交易日 ({latest_date.strftime("%Y-%m-%d")}) 市场定价曲线'
    ).interactive()

    # 趋势线 (线性回归)
    regression_line = scatter.transform_regression('剩余年限', '收盘收益率', method='linear').mark_line(color='gray', strokeDash=[3,3])

    # 合并散点图和趋势线
    chart = scatter + regression_line

    st.altair_chart(chart, use_container_width=True)
    
    # --- 筛选结果详情表格 (使用相似券 filtered_df) ---
    
    # 结果排序：按剩余年限降序
    filtered_df = filtered_df.sort_values(by='剩余年限', ascending=False)
    
    st.subheader("相似券筛选结果详情")

    display_cols = [
        '债券代码', 
        '债券名称', 
        '剩余年限', 
        '当前日期',  
        '收盘收益率', 
        '估值', 
        '票面', 
        '区域', 
        '区域等级',
        '专项一般',
        '是否交税',
        '余额',
        '成交量',
    ]
    
    st.dataframe(
        filtered_df[display_cols].style.format({
            '剩余年限': "{:.2f} 年",
            '收盘收益率': "{:.4f}%",
            '估值': "{:.4f}%",
            '票面': "{:.2f}%",
            '余额': "{:,.2f}",
            '成交量': "{:,.0f}",
            '当前日期': lambda t: t.strftime('%Y-%m-%d') if pd.notna(t) and not pd.isna(t) else ''
        }),
        width='stretch'
    )
    
    st.markdown("---")
    st.caption(f"**目标券信息：** 剩余年限 {target_attributes['term']:.2f}年 | 票面 {target_attributes['coupon']:.2f}% | 区域等级 {region_level}")

if __name__ == '__main__':
    main()
+    # 在侧边栏显示最新数据日期，便于用户核对
+    if '当前日期' in df.columns and not df['当前日期'].empty:
+        try:
+            st.sidebar.success(f"最新数据日期：{df['当前日期'].max().strftime('%Y-%m-%d')}")
+        except Exception:
+            pass