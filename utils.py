import pandas as pd
import numpy as np
from datetime import timedelta
import re

# ***************************************************************
# 1. 配置参数
# ***************************************************************

# 定义 A, C 匹配的简洁列表 (用于 create_region_level 内部匹配)
# 【注：这里的名称已经清理掉后缀，如'省'/'自治区'，匹配时需要使用清理后的名称】
A_MATCH_LIST = ['浙江', '广东', '北京', '上海', '江苏']
C_MATCH_LIST = ['云南', '贵州', '内蒙古', '黑龙江', '吉林', '辽宁', '天津', '西藏', '甘肃']

# 定义 Streamlit App 中下拉菜单的选项 (包含 B1, B2, B3)
REGION_LEVELS = ['A', 'B1', 'B2', 'B3', 'C'] 

# 定义新券的月数阈值
NEW_ISSUE_MONTHS = 6 

# ***************************************************************
# 2. 区域等级处理函数 (新增省份清理函数)
# ***************************************************************

def get_clean_region_name(region):
    """ 清理省份名称，移除所有后缀，用于匹配和前端展示。"""
    if pd.isna(region):
        return None
    # 移除 '省', '市', '自治区', '维吾尔', '壮族', '回族' 等后缀
    return str(region).replace('省', '').replace('市', '').replace('自治区', '').replace('维吾尔', '').replace('壮族', '').replace('回族', '').strip()

def create_region_level(region):
    """ 
    定义区域信用等级：A (好), B (中), C (差)。
    此函数返回原始的 A/B/C 三级分类。
    """
    if pd.isna(region):
        return 'B' 
        
    # 使用清理函数进行匹配
    clean_region = get_clean_region_name(region)
    
    if clean_region in A_MATCH_LIST: 
        return 'A'
        
    if clean_region in C_MATCH_LIST:
        return 'C'
        
    # 其余为 B 区
    return 'B'

# ***************************************************************
# 3. 数据加载、清洗和特征工程 (保持不变)
# ***************************************************************

def load_data(file_upload):
    # ... (此函数体保持与上次更新一致，因为它依赖 create_region_level) ...
    # 确保 load_data 中对 df_latest_day 的特征工程保持不变
    # a. 区域等级 (使用原始 A/B/C 分类)
    # df_latest_day['区域等级'] = df_latest_day['区域'].apply(create_region_level)
    # ...
    # 为了简洁，此处省略 load_data 完整代码，但请确保与上次提供的一致
    
    if file_upload is None:
        return None, None
    
    try:
        if file_upload.name.endswith('.xlsx') or file_upload.name.endswith('.xls'):
            df = pd.read_excel(file_upload)
        else:
            df = pd.read_csv(file_upload)
    except Exception:
        return None, None
    
    df.columns = df.columns.astype(str).str.strip()

    num_cols = ['剩余年限', '收盘收益率', '估值', '票面', '余额', '成交量']
    date_cols = ['当前日期', '发行日期']
    
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce') 

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    if '当前日期' not in df.columns or df['当前日期'].isna().all():
        return None, None 
        
    latest_date = df['当前日期'].max()
    date_threshold = latest_date - timedelta(days=5) 
    df_recent = df[df['当前日期'] >= date_threshold].copy()
    
    if df_recent.empty:
        return None, latest_date
        
    df_latest_day = df_recent[df_recent['当前日期'] == latest_date].copy()
    
    # --- 4. 核心特征工程 ---
    df_latest_day['区域等级'] = df_latest_day['区域'].apply(create_region_level)
    df_latest_day['余额_ln'] = np.log(df_latest_day['余额'].where(df_latest_day['余额'] > 0))
    df_latest_day['Is_Special'] = df_latest_day['专项一般'].apply(lambda x: 1 if '专项' in str(x) else 0)
    df_latest_day['Is_Taxable'] = df_latest_day['是否交税'].apply(lambda x: 1 if str(x).upper() in ['是', 'YES', 'Y'] else 0)
    
    df_latest_day['Age_Days'] = (latest_date - df_latest_day['发行日期']).dt.days
    df_latest_day['Is_New'] = df_latest_day['Age_Days'].apply(
        lambda x: 1 if x < NEW_ISSUE_MONTHS * 30.4 else 0 
    ).astype(int)

    # --- 5. 清理用于模型的关键数据 ---
    df_filtered = df_latest_day.dropna(subset=[
        '剩余年限', '收盘收益率', '区域等级', '是否交税', '余额_ln', 'Is_Special', 'Is_New', 'Is_Taxable', '票面'
    ]).copy()
    
    if df_filtered.empty or df_filtered['剩余年限'].nunique() < 4:
         return None, latest_date

    return df_filtered, latest_date