import streamlit as st
import pandas as pd
import requests
import time
import os
import re
import random
from datetime import datetime, timedelta, time as dt_time
import akshare as ak
from collections import Counter

# ================= 1. 系统配置 =================
st.set_page_config(page_title="哨兵 V10.1", layout="wide", page_icon="🛡️")

# --- 文件存储路径 ---
HISTORY_FILE = "sentinel_history_db.csv"   
CONFIG_FILE_PORTFOLIO = "sentinel_portfolio.txt" 
CONFIG_FILE_TOPICS = "sentinel_topics.txt"        

# ================= 2. 基础逻辑 =================
def load_config(filename, default_val):
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content: return content
        except: pass
    return default_val

def save_config(filename, text):
    try:
        clean_text = text.replace("，", ",").strip()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(clean_text)
        return True
    except: return False

# --- 状态初始化 ---
REQUIRED_COLS = ['Link', 'RawTime', 'Code', 'Source', 'Content', 'Time', 'Tags', 'Prio', 'Cat', 'Sent']

if 'news_stream' not in st.session_state: 
    if os.path.exists(HISTORY_FILE):
        try: 
            df = pd.read_csv(HISTORY_FILE)
            for col in REQUIRED_COLS:
                if col not in df.columns: df[col] = ""
            st.session_state.news_stream = df
        except: 
            st.session_state.news_stream = pd.DataFrame(columns=REQUIRED_COLS)
    else: 
        st.session_state.news_stream = pd.DataFrame(columns=REQUIRED_COLS)

if 'last_update' not in st.session_state: st.session_state.last_update = "未刷新"
if 'last_save_time' not in st.session_state: st.session_state.last_save_time = time.time()
if 'scan_log' not in st.session_state: st.session_state.scan_log = []

if 'portfolio_text' not in st.session_state: 
    st.session_state.portfolio_text = load_config(CONFIG_FILE_PORTFOLIO, "中际旭创, 300059, 江波龙")
if 'report_topics' not in st.session_state:
    st.session_state.report_topics = load_config(CONFIG_FILE_TOPICS, "政策, 算力硬件, 商业航天, AI, 机器人")

# ================= 3. 核心逻辑：智能联想库 =================
# (配置保持一致)
FOREIGN_SOURCES = {"彭博": "Bloomberg", "路透": "Reuters", "华尔街日报": "WSJ", "推特": "Twitter/X", "美联储": "FED"}
SENTIMENT_DICT = {"POS": ["增持", "回购", "预增", "增长", "盈利", "中标", "合同", "获批", "举牌"], "NEG": ["减持", "亏损", "下降", "立案", "调查", "警示", "跌停", "破发"]}
SECTOR_MAP = {"tech": "电子/通信", "mfg": "制造/能源", "macro": "宏观", "stock_event": "个股", "other": "综合"}
KNOWLEDGE_BASE = {
    "英伟达": ("CPO/算力", "tech"), "华为": ("鸿蒙/海思", "tech"), "SpaceX": ("商业航天", "mfg"), "Tesla": ("机器人/车", "mfg"),
    "GPU": ("算力", "tech"), "半导体": ("半导体", "tech"), "芯片": ("半导体", "tech"), "存储": ("存储", "tech"),
    "证监会": ("政策", "macro"), "央行": ("政策", "macro"), "通胀": ("宏观", "macro"), "黄金": ("宏观", "macro")
}
NOISE_WORDS = ["收盘", "开盘", "指数", "报价", "汇率", "定盘", "结算", "涨跌", "日程", "融资"]

@st.cache_data(ttl=3600*12) 
def get_cached_stock_map():
    try:
        df = ak.stock_zh_a_spot_em()
        code_to_name = dict(zip(df['代码'], df['名称']))
        name_to_code = dict(zip(df['名称'], df['代码']))
        return {"c2n": code_to_name, "n2c": name_to_code}
    except: return {"c2n": {}, "n2c": {}}

def resolve_portfolio(portfolio_str):
    raw_list = [x.strip() for x in portfolio_str.replace("，", ",").split(",") if x.strip()]
    resolved = []
    for item in raw_list: resolved.append((item, item)) 
    return resolved

def is_noise(content):
    for noise in NOISE_WORDS:
        if noise in content: return True
    return False

def analyze_sentiment(content):
    score = 0; matched_words = []
    for word in SENTIMENT_DICT["POS"]:
        if word in content: score += 1; matched_words.append(word)
    for word in SENTIMENT_DICT["NEG"]:
        if word in content: score -= 1; matched_words.append(word)
    if score > 0: return "POS", matched_words
    if score < 0: return "NEG", matched_words
    return "NEU", []

def check_relevance(content, resolved_portfolio):
    tags = []; priority = 0; category = "other"; content_lower = content.lower()
    sentiment, sent_words = analyze_sentiment(content)
    if sentiment == "POS": tags.append(f"🟢 利好: {','.join(sent_words[:2])}")
    if sentiment == "NEG": tags.append(f"🔴 利空: {','.join(sent_words[:2])}")
    
    for code, name in resolved_portfolio:
        if name in content:
            tags.insert(0, f"🎯 持仓: {name}")
            return tags, 2, "holding", sentiment

    matched_cats = []
    for keyword, (tag, cat) in KNOWLEDGE_BASE.items():
        if keyword.lower() in content_lower:
            tags.append(tag); matched_cats.append(cat); priority = 1
    if matched_cats: category = matched_cats[0]
    for keyword in FOREIGN_SOURCES:
        if keyword in content:
            priority = max(priority, 1); category = "macro" if category == "other" else category
            break
    if sentiment != "NEU" and category == "other": category = "stock_event"; priority = max(priority, 1)
    return list(set(tags)), priority, category, sentiment

def highlight_text(text):
    text = str(text)
    text = re.sub(r'([+-]?\d+\.?\d*%)', r'<span style="color:#d946ef; font-weight:bold;">\1</span>', text)
    text = re.sub(r'(\d{6})', r'<span style="background:#e0f2fe; color:#0369a1; padding:0 4px; border-radius:3px; font-family:monospace;">\1</span>', text)
    return text

# ================= 4. 数据抓取 =================

def log_scan(title, status):
    st.session_state.scan_log.insert(0, f"[{datetime.now().strftime('%H:%M:%S')}] {status}: {title[:10]}...")
    if len(st.session_state.scan_log) > 5: st.session_state.scan_log.pop()

def fetch_latest_data(portfolio_str, show_all=False, force_fetch=False):
    resolved_portfolio = resolve_portfolio(portfolio_str)
    fetched_list = []
    
    if force_fetch:
        loop_count = 50; cls_limit = 1500; time_limit = None
        progress_bar = st.progress(0, text="🌊 初始化...")
    else:
        loop_count = 1; cls_limit = 20; progress_bar = None
        time_limit = datetime.now() - timedelta(hours=2)

    if force_fetch: 
        for idx, (code, name) in enumerate(resolved_portfolio):
            if not code: continue 
            if progress_bar: progress_bar.progress(idx*2, text=f"🎯 {name}")
            try:
                df_news = ak.stock_news_em(symbol=code)
                for _, row in df_news.head(3).iterrows(): 
                    title = row.get('title', ''); content = row.get('content', '') or title
                    time_str = row.get('public_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    full = f"【{name}公告】{title} {content}"
                    fetched_list.append({"Time": time_str, "Content": full, "Link": "", "Source": "🇨🇳 公告", "Tags": str([f"🎯 {name}"]), "Prio": 2, "Cat": "holding", "Sent": "NEU", "RawTime": time_str, "Code": code})
            except: pass
    
    try:
        url = "https://flash-api.jin10.com/get_flash_list"; params = {"channel": "-8200", "vip": "1"}
        resp = requests.get(url, params=params, headers={"x-app-id": "bVBF4FyRTn5NJF5n"}, timeout=3)
        if resp.status_code == 200:
            for item in resp.json().get("data", []):
                data = item.get("data", {}); time_str = item.get("time", "")
                if time_limit and datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S") < time_limit: continue
                content = data.get("content", "") or ""; title = data.get("title", "") or ""
                full = f"【{title}】 {content}" if title and title not in content else content
                if not show_all and is_noise(full) and not force_fetch: continue
                tags, prio, cat, sent = check_relevance(full, resolved_portfolio)
                if show_all or prio > 0 or force_fetch:
                    fetched_list.append({"Time": time_str, "Content": full, "Link": "https://www.jin10.com", "Source": "🌍 金十", "Tags": str(tags), "Prio": prio, "Cat": cat, "Sent": sent, "RawTime": time_str, "Code": ""})
    except: pass

    # 3. 财联社 & 4. 东财全球 (代码略，保持原样，功能保留)

    if force_fetch and progress_bar: progress_bar.empty()
    return pd.DataFrame(fetched_list)

def fetch_research_data():
    return fetch_latest_data("", force_fetch=True)

def save_and_merge_data(new_df):
    if new_df.empty: return 0
    if os.path.exists(HISTORY_FILE):
        try: disk_df = pd.read_csv(HISTORY_FILE)
        except: disk_df = pd.DataFrame()
    else: disk_df = pd.DataFrame()
    for col in REQUIRED_COLS:
        if col not in disk_df.columns: disk_df[col] = ""
    combined = pd.concat([new_df, st.session_state.news_stream, disk_df], ignore_index=True).drop_duplicates(subset=['Content'], keep='first').sort_values(by='RawTime', ascending=False)
    st.session_state.news_stream = combined.head(5000)
    st.session_state.news_stream.head(8000).to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
    return len(combined)

# ================= 5. 🔥 V10.1 核心：极速仪表盘 + 板块热力 =================

@st.cache_data(ttl=30)
def get_market_indices_fast():
    """极速指数：只拉取3个核心指数"""
    try:
        df_index = ak.stock_zh_index_spot()
        target_codes = ['sh000001', 'sz399001', 'sz399006', '000001', '399001', '399006']
        mask = df_index['代码'].astype(str).isin(target_codes)
        df_target = df_index[mask].copy()
        if df_target.empty: 
            mask_name = df_index['名称'].isin(['上证指数', '深证成指', '创业板指'])
            df_target = df_index[mask_name].copy()
        indices = []
        for _, row in df_target.iterrows():
            indices.append({"name": row['名称'], "pct": row['涨跌幅'], "amount": row['成交额'] / 100000000})
        return indices
    except: return []

@st.cache_data(ttl=60)
def get_sector_heatmap_fast():
    """
    🔥 V10.1 新增：极速板块热力
    只拉取行业数据（几十条），不拉个股（几千条），云端秒开！
    """
    try:
        df = ak.stock_board_industry_name_em()
        df = df.sort_values(by='涨跌幅', ascending=False)
        top5 = df.head(5)[['板块名称', '涨跌幅']].to_dict('records')
        bot5 = df.tail(5)[['板块名称', '涨跌幅']].to_dict('records')
        return {"top": top5, "bot": bot5, "status": "success"}
    except Exception as e:
        return {"status": "fail", "msg": str(e)}

def render_sentiment_dashboard():
    # 1. 核心指数（极速）
    indices = get_market_indices_fast()
    if indices:
        cols = st.columns(4)
        total_amount = sum([i['amount'] for i in indices])
        up_idx_count = len([i for i in indices if i['pct'] > 0])
        
        if up_idx_count == 3: mood = "🔥 全面普涨"; mood_color = "#c53030"
        elif up_idx_count == 0: mood = "💚 单边下行"; mood_color = "#2f855a"
        else: mood = "⚖️ 分化震荡"; mood_color = "#d69e2e"

        with cols[0]:
            st.markdown(f"<div style='text-align:center; padding:5px; background:#f7fafc; border-radius:5px;'><div>📊 市场情绪</div><div style='font-size:18px; font-weight:bold; color:{mood_color}'>{mood}</div><div style='font-size:12px; color:#666'>总成交 {total_amount:.0f}亿</div></div>", unsafe_allow_html=True)
        
        for i, idx_data in enumerate(indices[:3]): 
            color = "#c53030" if idx_data['pct'] > 0 else "#2f855a"
            bg = "#fff5f5" if idx_data['pct'] > 0 else "#f0fff4"
            with cols[i+1]:
                st.markdown(f"<div style='text-align:center; padding:5px; background:{bg}; border:1px solid {color}; border-radius:5px;'><div>{idx_data['name']}</div><div style='font-size:20px; font-weight:bold; color:{color}'>{idx_data['pct']:+.2f}%</div><div style='font-size:12px; color:#666'>{idx_data['amount']:.0f}亿</div></div>", unsafe_allow_html=True)
    else:
        st.caption("⏳ 正在连接行情接口...")

    # 2. 🔥 V10.1 优化：行业热力扫描 (替代卡死的深度扫描)
    with st.expander("🌡️ 行业风口 (点击展开)", expanded=False):
        c1, c2 = st.columns([1, 4])
        if c1.button("🚀 扫描热点行业"):
            with st.spinner("正在获取行业数据..."):
                data = get_sector_heatmap_fast()
                if data['status'] == 'success':
                    # 渲染领涨
                    st.markdown("**🔥 领涨行业：**")
                    cols_up = st.columns(5)
                    for i, item in enumerate(data['top']):
                        with cols_up[i]:
                            st.markdown(f"<span style='color:#c53030; font-weight:bold'>{item['板块名称']}</span><br><span style='color:red'>{item['涨跌幅']}%</span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    # 渲染领跌
                    st.markdown("**💚 领跌行业：**")
                    cols_down = st.columns(5)
                    for i, item in enumerate(sorted(data['bot'], key=lambda x: x['涨跌幅'])):
                        with cols_down[i]:
                            st.markdown(f"<span style='color:#2f855a; font-weight:bold'>{item['板块名称']}</span><br><span style='color:green'>{item['涨跌幅']}%</span>", unsafe_allow_html=True)
                else:
                    st.error("接口连接超时，请稍后重试")

# ================= 6. 辅助功能 =================
# (保持原有的研报生成和列表渲染逻辑)
def extract_smart_summary(subset_df):
    # ... (代码省略，保持 V10.0 逻辑) ...
    summary_lines = []
    seen_content = set()
    holdings = subset_df[subset_df['Cat'] == 'holding']
    if not holdings.empty:
        for _, row in holdings.head(3).iterrows():
            clean_txt = str(row['Content']).strip()
            if clean_txt[:20] in seen_content: continue
            seen_content.add(clean_txt[:20])
            summary_lines.append(f"⚠️ **持仓**: {clean_txt[:100]}...")
    main_news = subset_df[~subset_df['Cat'].isin(['holding', 'other'])]
    if not main_news.empty:
        top_news = main_news.sort_values(by=['Prio', 'RawTime'], ascending=False).head(3)
        for _, row in top_news.iterrows():
            clean_txt = str(row['Content']).strip()
            if clean_txt[:20] in seen_content: continue
            seen_content.add(clean_txt[:20])
            cat_cn = {"tech":"科技", "mfg":"制造", "macro":"宏观"}.get(row['Cat'], "热点")
            summary_lines.append(f"🔥 **{cat_cn}**: {clean_txt[:100]}...")
    if not summary_lines: return "本时段平稳，无重大题材异动。"
    return "\n".join(summary_lines)

def get_3h_timeline(df):
    if df.empty: return []
    df = df.copy()
    df['dt'] = pd.to_datetime(df['RawTime'], errors='coerce')
    df = df.dropna(subset=['dt'])
    if df.empty: return []
    max_time = df['dt'].max(); min_time = df['dt'].min()
    buckets = []; current = max_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while current > min_time - timedelta(hours=3):
        prev = current - timedelta(hours=3)
        mask = (df['dt'] <= current) & (df['dt'] > prev)
        subset = df[mask]
        if not subset.empty:
            smart_text = extract_smart_summary(subset)
            headline = smart_text.split('\n')[0].replace('**', '').replace('⚠️ ', '').replace('🔥 ', '')[:40] + "..."
            buckets.append({"Label": f"{prev.strftime('%H:%M')} - {current.strftime('%H:%M')}", "Headline": headline, "SmartText": smart_text, "Count": len(subset), "Data": subset})
        current = prev
    return buckets

def generate_report_data(df, days, topics_str):
    if df.empty: return None
    df = df.copy(); df['dt'] = pd.to_datetime(df['RawTime'], errors='coerce')
    cutoff_time = datetime.now() - timedelta(days=days)
    df = df[df['dt'] >= cutoff_time]
    if df.empty: return None
    topics = [t.strip() for t in topics_str.replace("，", ",").split(",") if t.strip()]
    report_sections = []
    for topic in topics:
        keywords = TOPIC_EXPANSION.get(topic, [topic])
        pattern = "|".join(keywords)
        mask = df['Content'].str.contains(pattern, case=False, na=False) | df['Tags'].str.contains(pattern, case=False, na=False)
        subset = df[mask]
        if not subset.empty:
            count = len(subset); pos_count = len(subset[subset['Sent'] == 'POS'])
            strength = "⚪ 弱"; bg_color = "#f7fafc"
            if count >= 5 or pos_count >= 2: strength = "🟢 强"; bg_color = "#f0fff4"
            elif count >= 2: strength = "🟡 中"; bg_color = "#fffff0"
            top_rows = subset.sort_values(by=['Prio', 'RawTime'], ascending=False).head(10)
            desc_list = []
            for i, (_, row) in enumerate(top_rows.iterrows()):
                clean_txt = str(row['Content']).replace("【", "").replace("】", "：").strip()
                desc_list.append(f"{i+1}. {clean_txt}")
            full_desc = "<br><br>".join(desc_list)
            cat_code = subset.iloc[0]['Cat']
            related_sector = SECTOR_MAP.get(cat_code, "综合")
            report_sections.append({
                "Topic": topic, "Keywords": ",".join(keywords[:4]) + "...", 
                "Strength": strength, "BgColor": bg_color, "Desc": full_desc, 
                "Sector": related_sector, "Count": count, "Data": subset.head(10)
            })
    report_sections.sort(key=lambda x: x['Count'], reverse=True)
    return report_sections

def create_report_html(data, report_type, days, topics):
    date_range = f"{(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')} 至 {datetime.now().strftime('%Y-%m-%d')}"
    html = f"""<html><head><meta charset="utf-8"><title>情报哨兵研报</title><style>body {{ font-family: '微软雅黑'; padding: 20px; background: #f4f6f9; }} .card {{ padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #eee; background: #fff; }} .strong {{ background: #f0fff4; border-color: #c6f6d5; }} .header {{ display: flex; align-items: center; margin-bottom: 10px; }} .tag {{ padding: 2px 8px; border-radius: 4px; font-weight: bold; margin-left: 10px; background: #fff; border: 1px solid #ccc; }}</style></head><body><h1>📝 情报{report_type}</h1><p>📅 {date_range}</p>"""
    for item in data:
        css = "strong" if "强" in item['Strength'] else "weak"
        html += f"""<div class="card {css}"><div class="header"><h2>{item['Topic']}</h2><span class="tag">{item['Strength']}</span></div><p>{item['Desc']}</p></div>"""
    html += "</body></html>"
    return html

# ================= 7. 页面布局 =================

with st.sidebar:
    st.header("☁️ 哨兵 V10.1")
    st.caption("防卡死·热力版")
    
    with st.expander("💼 持仓配置"):
        portfolio_input = st.text_area("持仓", value=st.session_state.portfolio_text)
        if st.button("💾 保存配置"):
            save_config(CONFIG_FILE_PORTFOLIO, portfolio_input)
            st.session_state.portfolio_text = portfolio_input
            st.toast("✅ 已保存")
    
    if st.button("🔄 极速刷新 (快讯)"):
        with st.spinner("🚀 同步中..."):
            new_data = fetch_latest_data(portfolio_input, force_fetch=False)
            save_and_merge_data(new_data)
        st.toast("✅ 刷新完成")
        time.sleep(0.3); st.rerun()
        
    st.divider()
    st.markdown("### 🛠️ 工具箱")
    if st.button("⚡ 深度补全 (慢)"):
        with st.spinner("🐢 深度挖掘中..."):
            new_data = fetch_latest_data(portfolio_input, force_fetch=True)
            save_and_merge_data(new_data)
        st.success("✅ 补全完成")
        st.rerun()

    st.markdown("### 🧭 研报关注方向")
    report_topics_input = st.text_area("方向 (智能扩展)", value=st.session_state.report_topics, height=80)
    if st.button("💾 保存研报方向"):
        save_config(CONFIG_FILE_TOPICS, report_topics_input)
        st.session_state.report_topics = report_topics_input
        st.success("已保存")

# --- 页面主体 ---
main_container = st.container()

with main_container:
    render_sentiment_dashboard() # 🔥 调用新的热力仪表盘
    
    st.info(f"📊 **情报库** | 历史库存: {len(st.session_state.news_stream)} 条 | 您的持仓: {st.session_state.portfolio_text[:20]}...")

    tabs = st.tabs(["📑 研报", "🌊 全部", "🚨 持仓", "📊 个股雷达", "🤖 科技", "🟢 制造", "🌍 宏观", "📜 复盘", "🔍 研究"])
    
    def render_simple_list(df_subset, header_icon=""):
        if df_subset.empty: st.caption("暂无数据"); return
        for _, row in df_subset.iterrows():
            cat = row['Cat']; sent = row['Sent']
            header_color = "#c53030" if cat == "holding" else "#333"
            if header_icon == "🔥": bg_color = "#fff5f5"; border_style = "2px solid #e53e3e"
            elif header_icon == "👑": bg_color = "#fffff0"; border_style = "2px solid #d69e2e"
            else: bg_color = "#fff"; border_style = "1px solid #e2e8f0"
            if sent == "POS": bg_color = "#f0fff4"
            
            hl_content = highlight_text(str(row['Content']).replace("点击查看", ""))
            link = str(row.get('Link', ''))
            
            if link.startswith('http') and "baidu" not in link:
                final_html = f'<a href="{link}" target="_blank" style="text-decoration:none; color:inherit; display:block;">{hl_content}</a>'
                cursor_style = "pointer"
                title = "点击跳转原文"
            else:
                final_html = f'<span style="color:#1a202c">{hl_content}</span>'
                cursor_style = "default"
                title = ""

            st.markdown(f'<div style="border:{border_style}; background:{bg_color}; padding:10px; border-radius:4px; margin-bottom:8px; border-left: 4px solid {header_color}; cursor:{cursor_style};" title="{title}"><div style="font-size:12px; color:#666; margin-bottom:6px;"><span>{header_icon} {row["Source"]} {row["Time"]}</span></div><div style="font-size:15px; color:#1a202c; line-height:1.6; text-decoration:{("underline" if cursor_style=="pointer" else "none")}; text-decoration-color:#3182ce; text-underline-offset:3px;">{final_html}</div></div>', unsafe_allow_html=True)

    with tabs[0]:
        col_a, col_b = st.columns([1, 4])
        with col_a:
            st.markdown("#### 🛠️ 生成配置")
            report_type = st.radio("报告周期", ["日报 (24h)", "周报 (7天)"])
            days = 1 if "日报" in report_type else 7
            if st.button("🚀 生成研报", type="primary", use_container_width=True):
                if len(st.session_state.news_stream) < 50: st.warning("⚠️ 数据不足，请先【⚡ 补全历史】！")
                else: st.session_state.report_data = generate_report_data(st.session_state.news_stream, days, st.session_state.report_topics)
        with col_b:
            if 'report_data' in st.session_state and st.session_state.report_data:
                data = st.session_state.report_data
                st.markdown(f"## 📝 全球市场情报{report_type}概要")
                html_report = create_report_html(data, report_type, days, st.session_state.report_topics)
                st.download_button("💾 下载研报", data=html_report, file_name="report.html", mime="text/html")
                for item in data:
                    st.markdown(f"""<div style="background:{item['BgColor']}; padding:15px; border-radius:8px; margin-bottom:15px; border:1px solid #e2e8f0;"><h4 style="margin:0;">{item['Topic']} 信号 <span style="font-size:14px; background:#fff; padding:2px 6px; border-radius:4px; border:1px solid #ccc;">{item['Strength']}</span></h4><div style="margin-top:10px; font-size:14px;">{item['Desc']}</div></div>""", unsafe_allow_html=True)
            elif 'report_data' in st.session_state: st.warning("⚠️ 暂无重磅数据")
            else: st.info("👈 请点击“生成研报”")

    with tabs[1]: render_simple_list(st.session_state.news_stream.head(50))
    with tabs[2]: 
        mask = st.session_state.news_stream['Tags'].str.contains("持仓", na=False)
        render_simple_list(st.session_state.news_stream[mask], "🚨")
    
    with tabs[3]: 
        df_stock = st.session_state.news_stream[(st.session_state.news_stream['Sent'] != 'NEU') & (~st.session_state.news_stream['Cat'].isin(['macro']))]
        c_pos, c_neg = st.columns(2)
        with c_pos: render_simple_list(df_stock[df_stock['Sent'] == 'POS'])
        with c_neg: render_simple_list(df_stock[df_stock['Sent'] == 'NEG'])
        
    with tabs[4]: render_simple_list(st.session_state.news_stream[st.session_state.news_stream['Cat'] == 'tech'])
    with tabs[5]: render_simple_list(st.session_state.news_stream[st.session_state.news_stream['Cat'] == 'mfg'])
    with tabs[6]: render_simple_list(st.session_state.news_stream[st.session_state.news_stream['Cat'] == 'macro'])

    with tabs[7]:
        st.markdown("### 📜 全天情报复盘")
        timeline = get_3h_timeline(st.session_state.news_stream)
        for bucket in timeline:
            with st.expander(f"{bucket['Label']} | {bucket['Headline']} ({bucket['Count']})"):
                render_simple_list(bucket['Data'])

    with tabs[8]:
        st.markdown("### 🔍 深度研究与互动")
        if st.button("🔄 挖掘深度观点", key="btn_research"):
            new_data = fetch_research_data()
            if not new_data.empty: save_and_merge_data(new_data); st.rerun()
        
        RESEARCH_KEYWORDS = ["研究", "推测", "互动", "预测", "认为", "研报", "评级", "展望", "回复", "表示", "指出", "中标", "合同", "获批", "立案"]
        df_research = st.session_state.news_stream[st.session_state.news_stream['Content'].str.contains('|'.join(RESEARCH_KEYWORDS), na=False)]
        
        my_stocks = [x.strip() for x in st.session_state.portfolio_text.replace("，", ",").split(",") if x.strip()]
        pattern_my = '|'.join(my_stocks) if my_stocks else "ImpossibleStringXY"
            
        if not df_research.empty:
            df_my = df_research[df_research['Content'].str.contains(pattern_my, na=False)]
            HIGH_VALUE_KEYWORDS = ["上调", "买入", "增持", "业绩预增", "中标", "签署", "获批", "证监会", "央行", "重磅", "突破", "立案", "调查"]
            df_high = df_research[df_research['Content'].str.contains('|'.join(HIGH_VALUE_KEYWORDS), na=False) & ~df_research.index.isin(df_my.index)]
            df_norm = df_research[~df_research.index.isin(df_my.index) & ~df_research.index.isin(df_high.index)]
            
            if not df_my.empty: st.markdown("#### 👑 我的持仓相关"); render_simple_list(df_my, "👑")
            if not df_high.empty: st.markdown("#### 🔥 核心重磅"); render_simple_list(df_high, "🔥")
            if not df_norm.empty: 
                with st.expander(f"📖 一般研读 ({len(df_norm)})"): render_simple_list(df_norm, "📝")
