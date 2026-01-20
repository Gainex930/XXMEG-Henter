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
st.set_page_config(page_title="哨兵 V9.8", layout="wide", page_icon="☁️")

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

# --- 🔥 核心修复：状态初始化 (防崩溃) ---
# 定义必须存在的列名，防止云端空启动报错
REQUIRED_COLS = ['Link', 'RawTime', 'Code', 'Source', 'Content', 'Time', 'Tags', 'Prio', 'Cat', 'Sent']

if 'news_stream' not in st.session_state: 
    if os.path.exists(HISTORY_FILE):
        try: 
            df = pd.read_csv(HISTORY_FILE)
            # 补全可能缺失的列
            for col in REQUIRED_COLS:
                if col not in df.columns: df[col] = ""
            st.session_state.news_stream = df
        except: 
            # 读取失败，创建带表头的空表
            st.session_state.news_stream = pd.DataFrame(columns=REQUIRED_COLS)
    else: 
        # 🔥 文件不存在（云端首次运行），创建带表头的空表
        st.session_state.news_stream = pd.DataFrame(columns=REQUIRED_COLS)

if 'market_trend' not in st.session_state: st.session_state.market_trend = "初始化..." 
if 'last_update' not in st.session_state: st.session_state.last_update = "未刷新"
if 'last_save_time' not in st.session_state: st.session_state.last_save_time = time.time()
if 'scan_log' not in st.session_state: st.session_state.scan_log = []
if 'show_dashboard' not in st.session_state: st.session_state.show_dashboard = False 

if 'portfolio_text' not in st.session_state: 
    st.session_state.portfolio_text = load_config(CONFIG_FILE_PORTFOLIO, "中际旭创, 300059, 江波龙")
if 'report_topics' not in st.session_state:
    st.session_state.report_topics = load_config(CONFIG_FILE_TOPICS, "政策, 算力硬件, 商业航天, AI, 机器人")

# ================= 3. 核心逻辑：智能联想库 =================

FOREIGN_SOURCES = {
    "彭博": "Bloomberg", "路透": "Reuters", "华尔街日报": "WSJ", "推特": "Twitter/X", "美联储": "FED"
}

SENTIMENT_DICT = {
    "POS": ["增持", "回购", "预增", "增长", "扭亏", "盈利", "分红", "中标", "合同", "签署", "获批", "突破", "上线", "发布", "举牌", "买入", "跑赢", "上调"],
    "NEG": ["减持", "亏损", "下降", "预减", "立案", "调查", "警示", "问询", "处罚", "解禁", "跌停", "破发", "下修", "利空", "违约", "诉讼"]
}

SECTOR_MAP = {
    "tech": "电子/通信/半导体", "mfg": "高端制造/能源", "macro": "宏观/金融", "stock_event": "个股异动", "other": "综合"
}

BASE_POLICY = ["政策", "意见", "通知", "规划", "行动计划", "获批", "支持", "谣言", "监管", "立案", "发布", "印发", "证监会", "央行", "财政部", "发改委", "工信部", "国常会", "政治局", "降准", "降息", "专项债", "逆回购", "LPR", "房贷", "新质生产力", "数据要素", "以旧换新", "国企改革", "市值管理", "耐心资本", "研报", "解读", "分析", "点评", "策略", "展望", "预测", "研判", "券商", "证券", "评级", "增持评级", "目标价", "首席", "宏观团队", "纪要"]
BASE_COMPUTING = ["算力", "GPU", "服务器", "数据中心", "英伟达", "H20", "B200", "超算", "液冷", "智算", "CPO", "光模块", "交换机", "光通信", "东数西算", "寒武纪", "海光", "昇腾", "鲲鹏"]
BASE_HARDWARE = ["硬件", "手机", "PC", "消费电子", "面板", "显卡", "苹果", "华为", "Mate", "电子", "AI手机", "AI PC", "折叠屏", "穿戴设备", "VR", "MR", "智能家居"]
BASE_CHIP = ["半导体", "芯片", "晶圆", "集成电路", "IC", "第三代", "IGBT", "MCU", "制造", "代工", "中芯", "台积电", "华虹", "封装", "测试", "封测", "长电", "通富", "华天", "先进封装", "CoWoS", "光刻机", "蚀刻", "薄膜", "清洗", "设备", "北方华创", "中微", "组件", "零部件", "材料", "光刻胶", "靶材"]
BASE_STORAGE = ["存储", "HBM", "DRAM", "NAND", "闪存", "美光", "海力士", "长鑫", "江波龙", "佰维", "兆易"]
BASE_AEROSPACE = ["商业航天", "航天", "火箭", "卫星", "太空", "发射", "深空", "星链", "SpaceX", "G60", "垣信", "千帆", "蓝箭", "星际荣耀", "低轨", "星座", "遥感", "通信卫星", "推进", "发动机", "液氧", "甲烷", "燃料", "整流罩", "零部件", "高温合金", "碳纤维", "3D打印"]
BASE_AI = ["AI", "人工智能", "大模型", "GPT", "Sora", "生成式", "机器视觉", "Agent", "OpenAI", "豆包", "Kimi", "文心", "通义", "智谱", "月之暗面", "文生图", "文生视频", "多模态", "AIGC", "算法", "边缘计算"]
BASE_ROBOT = ["机器人", "人形", "优必选", "拓普", "三花", "绿的", "具身智能", "灵巧手", "传感器", "IMU", "视觉", "减速器", "谐波", "RV", "丝杠", "滚柱", "行星", "空心杯", "电机", "伺服"]

TOPIC_EXPANSION = {
    "政策": BASE_POLICY, "算力": BASE_COMPUTING, "硬件": BASE_HARDWARE, "半导体": BASE_CHIP, "芯片": BASE_CHIP, "存储": BASE_STORAGE, "存储芯片": BASE_STORAGE, "商业航天": BASE_AEROSPACE, "航天": BASE_AEROSPACE, "AI": BASE_AI, "机器人": BASE_ROBOT,
    "算力硬件": BASE_COMPUTING + BASE_HARDWARE, "储存": BASE_STORAGE, "储存芯片": BASE_STORAGE, "半导体产业链": BASE_CHIP,
    "低空": ["低空", "无人机", "eVTOL", "飞行汽车", "通航", "亿航", "万丰"],
    "汽车": ["汽车", "新能源车", "智驾", "自动驾驶", "特斯拉", "问界", "小米汽车", "赛力斯", "比亚迪"]
}

KNOWLEDGE_BASE = {
    "英伟达": ("CPO/算力", "tech"), "Nvidia": ("CPO/算力", "tech"), "AMD": ("芯片", "tech"), "光模块": ("CPO", "tech"), "OpenAI": ("AI应用", "tech"), "华为": ("鸿蒙/海思", "tech"), "SpaceX": ("商业航天", "mfg"), "核聚变": ("核电", "mfg"), "电力": ("电网", "mfg"), "Tesla": ("机器人/车", "mfg"), "低空": ("低空经济", "mfg"), "固态": ("固态电池", "mfg"), "脑机": ("脑机接口", "tech"), "互联网": ("工业互联网", "tech"), "平台": ("平台经济", "tech"),
    "GPU": ("算力", "tech"), "服务器": ("算力", "tech"), "半导体": ("半导体", "tech"), "芯片": ("半导体", "tech"), "存储": ("存储芯片", "tech"), "HBM": ("存储芯片", "tech"), "光刻机": ("半导体", "tech"), "封测": ("半导体", "tech"), "晶圆": ("半导体", "tech"), "火箭": ("商业航天", "mfg"), "卫星": ("商业航天", "mfg"), "星链": ("商业航天", "mfg"), "人形": ("机器人", "mfg"), "具身智能": ("机器人", "mfg"),
    "关税": ("宏观", "macro"), "制裁": ("宏观", "macro"), "汇率": ("宏观", "macro"), "证监会": ("政策", "macro"), "央行": ("政策", "macro"), "研报": ("研报", "macro"), "评级": ("研报", "macro"), "策略": ("研报", "macro"),
    "通胀": ("宏观", "macro"), "CPI": ("宏观", "macro"), "PPI": ("宏观", "macro"), "GDP": ("宏观", "macro"),
    "黄金": ("宏观", "macro"), "原油": ("宏观", "macro"), "天然气": ("宏观", "macro"), "期货": ("宏观", "macro"),
    "指数": ("宏观", "macro"), "成交额": ("宏观", "macro"), "北向": ("宏观", "macro"), "两市": ("宏观", "macro")
}

NOISE_WORDS = ["收盘", "开盘", "指数", "报价", "汇率", "定盘", "结算", "涨跌", "日程", "前值", "融资"]

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
    # 极速处理，不依赖庞大的字典加载
    for item in raw_list:
        resolved.append((item, item)) 
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
            if priority < 1: priority = 1
            if category == "other": category = "macro"
            break
    if sentiment != "NEU" and category == "other":
        category = "stock_event"
        if priority < 1: priority = 1
    return list(set(tags)), priority, category, sentiment

def highlight_text(text):
    text = str(text)
    text = re.sub(r'([+-]?\d+\.?\d*%)', r'<span style="color:#d946ef; font-weight:bold;">\1</span>', text)
    text = re.sub(r'(\d{6})', r'<span style="background:#e0f2fe; color:#0369a1; padding:0 4px; border-radius:3px; font-family:monospace;">\1</span>', text)
    text = re.sub(r'(\d+\.?\d*[亿万])', r'<span style="color:#d97706; font-weight:bold;">\1</span>', text)
    actions = ["增持", "买入", "中标", "签署", "获批", "立案", "调查", "突破", "首发", "启动", "减持"]
    for act in actions:
        text = text.replace(act, f'<span style="font-weight:900; color:#2d3748; background-color:#edf2f7; padding:0 2px;">{act}</span>')
    return text

# ================= 4. 数据处理 (V9.8: 极速分流 + 智能休眠) =================

def log_scan(title, status):
    current_time = datetime.now().strftime("%H:%M:%S")
    st.session_state.scan_log.insert(0, f"[{current_time}] {status}: {title[:10]}...")
    if len(st.session_state.scan_log) > 5: st.session_state.scan_log.pop()

def fetch_latest_data(portfolio_str, show_all=False, force_fetch=False):
    resolved_portfolio = resolve_portfolio(portfolio_str)
    fetched_list = []
    
    if force_fetch:
        loop_count = 50; cls_limit = 1500
        progress_bar = st.progress(0, text="🌊 正在初始化 (加载全市场名单)...")
        get_cached_stock_map() 
        time_limit = None
    else:
        loop_count = 1; cls_limit = 20; progress_bar = None
        time_limit = datetime.now() - timedelta(hours=2)

    # 1. 持仓狙击 (🔥 严格限制：只有 force_fetch 为 True 时才执行)
    if force_fetch: 
        total_stocks = len(resolved_portfolio)
        for idx, (code, name) in enumerate(resolved_portfolio):
            if not code: continue 
            if progress_bar: progress_bar.progress(int((idx / (total_stocks + 1)) * 30), text=f"🎯 正在狙击持仓: {name}...")
            try:
                df_stock_news = ak.stock_news_em(symbol=code)
                for _, row in df_stock_news.head(5).iterrows(): 
                    title = row.get('title', ''); content = row.get('content', '') or title
                    time_str = row.get('public_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    link = row.get('url', '') or row.get('Url', '') or row.get('link', '')
                    if not link: link = f"http://guba.eastmoney.com/list,{code}.html"
                    full = f"【{name}公告】{title} {content}"
                    fetched_list.append({
                        "Time": time_str, "Content": full, "Link": link, "Source": "🇨🇳 东财个股",
                        "Tags": str([f"🎯 持仓: {name}"]), "Prio": 2, "Cat": "holding", "Sent": "NEU", "RawTime": time_str, "Code": code
                    })
            except: pass
    
    # 2. 金十
    max_id = ""
    for i in range(loop_count):
        if force_fetch and progress_bar: progress_bar.progress(30 + int(i), text="🌍 扫描金十数据...")
        try:
            url = "https://flash-api.jin10.com/get_flash_list"
            params = {"channel": "-8200", "vip": "1", "max_time": max_id}
            headers = {"x-app-id": "bVBF4FyRTn5NJF5n", "x-version": "1.0.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=3)
            if resp.status_code == 200:
                data_list = resp.json().get("data", [])
                if not data_list: break
                if data_list: max_id = data_list[-1].get("id", "")
                for item in data_list:
                    data = item.get("data", {})
                    time_str = item.get("time", "")
                    if time_limit:
                        try:
                            if datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S") < time_limit: continue
                        except: pass
                    content = data.get("content", "") or ""; title = data.get("title", "") or ""
                    item_id = item.get("id")
                    link = f"https://flash.jin10.com/detail/{item_id}" if item_id else "https://www.jin10.com"
                    full = f"【{title}】 {content}" if title and title not in content else content
                    if len(full) < 5: continue
                    if not show_all and is_noise(full) and not force_fetch: continue
                    tags, prio, cat, sent = check_relevance(full, resolved_portfolio)
                    if i == 0 and prio > 0 and not force_fetch: log_scan(full, "✅")
                    if show_all or prio > 0 or force_fetch:
                        fetched_list.append({
                            "Time": time_str, "Content": full, "Link": link, "Source": "🌍 金十",
                            "Tags": str(tags), "Prio": prio, "Cat": cat, "Sent": sent, "RawTime": time_str, "Code": ""
                        })
                if force_fetch: time.sleep(0.05)
            else: break
        except: break

    # 3. 财联社
    try:
        df_cls = ak.stock_telegraph_cls(symbol="A股24小时电报")
        df_cls = df_cls.head(cls_limit)
        for _, row in df_cls.iterrows():
            time_str = str(row['publish_time'])
            if time_limit:
                try:
                    if datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S") < time_limit: continue
                except: pass
            content = row['content'] or ""; title = row['title'] or ""
            full = f"【{title}】 {content}" if title != "无" else content
            if not show_all and is_noise(full) and not force_fetch: continue
            tags, prio, cat, sent = check_relevance(full, resolved_portfolio)
            if not force_fetch and prio > 0: log_scan(full, "✅")
            if show_all or prio > 0 or force_fetch:
                fetched_list.append({
                    "Time": time_str, "Content": full, "Link": "https://www.cls.cn/telegraph", "Source": "🇨🇳 财联社",
                    "Tags": str(tags), "Prio": prio, "Cat": cat, "Sent": sent, "RawTime": time_str, "Code": ""
                })
    except: pass
    
    # 4. 东财全球
    try:
        df_em = ak.stock_info_global_em()
        limit = 100 if force_fetch else 30
        for _, row in df_em.head(limit).iterrows():
            time_str = str(row['发布时间'])
            if time_limit:
                try:
                    if datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S") < time_limit: continue
                except: pass
            content = row['content']; title = row['标题']
            link = row['原文链接']
            if not link: link = "https://kuaixun.eastmoney.com/"
            full = f"【{title}】 {content}" if title else content
            if not show_all and is_noise(full) and not force_fetch: continue
            tags, prio, cat, sent = check_relevance(full, resolved_portfolio)
            if show_all or prio > 0 or force_fetch:
                fetched_list.append({
                    "Time": time_str, "Content": full, "Link": link, "Source": "🚀 东财",
                    "Tags": str(tags), "Prio": prio, "Cat": cat, "Sent": sent, "RawTime": time_str
                })
    except: pass

    if force_fetch and progress_bar: 
        progress_bar.progress(100, text="✅ 抓取完成")
        time.sleep(0.5)
        progress_bar.empty()
        
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
    mem_df = st.session_state.news_stream
    combined = pd.concat([new_df, mem_df, disk_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['Content'], keep='first')
    combined = combined.sort_values(by='RawTime', ascending=False)
    combined.head(8000).to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
    st.session_state.news_stream = combined.head(5000)
    return len(combined)

@st.cache_data(ttl=60)
def get_realtime_sentiment():
    try:
        df = ak.stock_zh_a_spot_em()
        up_count = len(df[df['涨跌幅'] > 0])
        down_count = len(df[df['涨跌幅'] < 0])
        total = up_count + down_count + len(df[df['涨跌幅'] == 0])
        limit_up = len(df[df['涨跌幅'] > 9.5])
        limit_down = len(df[df['涨跌幅'] < -9.5])
        median_chg = df['涨跌幅'].median()
        total_amount = df['成交额'].sum() / 100000000 
        return {
            "up": up_count, "down": down_count, "total": total,
            "limit_up": limit_up, "limit_down": limit_down,
            "median": median_chg, "amount": total_amount, "status": "success"
        }
    except Exception as e:
        return {"status": "fail", "msg": str(e)}

def render_sentiment_dashboard():
    if not st.session_state.show_dashboard:
        if st.button("🌡️ 点击加载实时大盘情绪 (耗时约2秒)", type="primary", use_container_width=True):
            st.session_state.show_dashboard = True
            st.rerun()
        return

    with st.spinner("正在连接交易所行情..."):
        data = get_realtime_sentiment()
    
    if data["status"] == "fail": 
        st.warning("行情连接失败，请重试")
        return
    if data['total'] == 0: return 
    
    up_ratio = (data['up'] / data['total']) * 100
    down_ratio = (data['down'] / data['total']) * 100
    if up_ratio > 80: mood = "🔥 极度亢奋"
    elif up_ratio > 60: mood = "🔴 多头主导"
    elif up_ratio < 20: mood = "❄️ 极度冰点"
    elif up_ratio < 40: mood = "💚 空头主导"
    else: mood = "⚖️ 震荡平衡"
    
    html = f"""<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);"><div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;"><div style="font-size:18px; font-weight:bold; color:#333;">🌡️ 市场全景驾驶舱 <span style="font-size:14px; color:#666; font-weight:normal; margin-left:10px;">({mood})</span></div><div style="font-size:14px; font-weight:bold; color:#555;">成交额: <span style="color:#333;">{data['amount']:.0f} 亿</span></div></div><div style="width:100%; height:12px; background:#e2e8f0; border-radius:6px; display:flex; overflow:hidden;"><div style="width:{up_ratio}%; background:#f56565; height:100%;"></div><div style="width:{down_ratio}%; background:#48bb78; height:100%; margin-left:auto;"></div></div><div style="display:flex; justify-content:space-between; font-size:13px; margin-top:5px; color:#666;"><span style="color:#c53030; font-weight:bold;">🔴 上涨: {data['up']} 家</span><span style="color:#2f855a; font-weight:bold;">💚 下跌: {data['down']} 家</span></div><div style="display:flex; gap:15px; margin-top:15px;"><div style="flex:1; background:#fff; padding:10px; border-radius:6px; text-align:center; border:1px solid #fee2e2;"><div style="font-size:12px; color:#999;">🚀 涨停/连板</div><div style="font-size:18px; color:#c53030; font-weight:bold;">{data['limit_up']}</div></div><div style="flex:1; background:#fff; padding:10px; border-radius:6px; text-align:center; border:1px solid #f0fff4;"><div style="font-size:12px; color:#999;">📉 跌停/核按钮</div><div style="font-size:18px; color:#2f855a; font-weight:bold;">{data['limit_down']}</div></div><div style="flex:1; background:#fff; padding:10px; border-radius:6px; text-align:center; border:1px solid #edf2f7;"><div style="font-size:12px; color:#999;">📊 赚钱效应 (中位数)</div><div style="font-size:18px; color:{'#c53030' if data['median']>0 else '#2f855a'}; font-weight:bold;">{data['median']:+.2f}%</div></div></div></div>"""
    st.markdown(html, unsafe_allow_html=True)
    if st.button("❌ 收起仪表盘", type="secondary"):
        st.session_state.show_dashboard = False
        st.rerun()

def extract_smart_summary(subset_df):
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
    NOISE_TITLES = ["午评", "收盘", "早盘", "三大指数", "数据整理", "要闻汇总", "日历", "投资避雷针", "早间新闻", "昨日", "复盘", "一览"]
    report_sections = []
    for topic in topics:
        keywords = TOPIC_EXPANSION.get(topic, [topic])
        pattern = "|".join(keywords)
        mask = df['Content'].str.contains(pattern, case=False, na=False) | df['Tags'].str.contains(pattern, case=False, na=False)
        if topic not in ["政策", "全球宏观", "宏观"]:
             mask = mask & ~df['Content'].str.contains('|'.join(NOISE_TITLES), case=False)
        subset = df[mask]
        if not subset.empty:
            count = len(subset); pos_count = len(subset[subset['Sent'] == 'POS'])
            strength = "⚪ 弱"; bg_color = "#f7fafc"
            if count >= 5 or pos_count >= 2: strength = "🟢 强"; bg_color = "#f0fff4"
            elif count >= 2: strength = "🟡 中"; bg_color = "#fffff0"
            top_rows = subset.sort_values(by=['Prio', 'RawTime'], ascending=False).head(10)
            desc_list = []
            seen_content = set()
            count_valid = 0
            for i, (_, row) in enumerate(top_rows.iterrows()):
                if count_valid >= 5: break
                clean_txt = str(row['Content']).replace("【", "").replace("】", "：").strip()
                if clean_txt[:20] in seen_content: continue
                seen_content.add(clean_txt[:20])
                desc_list.append(f"{count_valid+1}. {clean_txt}")
                count_valid += 1
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
    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>情报哨兵研报 {datetime.now().strftime('%Y%m%d')}</title>
        <style>
            body {{ font-family: '微软雅黑', sans-serif; padding: 40px; background: #f4f6f9; color: #333; }}
            .container {{ max-width: 900px; margin: 0 auto; background: #fff; padding: 40px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 15px; }}
            .meta {{ color: #7f8c8d; margin-bottom: 30px; font-size: 14px; }}
            .card {{ padding: 20px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #eee; }}
            .strong {{ background: #f0fff4; border-color: #c6f6d5; }}
            .medium {{ background: #fffff0; border-color: #fefcbf; }}
            .weak {{ background: #f7fafc; border-color: #edf2f7; }}
            .header {{ display: flex; align-items: center; margin-bottom: 15px; }}
            .tag {{ padding: 4px 10px; border-radius: 4px; font-weight: bold; font-size: 14px; margin-left: 10px; background: #fff; border: 1px solid #ccc; }}
            .content {{ line-height: 1.8; color: #2c3e50; font-size: 15px; }}
            .footer {{ margin-top: 40px; text-align: center; color: #aaa; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📝 全球市场情报{report_type}概要</h1>
            <div class="meta">📅 周期: {date_range}<br>🔍 覆盖方向: {topics}</div>
    """
    for item in data:
        css_class = "weak"
        if "强" in item['Strength']: css_class = "strong"
        elif "中" in item['Strength']: css_class = "medium"
        html += f"""
        <div class="card {css_class}">
            <div class="header"><h2 style="margin:0;">{item['Topic']} 信号</h2><span class="tag">{item['Strength']}</span></div>
            <div style="font-size:12px; color:#999; margin-bottom:10px;">智能联想: {item['Keywords']}</div>
            <div class="content">{item['Desc']}</div>
            <div style="margin-top:15px; font-size:13px; color:#666; border-top:1px dashed #ccc; padding-top:10px;">🔗 <b>产业链关联：</b>{item['Sector']}</div>
        </div>"""
    html += """<div class="footer">由 情报哨兵 V9.8 系统自动生成</div></div></body></html>"""
    return html

# ================= 6. 页面布局 =================

with st.sidebar:
    st.header("☁️ 哨兵 V9.8")
    st.caption("云端/本地通用版")
    
    with st.expander("💼 持仓配置"):
        portfolio_input = st.text_area("持仓", value=st.session_state.portfolio_text)
        if st.button("💾 保存"):
            save_config(CONFIG_FILE_PORTFOLIO, portfolio_input)
            st.session_state.portfolio_text = portfolio_input
            st.success("已保存")
    
    c1, c2 = st.columns(2)
    
    if c1.button("🔄 极速刷新"):
        with st.spinner("🚀 极速同步总线..."):
            new_data = fetch_latest_data(portfolio_input, force_fetch=False) # 极速模式
            save_and_merge_data(new_data)
        st.toast("✅ 刷新完成 (秒级)", icon="⚡")
        time.sleep(0.3); st.rerun()
        
    if c2.button("⚡ 深度补全"):
        with st.spinner("🐢 深度扫描持仓公告..."):
            new_data = fetch_latest_data(portfolio_input, force_fetch=True) # 深度模式
            save_and_merge_data(new_data)
        st.success("✅ 全量补全完成")
        time.sleep(1); st.rerun()

    if st.button("📥 立即落盘 (存盘)"):
        save_and_merge_data(pd.DataFrame()) 
        st.session_state.last_save_time = time.time()
        st.success(f"已将 {len(st.session_state.news_stream)} 条数据写入硬盘")

    st.markdown("### 🧭 研报关注方向")
    report_topics_input = st.text_area("方向 (智能扩展)", value=st.session_state.report_topics, height=80)
    if st.button("💾 保存研报方向"):
        save_config(CONFIG_FILE_TOPICS, report_topics_input)
        st.session_state.report_topics = report_topics_input
        st.success("已保存")

# --- 页面主体 ---
main_container = st.container()

with main_container:
    render_sentiment_dashboard()
    
    st.info(f"📊 **情报库** | 历史库存: {len(st.session_state.news_stream)} 条 | 您的持仓: {st.session_state.portfolio_text[:20]}...")

    tabs = st.tabs(["📑 研报", "🌊 全部", "🚨 持仓", "📊 个股雷达", "🤖 科技", "🟢 制造", "🌍 宏观", "📜 复盘", "🔍 研究"])
    
    def render_simple_list(df_subset, header_icon=""):
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
        
        # 🔥 云端适配优化：不依赖 stock_map 缓存，直接用字符串匹配
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
