import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib
import numpy as np
import plotly.graph_objects as go
from youtube_scraper import search_video_id, get_comments, analyze_sentiment
import time

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smartphone AI Analyzer",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MINIMALIST CSS ---
st.markdown("""
<style>
    .stButton>button {
        border-radius: 20px;
        height: 3em;
        width: 100%;
    }
    div[data-testid="stMetric"] {
        padding: 10px;
        border-radius: 10px;
    }
    .comment-bubble {
        padding: 10px 15px;
        margin-bottom: 10px;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        font-size: 0.9em;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. SESSION STATE INIT ---
if 'sentiment_result' not in st.session_state:
    st.session_state['sentiment_result'] = None
if 'current_device' not in st.session_state:
    st.session_state['current_device'] = ""

# --- 4. LOAD RESOURCES ---
class PerformancePredictor(nn.Module):
    def __init__(self, input_dim):
        super(PerformancePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_resources():
    model = PerformancePredictor(input_dim=3)
    model.load_state_dict(torch.load("model_antutu.pth", map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    
    scaler_x = joblib.load("scaler_x.pkl")
    scaler_y = joblib.load("scaler_y.pkl")
    
    df = pd.read_csv("data/processed/training_data.csv")
    df['brand_extracted'] = df['full_name'].apply(lambda x: str(x).split()[0])
    
    return model, scaler_x, scaler_y, df

try:
    model, scaler_x, scaler_y, df = load_resources()
except FileNotFoundError:
    st.error("Model/Scaler not found! Please run `python src/train.py` first.")
    st.stop()

# --- 5. SIDEBAR (CASCADING FILTER) ---

st.sidebar.title("AI Gen")
st.sidebar.caption("Smartphone Intelligence System")

st.sidebar.divider()

# Dashboard Stats
total_phones = len(df)
avg_score = int(df['antutu_score'].mean())
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Devices", total_phones)
col_s2.metric("Avg Score", f"{avg_score // 1000}k")

st.sidebar.divider()

st.sidebar.subheader("🔎 Find Device")

# Filter Brand
unique_brands = sorted(df['brand_extracted'].unique())
brand_options = ["All Brands"] + unique_brands
selected_brand = st.sidebar.selectbox("Filter by Brand", brand_options)

# Filter Model
if selected_brand == "All Brands":
    filtered_models = sorted(df['full_name'].unique())
else:
    filtered_models = sorted(df[df['brand_extracted'] == selected_brand]['full_name'].unique())

selected_device_name = st.sidebar.selectbox("Select Model", filtered_models, key="device_select")

# Reset logic
if selected_device_name != st.session_state['current_device']:
    st.session_state['sentiment_result'] = None
    st.session_state['current_device'] = selected_device_name

st.sidebar.divider()

# --- BAGIAN INI YANG DIUBAH (SYSTEM ARCHITECTURE DESCRIPTION) ---
st.sidebar.markdown("#### 🛠️ System Architecture")
st.sidebar.markdown("""
<div style="font-size: 0.85em; color: gray;">
    <b>• Prediction Model:</b><br>
    PyTorch Artificial Neural Network (ANN)<br><br>
    <b>• Sentiment Engine:</b><br>
    Real-time YouTube NLP Analysis<br><br>
    <b>• Data Processing:</b><br>
    Scikit-Learn & Pandas Pipeline
</div>
""", unsafe_allow_html=True)


# --- 6. MAIN UI ---
device_data = df[df['full_name'] == selected_device_name].iloc[0]

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title(device_data['full_name'])
with col2:
    price_val = int(device_data['clean_price'])
    st.metric(label="Estimated Price", value=f"${price_val}" if price_val > 0 else "N/A")

st.divider()

# === SECTION 1: HARDWARE ===
st.subheader("⚡ Hardware Intelligence")

with st.container(border=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric(label="RAM", value=f"{int(device_data['clean_ram'])} GB")
    with col_b:
        st.metric(label="Storage", value=f"{int(device_data['storage'])} GB")
    with col_c:
        st.metric(label="Antutu Benchmark", value=f"{int(device_data['antutu_score']):,}")

# AI Prediction
input_data = np.array([[device_data['clean_ram'], device_data['storage'], device_data['clean_price']]])
input_scaled = scaler_x.transform(input_data)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

with torch.no_grad():
    prediction_scaled = model(input_tensor).numpy()
    prediction_real = scaler_y.inverse_transform(prediction_scaled)[0][0]

# Gauge Chart
fig = go.Figure(go.Indicator(
    mode = "gauge+number+delta",
    value = prediction_real,
    delta = {'reference': device_data['antutu_score'], 'position': "top"},
    title = {'text': "AI Predicted Performance"},
    gauge = {
        'axis': {'range': [None, 2000000]},
        'bar': {'color': "rgba(30, 136, 229, 0.8)"},
        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': prediction_real}
    }
))
fig.update_layout(
    height=300, 
    margin=dict(l=20, r=20, t=80, b=20),
    font={'family': "Arial"}
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# === SECTION 2: NETIZEN VERDICT ===
st.subheader("💬 Netizen Verdict")
st.caption("Real-time sentiment analysis from YouTube")

if st.button("🔴 Analyze YouTube Live", type="primary"):
    with st.spinner(f"Analyzing {selected_device_name}..."):
        try:
            video_id = search_video_id(selected_device_name)
            if video_id:
                comments = get_comments(video_id, limit=40)
                if comments:
                    score, pos, neg = analyze_sentiment(comments)
                    st.session_state['sentiment_result'] = {
                        'score': score, 'pos': pos, 'neg': neg,
                        'comments': comments, 'video_id': video_id
                    }
                else:
                    st.warning("Video found but comments turned off.")
            else:
                st.error("No review video found.")
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

# --- RESULT DISPLAY ---
if st.session_state['sentiment_result']:
    res = st.session_state['sentiment_result']
    
    st.markdown("#### 📺 Review Video")
    st.video(f"https://www.youtube.com/watch?v={res['video_id']}")
    
    st.markdown("#### 📊 Sentiment Stats")
    
    with st.container(border=True):
        c_stat1, c_stat2 = st.columns([1, 2])
        
        with c_stat1:
            st.metric("Netizen Score", f"{res['score']}/100")
            if res['score'] > 70: st.success("Recommendation: HIGH")
            elif res['score'] > 50: st.info("Recommendation: AVERAGE")
            else: st.error("Recommendation: LOW")
            
        with c_stat2:
            fig_bar = go.Figure(data=[
                go.Bar(name='Positive', x=['Positive'], y=[res['pos']], marker_color='#28a745'),
                go.Bar(name='Negative', x=['Negative'], y=[res['neg']], marker_color='#dc3545')
            ])
            fig_bar.update_layout(barmode='group', height=200, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
    
    st.divider()
    st.markdown(f"#### 🗣️ Discussion ({len(res['comments'])} Comments)")
    
    all_comments = res['comments']
    mid = (len(all_comments) + 1) // 2
    left_c = all_comments[:mid]
    right_c = all_comments[mid:]
    
    col_l, col_r = st.columns(2)
    
    with col_l:
        for c in left_c:
            st.markdown(f"<div class='comment-bubble'>{c}</div>", unsafe_allow_html=True)
    with col_r:
        for c in right_c:
            st.markdown(f"<div class='comment-bubble'>{c}</div>", unsafe_allow_html=True)