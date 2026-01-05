import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import time
import requests

# --- 1. KONFIGURACE STRÁNKY ---
st.set_page_config(
    page_title="AI Krypto Terminál Top 15", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBÁLNÍ KONSTANTY
HISTORY_LIMIT = 1000
# Stablecoiny a fiat, které nás nezajímají
IGNORED_COINS = [
    'USDC', 'FDUSD', 'TUSD', 'BUSD', 'DAI', 'EUR', 'GBP', 'USDP', 
    'AEUR', 'UST', 'USDD', 'GUSD', 'PAXG', 'WBTC'
]

# --- 2. SIDEBAR - NASTAVENÍ ---
st.sidebar.title("⚙️ Nastavení")

# Automatická aktualizace
st.sidebar.subheader("🕒 Automatizace")
refresh_map = {
    "Vypnuto": 0, "10 min": 10, "20 min": 20, "30 min": 30, "60 min": 60
}
selected_refresh = st.sidebar.selectbox("Interval aktualizace", list(refresh_map.keys()), index=0)
refresh_min = refresh_map[selected_refresh]

# Výběr mince (placeholder naplněný skenerem)
coin_selector = st.sidebar.empty()

# Řazení trhu
st.sidebar.divider()
sort_criteria = st.sidebar.selectbox(
    "Řadit Top 15 podle:", 
    ["Nejvyšší Objem", "Největší Růst", "Největší Pokles", "Nejvyšší Volatilita"]
)

# Nastavení AI (3 svíčky pro ultra-krátký výhled)
st.sidebar.divider()
prediction_steps = st.sidebar.slider("Výhled AI (svíček)", 3, 96, 12)
selected_tf = st.sidebar.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=1)

# Discord Notifikace
st.sidebar.divider()
st.sidebar.subheader("🔔 Discord Alert")
discord_on = st.sidebar.checkbox("Aktivovat Discord", value=False)
discord_min_profit = st.sidebar.slider("Minimální zisk pro alert (%)", 1, 20, 2)
discord_url = st.sidebar.text_input("Webhook URL", value="https://discord.com/api/webhooks/1455837807561150526/zU1-LaHiOR36zMoNjoBTG9X_SRqbClaoam0Cv9-AOtGxyOZE_YYVXhZPsjekBNSHRkx-", type="password")

# Vizualizace
st.sidebar.divider()
show_ema20 = st.sidebar.checkbox("EMA 20", value=True)
show_ema50 = st.sidebar.checkbox("EMA 50", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=False) 

# --- 3. POMOCNÉ FUNKCE ---

def format_price(price):
    """Inteligentní formátování ceny."""
    if price is None: return "0.00"
    if price < 0.0001: return f"{price:.10f}"
    elif price < 1: return f"{price:.6f}"
    elif price < 10: return f"{price:.4f}"
    else: return f"{price:.2f}"

@st.cache_data(ttl=600)
def ziskej_top_15_pary(sort_mode):
    """Získá Top 15 párů z Binance."""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = exchange.fetch_tickers()
        data = []
        for symbol, t in tickers.items():
            if symbol.endswith('/USDT'):
                base = symbol.split('/')[0]
                if base in IGNORED_COINS or 'UP' in base or 'DOWN' in base: continue
                vol = t.get('quoteVolume', 0)
                change = t.get('percentage', 0)
                high, low = t.get('high', 0), t.get('low', 0)
                volatility = ((high - low) / low * 100) if low > 0 else 0
                data.append({'symbol': symbol, 'volume': vol, 'change': change, 'volatility': volatility})
        
        df = pd.DataFrame(data)
        if "Růst" in sort_mode: df = df.sort_values(by='change', ascending=False)
        elif "Pokles" in sort_mode: df = df.sort_values(by='change', ascending=True)
        elif "Volatilita" in sort_mode: df = df.sort_values(by='volatility', ascending=False)
        else: df = df.sort_values(by='volume', ascending=False)
        return df.head(15)['symbol'].tolist()
    except:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT']

@st.cache_data(ttl=300)
def nacti_data(symbol, timeframe):
    """Data z Binance s fallbackem na KuCoin."""
    for ex_name in ['binance', 'kucoin']:
        try:
            exchange = getattr(ccxt, ex_name)({'enableRateLimit': True})
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=HISTORY_LIMIT)
            if bars:
                df = pd.DataFrame(bars, columns=['Cas', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Cas'] = pd.to_datetime(df['Cas'], unit='ms')
                return df
        except: continue
    return None

def vypocitej_indicators(df):
    """Technická analýza."""
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    # EMA
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    # BB
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['std'] = df['Close'].rolling(20).std()
    df['BB_Up'] = df['SMA_20'] + (df['std'] * 2)
    df['BB_Low'] = df['SMA_20'] - (df['std'] * 2)
    # ATR & Slope
    df['ATR'] = (df['High'] - df['Low']).rolling(10).mean()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    return df.fillna(method='bfill')

def generuj_ai_setup(df, tf, steps):
    """Model a realistická simulace."""
    np.random.seed(int(time.time()) % 1000)
    df_train = df.dropna().copy()
    if len(df_train) < 100: return pd.DataFrame(), None

    df_train['Target'] = np.log(df_train['Close'].shift(-1) / df_train['Close'])
    df_train = df_train.dropna()
    
    feats = ['Close', 'RSI', 'EMA_20', 'EMA_50', 'MACD', 'Signal']
    model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    model.fit(df_train[feats], df_train['Target'])
    
    future = []
    last_close = df['Close'].iloc[-1]
    last_time = df['Cas'].iloc[-1]
    volatility = df_train['Target'].std()

    for i in range(1, steps + 1):
        # Predikce s šumem
        pred_ret = model.predict(df.tail(1)[feats])[0]
        # Realistický šum (zvyšuje se s časem)
        noise = np.random.normal(0, volatility * (0.8 + i/steps))
        new_close = last_close * np.exp(pred_ret + noise)
        
        t_inc = {'5m':5,'15m':15,'1h':60,'4h':240,'1d':1440}.get(tf, 60)
        atr = df['ATR'].iloc[-1]
        
        future.append({
            'Cas': last_time + timedelta(minutes=i*t_inc),
            'Open': last_close,
            'High': max(last_close, new_close) + (atr * 0.2),
            'Low': min(last_close, new_close) - (atr * 0.2),
            'Close': new_close
        })
        last_close = new_close

    f_df = pd.DataFrame(future)
    # Výpočet setupu
    buy_price = f_df['Low'].min()
    tp_price = f_df.loc[f_df['Low'].idxmin():, 'High'].max()
    potencial = ((tp_price / buy_price) - 1) * 100
    
    setup = {
        'Buy': buy_price, 'TP': tp_price, 'SL': buy_price * 0.98,
        'Potencial': potencial, 'Buy_Time': f_df.loc[f_df['Low'].idxmin(), 'Cas']
    }
    return f_df, setup

# --- 4. HLAVNÍ LOGIKA ---

with st.spinner("Načítám trh..."):
    top_15 = ziskej_top_15_pary(sort_criteria)

user_choice = coin_selector.selectbox("Vyberte minci", ["-- SKENER (Top Tip) --"] + top_15)

# SKENER (S progress barem)
def skenuj_trh(seznam, tf, steps):
    bar = st.progress(0)
    best = None
    for i, coin in enumerate(seznam):
        bar.progress((i+1)/len(seznam))
        data = nacti_data(coin, tf)
        if data is not None:
            data = vypocitej_indicators(data)
            _, setup = generuj_ai_setup(data, tf, steps)
            if not best or setup['Potencial'] > best['potencial']:
                best = {'coin': coin, 'potencial': setup['Potencial'], 'setup': setup, 'data': data}
    bar.empty()
    return best

if user_choice == "-- SKENER (Top Tip) --":
    best = skenuj_trh(top_15, selected_tf, prediction_steps)
    active_coin = best['coin'] if best else "BTC/USDT"
    if best:
        st.info(f"🚀 AI doporučuje: **{active_coin}** s potenciálem **{best['potencial']:.2f}%**")
else:
    active_coin = user_choice
    best = None

# Zobrazení detailu
df = nacti_data(active_coin, selected_tf)
if df is not None:
    df = vypocitej_indicators(df)
    f_df, setup = generuj_ai_setup(df, selected_tf, prediction_steps)
    
    # Graf
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_width=[0.2, 0.8])
    fig.add_trace(go.Candlestick(x=df['Cas'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Historie", opacity=0.4), row=1, col=1)
    fig.add_trace(go.Candlestick(x=f_df['Cas'], open=f_df['Open'], high=f_df['High'], low=f_df['Low'], close=f_df['Close'], name="AI Predikce", increasing_line_color='#00ffcc', decreasing_line_color='#ff00ff'), row=1, col=1)
    
    # EMA linky
    if show_ema20: fig.add_trace(go.Scatter(x=df['Cas'], y=df['EMA_20'], name="EMA 20", line=dict(color='yellow', width=1)), row=1, col=1)
    if show_ema50: fig.add_trace(go.Scatter(x=df['Cas'], y=df['EMA_50'], name="EMA 50", line=dict(color='orange', width=1)), row=1, col=1)
    
    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=700, margin=dict(l=10,r=10,t=30,b=10))
    
    # Dashboard Karty
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("VSTUP", format_price(setup['Buy']))
    with c2: st.metric("CÍL (TP)", format_price(setup['TP']), f"{setup['Potencial']:.2f}%")
    with c3: st.metric("STOP LOSS", format_price(setup['SL']), "-2.00%", delta_color="inverse")
    with c4: st.metric("AKTUÁLNÍ CENA", format_price(df['Close'].iloc[-1]))

    st.plotly_chart(fig, use_container_width=True)

    # Discord Logika
    if discord_on and setup['Potencial'] >= discord_min_profit:
        content = f"🚀 **SIGNÁL: {active_coin}**\nVstup: {format_price(setup['Buy'])}\nCíl: {format_price(setup['TP'])}\nPotenciál: **{setup['Potencial']:.2f}%**"
        try:
            requests.post(discord_url, json={"content": content})
            st.toast("Alert odeslán!")
        except: pass

# Automatický odpočet
if refresh_min > 0:
    ph = st.sidebar.empty()
    for i in range(refresh_min * 60, 0, -1):
        mm, ss = divmod(i, 60)
        ph.info(f"🔄 Aktualizace za {mm:02d}:{ss:02d}")
        time.sleep(1)
    st.rerun()
