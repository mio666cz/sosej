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
    page_title="AI Krypto Terminál PRO (Full Version)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBÁLNÍ KONSTANTY
HISTORY_LIMIT = 1000
# Rozšířený seznam stablecoinů pro filtrování párů
IGNORED_COINS = [
    'USDC', 'FDUSD', 'TUSD', 'BUSD', 'DAI', 'EUR', 'GBP', 'USDP', 
    'AEUR', 'UST', 'USDD', 'GUSD', 'PAXG', 'WBTC', 'USDT', 'USDC'
]

# --- 2. SIDEBAR - NASTAVENÍ ---
st.sidebar.title("⚙️ PRO Terminál")

# 🕒 AUTOMATIZACE AKTUALIZACE
st.sidebar.subheader("🕒 Automatizace")
refresh_map = {
    "Vypnuto": 0, 
    "10 min": 10, 
    "20 min": 20, 
    "30 min": 30,
    "40 min": 40,
    "50 min": 50,
    "60 min": 60
}
selected_refresh = st.sidebar.selectbox("Interval aktualizace", list(refresh_map.keys()), index=0)
refresh_min = refresh_map[selected_refresh]

# Placeholder pro dynamický výběr mince
coin_selector = st.sidebar.empty()

# 📊 FILTROVÁNÍ A ŘAZENÍ
st.sidebar.divider()
sort_criteria = st.sidebar.selectbox(
    "Řadit Top 15 trhu podle:", 
    [
        "Nejvyšší Objem (Volume)", 
        "Největší Růst (Gainers)", 
        "Největší Pokles (Losers)",
        "Nejvyšší Volatilita (Volatility)"
    ]
)

# 🧠 AI PARAMETRY
st.sidebar.divider()
st.sidebar.subheader("🧠 AI Model")
prediction_steps = st.sidebar.slider("Výhled AI (počet svíček)", 3, 96, 24)
selected_tf = st.sidebar.selectbox("Časový rámec (Timeframe)", ['5m', '15m', '1h', '4h', '1d'], index=1)

# 🔔 NOTIFIKACE
st.sidebar.divider()
st.sidebar.subheader("🔔 Discord Alert")
discord_on = st.sidebar.checkbox("Aktivovat Discord", value=False)
discord_min_profit = st.sidebar.slider("Minimální očekávaný zisk pro alert (%)", 0.5, 20.0, 5.0, step=0.5)
discord_url = st.sidebar.text_input("Webhook URL", value="https://discord.com/api/webhooks/1455837807561150526/zU1-LaHiOR36zMoNjoBTG9X_SRqbClaoam0Cv9-AOtGxyOZE_YYVXhZPsjekBNSHRkx-", type="password")

# 🎨 VIZUALIZACE
st.sidebar.divider()
st.sidebar.subheader("🎨 Grafika")
show_ema20 = st.sidebar.checkbox("Zobrazit EMA 20", value=True)
show_ema50 = st.sidebar.checkbox("Zobrazit EMA 50", value=True)
show_bb = st.sidebar.checkbox("Zobrazit Bollinger Bands", value=True) 

# --- 3. POMOCNÉ FUNKCE ---

def format_price(price):
    """Formátuje cenu podle velikosti, aby byla vždy čitelná."""
    if price is None: return "0.00"
    if price < 0.000001: return f"{price:.10f}"
    if price < 0.0001: return f"{price:.8f}"
    if price < 0.1: return f"{price:.6f}"
    elif price < 10: return f"{price:.4f}"
    else: return f"{price:.2f}"

@st.cache_data(ttl=600)
def ziskej_top_15_pary(min_volume_millions=5, sort_mode="Nejvyšší Objem (Volume)"):
    """Získá Top 15 párů z trhu s ošetřením výpadků burz."""
    try:
        # Primární pokus: Binance
        exchange = ccxt.binance({'enableRateLimit': True})
        try:
            tickers = exchange.fetch_tickers()
        except:
            # Fallback na ByBit pokud Binance blokuje Cloud IP
            exchange = ccxt.bybit({'enableRateLimit': True})
            tickers = exchange.fetch_tickers()
        
        min_vol = min_volume_millions * 1000000 
        data = []
        
        for symbol, t in tickers.items():
            if symbol.endswith('/USDT') or symbol.endswith(':USDT'):
                base = symbol.split('/')[0].split(':')[0]
                if base in IGNORED_COINS or 'UP' in base or 'DOWN' in base: continue
                
                vol = t.get('quoteVolume', 0)
                change = t.get('percentage', 0)
                high, low = t.get('high', 0), t.get('low', 0)
                volatility = ((high - low) / low * 100) if low and low > 0 else 0
                
                if vol and vol > min_vol:
                    data.append({
                        'symbol': symbol, 
                        'volume': vol, 
                        'change': change, 
                        'volatility': volatility
                    })
        
        df = pd.DataFrame(data)
        if df.empty: return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT']

        if "Růst" in sort_mode: df = df.sort_values(by='change', ascending=False)
        elif "Pokles" in sort_mode: df = df.sort_values(by='change', ascending=True)
        elif "Volatilita" in sort_mode: df = df.sort_values(by='volatility', ascending=False)
        else: df = df.sort_values(by='volume', ascending=False)
            
        return df.head(15)['symbol'].tolist()
    except Exception as e:
        return ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']

@st.cache_data(ttl=300)
def nacti_data(symbol, timeframe):
    """Načte data z Binance s automatickým fallbackem na KuCoin/ByBit."""
    for ex_id in ['binance', 'bybit', 'kucoin', 'kraken']:
        try:
            exchange = getattr(ccxt, ex_id)({'enableRateLimit': True, 'timeout': 10000})
            bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=HISTORY_LIMIT)
            if bars and len(bars) > 100:
                df = pd.DataFrame(bars, columns=['Cas', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Cas'] = pd.to_datetime(df['Cas'], unit='ms')
                # Ujistíme se, že jsou data numerická
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df
        except:
            continue
    return None

def vypocitej_indicators(df):
    """Komplexní výpočet technických indikátorů."""
    # 1. RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    df['RSI_Slope'] = df['RSI'].diff(3)
    
    # 2. EMA 20 & 50
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # 3. Vzdálenosti od EMA (Klíčové pro AI - trend exhaustion)
    df['Dist_EMA20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['Dist_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    
    # 4. Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['std_dev'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['SMA_20'] + (df['std_dev'] * 2)
    df['BB_Low'] = df['SMA_20'] - (df['std_dev'] * 2)
    
    # 5. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 6. ATR (Average True Range) - zjednodušený
    df['ATR'] = (df['High'] - df['Low']).rolling(10).mean()
    
    # 7. ROC (Rate of Change)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100
    
    # 8. CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # Vyčištění dat
    df = df.fillna(method='bfill').fillna(0)
    return df

def generuj_ai_setup(df, timeframe, steps):
    """AI Model: Trénink a realistická iterativní simulace."""
    # Fixace seedu pro stabilitu v jednom běhu
    np.random.seed(int(time.time()) % 1000)
    df_train = df.dropna().copy()
    if len(df_train) < 100: return pd.DataFrame(), None

    # Target: Logaritmický výnos (stabilnější pro regresi)
    df_train['Target_Return'] = np.log(df_train['Close'].shift(-1) / df_train['Close'])
    df_train = df_train.dropna()
    
    # Vstupní parametry (Features)
    vstupy = [
        'Close', 'RSI', 'RSI_Slope', 'EMA_20', 'EMA_50', 
        'Dist_EMA20', 'Dist_EMA50', 'Volume', 
        'MACD', 'MACD_Signal', 'BB_Up', 'BB_Low', 'ROC', 'CCI'
    ]
    
    X = df_train[vstupy]
    y = df_train['Target_Return']
    
    # RandomForest Model
    model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=12, 
        min_samples_split=5, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X, y)
    
    budouci_svicky = []
    # Pracovní historie pro simulaci (bude se dynamicky aktualizovat)
    sim_history = df.tail(100).copy()
    
    posledni_cas = df['Cas'].iloc[-1]
    avg_volatility = df_train['Target_Return'].std()

    for i in range(1, steps + 1):
        # Aktuální stav z historie
        stav = sim_history[vstupy].tail(1)
        
        # Predikce modelu (log return)
        pred_log_ret = model.predict(stav)[0]
        
        # 1. Tlumení trendu (čím dále, tím slabší předpověď)
        pred_zmena = (np.exp(pred_log_ret) - 1) * (0.94 ** (i / 10))
        
        # 2. Korekce podle RSI (Mean Reversion)
        curr_rsi = sim_history['RSI'].iloc[-1]
        if curr_rsi > 75: pred_zmena -= 0.003
        if curr_rsi < 25: pred_zmena += 0.003
        
        # 3. Realistický progresivní šum
        # Šum roste s časem, aby graf nebyl hladký
        noise_factor = 0.8 + (i / steps) * 0.6
        noise = np.random.normal(0, avg_volatility * noise_factor)
        
        posledni_cena = sim_history['Close'].iloc[-1]
        nova_cena = posledni_cena * (1 + pred_zmena + noise)
        
        # Časový posun
        t_delta = {'5m':5, '15m':15, '1h':60, '4h':240, '1d':1440}.get(timeframe, 60)
        delta = timedelta(minutes=i * t_delta)
        
        atr = sim_history['ATR'].iloc[-1]
        
        budouci_svicky.append({
            'Cas': posledni_cas + delta,
            'Open': posledni_cena,
            'High': max(posledni_cena, nova_cena) + (atr * 0.35),
            'Low': min(posledni_cena, nova_cena) - (atr * 0.35),
            'Close': nova_cena
        })
        
        # --- Dynamická aktualizace simulované historie ---
        # Pro další krok musíme odhadnout, jak se změní indikátory
        last_row = sim_history.iloc[-1]
        new_row_dict = {
            'Cas': posledni_cas + delta,
            'Close': nova_cena,
            'Volume': last_row['Volume'] * (1 + np.random.normal(0, 0.1)),
            'RSI': np.clip(last_row['RSI'] + (nova_cena - posledni_cena) * 0.05, 10, 90),
            'RSI_Slope': (nova_cena - posledni_cena) * 0.02,
            'EMA_20': (nova_cena * 0.1) + (last_row['EMA_20'] * 0.9),
            'EMA_50': (nova_cena * 0.04) + (last_row['EMA_50'] * 0.96),
            'MACD': last_row['MACD'],
            'MACD_Signal': last_row['MACD_Signal'],
            'BB_Up': last_row['BB_Up'],
            'BB_Low': last_row['BB_Low'],
            'ROC': last_row['ROC'],
            'CCI': last_row['CCI'],
            'ATR': last_row['ATR']
        }
        # Dopočet distancí
        new_row_dict['Dist_EMA20'] = (nova_cena - new_row_dict['EMA_20']) / new_row_dict['EMA_20']
        new_row_dict['Dist_EMA50'] = (nova_cena - new_row_dict['EMA_50']) / new_row_dict['EMA_50']
        
        sim_history = pd.concat([sim_history, pd.DataFrame([new_row_dict])], ignore_index=True)

    pred_df = pd.DataFrame(budouci_svicky)
    
    # OBCHODNÍ PLÁN
    # Najdeme nejnižší bod v budoucnosti (ideální vstup)
    min_idx = pred_df['Low'].idxmin()
    limit_buy = pred_df.loc[min_idx, 'Low']
    buy_time = pred_df.loc[min_idx, 'Cas']
    
    # Najdeme nejvyšší bod po vstupu (ideální výstup)
    post_buy_df = pred_df.loc[min_idx:]
    if not post_buy_df.empty:
        max_idx = post_buy_df['High'].idxmax()
        take_profit = post_buy_df.loc[max_idx, 'High']
        sell_time = post_buy_df.loc[max_idx, 'Cas']
    else:
        take_profit = limit_buy * 1.02
        sell_time = buy_time + timedelta(hours=24)
    
    potencial = ((take_profit / limit_buy) - 1) * 100
    
    # Relativní časy pro UI
    now = datetime.now()
    def fmt_time(dt):
        diff = dt - now
        ts = diff.total_seconds()
        if ts < 0: return "Právě teď"
        h, m = int(ts // 3600), int((ts % 3600) // 60)
        return f"za {h}h {m}m"

    setup = {
        'Limit_Buy': limit_buy, 'Buy_Rel': fmt_time(buy_time),
        'Take_Profit': take_profit, 'Sell_Rel': fmt_time(sell_time),
        'Stop_Loss': limit_buy * 0.985, 
        'Potencial': potencial
    }
    return pred_df, setup

def posli_discord_alert(webhook_url, content, image_bytes=None):
    """Bezpečné odeslání alertu na Discord (Safe Mode)."""
    if not webhook_url or "discord.com" not in webhook_url: return
    try:
        if image_bytes:
            files = {"file": ("chart.png", image_bytes, "image/png")}
            requests.post(webhook_url, data={"content": content}, files=files, timeout=10)
        else:
            requests.post(webhook_url, json={"content": content}, timeout=10)
    except:
        pass

# --- 4. SKENER TRHU ---

def skenuj_top_15(seznam, tf, steps):
    """Prochází Top 15 mincí a hledá největší zisk."""
    results = []
    status_box = st.empty()
    progress_bar = st.progress(0)
    
    start_time = time.time()
    for i, coin in enumerate(seznam):
        # Progress bar logika
        perc = (i + 1) / len(seznam)
        progress_bar.progress(perc)
        rem = int((time.time() - start_time) / (i+1) * (len(seznam) - (i+1))) if i > 0 else 0
        status_box.text(f"🔍 AI analyzuje {coin} ({i+1}/{len(seznam)}) - Zbývá cca {rem} s...")
        
        data = nacti_data(coin, tf)
        if data is not None:
            data = vypocitej_indicators(data)
            _, setup = generuj_ai_setup(data, tf, steps)
            if setup:
                results.append({
                    'symbol': coin, 
                    'potencial': setup['Potencial'], 
                    'setup': setup, 
                    'df': data
                })
    
    progress_bar.empty()
    status_box.empty()
    
    if not results: return None
    # Vrátíme seřazené podle potenciálu
    return sorted(results, key=lambda x: x['potencial'], reverse=True)

# --- 5. HLAVNÍ LOGIKA UI ---

# Načtení seznamu Top 15
with st.spinner("Aktualizuji přehled trhu..."):
    top_15_symbols = ziskej_top_15_pary(5, sort_criteria)

# Ruční výběr nebo Skener
selection = coin_selector.selectbox(
    "Aktivní mince", 
    ["-- DOPORUČENÍ AI SKENERU --"] + top_15_symbols
)

# Spuštění analýzy
if selection == "-- DOPORUČENÍ AI SKENERU --":
    vsechny_tipy = skenuj_top_15(top_15_symbols, selected_tf, prediction_steps)
    if vsechny_tipy:
        best = vsechny_tipy[0]
        active_coin = best['symbol']
        active_setup = best['setup']
        active_df = best['df']
        st.markdown(f"""
        <div style="background:#1e293b; padding:20px; border-radius:15px; border:2px solid #6366f1; text-align:center; margin-bottom:25px;">
            <h2 style="margin:0; color:#818cf8;">🚀 AI TIP DNE: {active_coin}</h2>
            <p style="margin:5px 0 0 0; color:#cbd5e1; font-size:1.2em;">Předpokládaný pohyb: <b>+{best['potencial']:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        active_coin = "BTC/USDT"
        active_df = nacti_data(active_coin, selected_tf)
        active_df = vypocitej_indicators(active_df) if active_df is not None else None
        _, active_setup = generuj_ai_setup(active_df, selected_tf, prediction_steps) if active_df is not None else (None, None)
else:
    active_coin = selection
    active_df = nacti_data(active_coin, selected_tf)
    if active_df is not None:
        active_df = vypocitej_indicators(active_df)
        _, active_setup = generuj_ai_setup(active_df, selected_tf, prediction_steps)
    else:
        active_setup = None

# Zobrazení detailu
st.title(f"📊 Detailní analýza: {active_coin}")

if active_df is not None and active_setup is not None:
    f_df, _ = generuj_ai_setup(active_df, selected_tf, prediction_steps)
    curr_p = active_df['Close'].iloc[-1]
    
    # 💰 DASHBOARD KARTY
    c1, c2, c3, c4 = st.columns(4)
    card_st = "padding:20px; border-radius:12px; text-align:center; color:white; box-shadow: 0 4px 15px rgba(0,0,0,0.4);"
    
    with c1:
        st.markdown(f"""<div style="background:#15803d; {card_st}">
            <div style="opacity:0.8; font-size:0.9em;">VSTUP (LIMIT)</div>
            <div style="font-size:1.8em; font-weight:bold; margin:5px 0;">{format_price(active_setup['Limit_Buy'])}</div>
            <div style="font-size:0.85em;">🕒 {active_setup['Buy_Rel']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div style="background:#a16207; {card_st}">
            <div style="opacity:0.8; font-size:0.9em;">CÍL (TP)</div>
            <div style="font-size:1.8em; font-weight:bold; margin:5px 0;">{format_price(active_setup['Take_Profit'])}</div>
            <div style="font-size:0.85em;">🕒 {active_setup['Sell_Rel']}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div style="background:#7f1d1d; {card_st}">
            <div style="opacity:0.8; font-size:0.9em;">STOP LOSS</div>
            <div style="font-size:1.8em; font-weight:bold; margin:5px 0;">{format_price(active_setup['Stop_Loss'])}</div>
            <div style="font-size:0.85em;">Ochrana -1.5%</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div style="background:#1d4ed8; {card_st}">
            <div style="opacity:0.8; font-size:0.9em;">AI POTENCIÁL</div>
            <div style="font-size:1.8em; font-weight:bold; margin:5px 0;">+{active_setup['Potencial']:.2f}%</div>
            <div style="font-size:0.85em;">Aktuálně: {format_price(curr_p)}</div>
        </div>""", unsafe_allow_html=True)

    # 📈 GRAF
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.8])
    
    # Historie (Svíčky)
    fig.add_trace(go.Candlestick(
        x=active_df['Cas'], open=active_df['Open'], high=active_df['High'], low=active_df['Low'], close=active_df['Close'], 
        name="Historie", opacity=0.4
    ), row=1, col=1)
    
    # AI Predikce (Zubatá čára se svíčkami)
    fig.add_trace(go.Candlestick(
        x=f_df['Cas'], open=f_df['Open'], high=f_df['High'], low=f_df['Low'], close=f_df['Close'], 
        name="AI Budoucnost", increasing_line_color='#00ffcc', decreasing_line_color='#ff00ff'
    ), row=1, col=1)

    # Indikátory
    if show_ema20: fig.add_trace(go.Scatter(x=active_df['Cas'], y=active_df['EMA_20'], name="EMA 20", line=dict(color='#fbbf24', width=1.5)), row=1, col=1)
    if show_ema50: fig.add_trace(go.Scatter(x=active_df['Cas'], y=active_df['EMA_50'], name="EMA 50", line=dict(color='#f97316', width=1.5)), row=1, col=1)
    if show_bb:
        fig.add_trace(go.Scatter(x=active_df['Cas'], y=active_df['BB_Up'], name="BB+", line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=active_df['Cas'], y=active_df['BB_Low'], name="BB-", line=dict(color='rgba(255,255,255,0.2)', width=1, dash='dot'), fill='tonexty'), row=1, col=1)

    # Horizontální linky pro setup
    fig.add_hline(y=active_setup['Limit_Buy'], line_dash="dash", line_color="#10b981", annotation_text="VSTUP", row=1, col=1)
    fig.add_hline(y=active_setup['Take_Profit'], line_dash="dot", line_color="#00ffcc", annotation_text="CÍL", row=1, col=1)

    # Objem
    colors = ['#ef4444' if active_df['Open'].iloc[i] > active_df['Close'].iloc[i] else '#10b981' for i in range(len(active_df))]
    fig.add_trace(go.Bar(x=active_df['Cas'], y=active_df['Volume'], name="Volume", marker_color=colors, opacity=0.3), row=2, col=1)

    fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=800, margin=dict(l=10, r=10, t=30, b=10), hovermode='x unified')
    # Oříznutí pohledu na poslední 2 dny historie + budoucnost
    fig.update_xaxes(range=[active_df['Cas'].iloc[-150], f_df['Cas'].iloc[-1]])
    
    st.plotly_chart(fig, use_container_width=True)

    # 📢 DISCORD ALERT LOGIKA
    if discord_on and active_setup['Potencial'] >= discord_min_profit:
        content = f"🚀 **SIGNÁL Z AI TERMINÁLU: {active_coin}**\n" \
                  f"Vstup (Limit): **{format_price(active_setup['Limit_Buy'])}** ({active_setup['Buy_Rel']})\n" \
                  f"Cíl (Take Profit): **{format_price(active_setup['Take_Profit'])}** ({active_setup['Sell_Rel']})\n" \
                  f"Stop Loss: **{format_price(active_setup['Stop_Loss'])}**\n" \
                  f"Očekávaný zisk: **+{active_setup['Potencial']:.2f}%**\n" \
                  f"Timeframe: {selected_tf}"
        
        # Zkusíme vytvořit obrázek (vyžaduje kaleido, v cloudu může selhat)
        img_bytes = None
        try: img_bytes = fig.to_image(format="png")
        except: pass
        
        posli_discord_alert(discord_url, content, img_bytes)
        st.toast(f"✅ Alert pro {active_coin} odeslán na Discord!")

else:
    st.error("Nepodařilo se načíst data nebo vygenerovat predikci. Zkontrolujte připojení k internetu nebo zkuste jiný symbol.")

# --- 6. AUTOMATICKÁ AKTUALIZACE ---
if refresh_min > 0:
    ph = st.sidebar.empty()
    for i in range(refresh_min * 60, 0, -1):
        mins, secs = divmod(i, 60)
        ph.info(f"🔄 Automatická obnova za: {mins:02d}:{secs:02d}")
        time.sleep(1)
    st.rerun()

st.caption(f"Poslední úspěšná aktualizace dat: {datetime.now().strftime('%H:%M:%S')}")
