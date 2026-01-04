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

# --- 1. KONFIGURACE STRÁNKY (Musí být první) ---
st.set_page_config(
    page_title="AI Krypto Terminál Top 15", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# GLOBÁLNÍ KONSTANTY
HISTORY_LIMIT = 1000
# Rozšířený seznam stablecoinů pro filtrování párů (např. USDC/USDT)
IGNORED_COINS = [
    'USDC', 'FDUSD', 'TUSD', 'BUSD', 'DAI', 'EUR', 'GBP', 'USDP', 
    'AEUR', 'UST', 'USDD', 'GUSD', 'PAXG', 'WBTC'
]

# --- 2. SIDEBAR - NASTAVENÍ ---
st.sidebar.title("⚙️ Nastavení")

# Časy aktualizace (10, 20, 30, 40, 50, 60)
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

# Placeholder pro výběr mince
coin_selector = st.sidebar.empty()

# --- VÝBĚR ŘAZENÍ (4 položky) ---
st.sidebar.divider()
# Rozšířený výběr řazení (4 možnosti včetně Volatility)
sort_criteria = st.sidebar.selectbox(
    "Řadit výsledky podle:", 
    [
        "Nejvyšší Objem (Volume)", 
        "Největší Růst (Gainers)", 
        "Největší Pokles (Losers)",
        "Nejvyšší Volatilita (Volatility)"
    ]
)

# Nastavení AI (Počátek 24 svíček)
st.sidebar.divider()
# Změněno: Minimum sníženo na 3 pro ultra-krátkodobou přesnost
prediction_steps = st.sidebar.slider("Výhled AI (svíček)", 3, 96, 12)
# Změněno: Přidán '5m' timeframe pro precizní krátkodobé obchody
selected_tf = st.sidebar.selectbox("Timeframe", ['5m', '15m', '1h', '4h', '1d'], index=1)

# Notifikace
st.sidebar.divider()
st.sidebar.subheader("🔔 Discord Alert")
discord_on = st.sidebar.checkbox("Aktivovat Discord", value=False)
# NOVÉ: Možnost nastavit hranici pro alert
discord_min_profit = st.sidebar.slider("Minimální zisk pro alert (%)", 1, 20, 10)
discord_url = st.sidebar.text_input("Webhook URL", value="https://discord.com/api/webhooks/1455837807561150526/zU1-LaHiOR36zMoNjoBTG9X_SRqbClaoam0Cv9-AOtGxyOZE_YYVXhZPsjekBNSHRkx-", placeholder="https://...", type="password")

# Zobrazení
st.sidebar.divider()
show_ema20 = st.sidebar.checkbox("EMA 20", value=True)
show_ema50 = st.sidebar.checkbox("EMA 50", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands", value=False) # Možnost zobrazit BB

# --- 3. POMOCNÉ FUNKCE ---

def format_price(price):
    """Formátuje cenu podle její velikosti, aby nebyla 0.0000."""
    if price is None:
        return "0.00"
    if price < 0.001:
        return f"{price:.8f}"
    elif price < 1:
        return f"{price:.5f}"
    elif price < 10:
        return f"{price:.4f}"
    else:
        return f"{price:.2f}"

@st.cache_data(ttl=600)
def ziskej_top_15_pary(min_volume_millions=10, sort_mode="Nejvyšší Objem (Volume)"):
    """Stáhne a vybere Top 15 párů podle kritéria (Objem/Změna/Volatilita)."""
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        tickers = exchange.fetch_tickers()
        
        # Pevný limit 10 mil. pro základní likviditu, i když uživatel nefiltruje
        min_vol = min_volume_millions * 1000000 
        data = []
        
        for symbol, t in tickers.items():
            if symbol.endswith('/USDT'):
                base = symbol.split('/')[0]
                # Filtrace stablecoinů a pákových tokenů (UP/DOWN)
                if base in IGNORED_COINS or 'UP' in base or 'DOWN' in base: continue
                
                vol = t.get('quoteVolume', 0)
                change = t.get('percentage', 0)
                
                # Výpočet 24h volatility (High - Low) / Low
                high = t.get('high', 0)
                low = t.get('low', 0)
                volatility = 0
                if low and low > 0:
                    volatility = ((high - low) / low) * 100
                
                if vol is None or vol < min_vol: continue
                
                data.append({
                    'symbol': symbol, 
                    'volume': vol, 
                    'change': change,
                    'volatility': volatility
                })
        
        df = pd.DataFrame(data)
        if df.empty: return ['BTC/USDT', 'ETH/USDT']

        # Řazení podle výběru uživatele
        if "Růst" in sort_mode:
            df = df.sort_values(by='change', ascending=False)
        elif "Pokles" in sort_mode:
            df = df.sort_values(by='change', ascending=True)
        elif "Volatilita" in sort_mode:
            df = df.sort_values(by='volatility', ascending=False)
        else: # Default: Volume
            df = df.sort_values(by='volume', ascending=False)
            
        return df.head(15)['symbol'].tolist()
    except Exception as e:
        return ['BTC/USDT', 'ETH/USDT']

@st.cache_data(ttl=300)
def nacti_data(symbol, timeframe):
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=HISTORY_LIMIT)
        if not bars: return None
        df = pd.DataFrame(bars, columns=['Cas', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Cas'] = pd.to_datetime(df['Cas'], unit='ms')
        # Ujistíme se, že data jsou čísla (float)
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[cols] = df[cols].astype(float)
        return df
    except:
        return None

def vypocitej_indicators(df):
    """Vypočítá rozšířenou sadu technických indikátorů se zaměřením na přesnost."""
    # 1. Základní RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    
    # 2. EMA
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # NOVÉ: Vzdálenost od EMA (klíčové pro mean-reversion strategie)
    df['Dist_EMA20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    df['Dist_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    
    # 3. ATR (Average True Range) - zjednodušený
    df['ATR_Sim'] = (df['High'] - df['Low']).rolling(10).mean()

    # 4. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 5. Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['std_dev'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (df['std_dev'] * 2)
    df['BB_Lower'] = df['SMA_20'] - (df['std_dev'] * 2)
    
    # NOVÉ: RSI Slope (Rychlost změny RSI - detekce momenta)
    df['RSI_Slope'] = df['RSI'].diff(3) # Změna za poslední 3 svíčky

    # 6. ROC
    df['ROC'] = df['Close'].pct_change(periods=12) * 100

    # 7. CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['CCI'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # Vyčištění NaN
    df = df.fillna(method='bfill').fillna(0)
    return df

def generuj_ai_setup(df, timeframe, steps):
    """Vylepšená AI Simulace s pokročilými vstupy."""
    np.random.seed(42)
    df_train = df.dropna().copy()
    if len(df_train) < 100: return pd.DataFrame(), None

    # Target: Logaritmický výnos (stabilnější pro ML než procenta)
    df_train['Target_Return'] = np.log(df_train['Close'].shift(-1) / df_train['Close'])
    df_train = df_train.dropna()
    
    # Vylepšené vstupy pro AI (včetně Distancí a Slope)
    vstupy = [
        'Close', 'RSI', 'RSI_Slope', 
        'Dist_EMA20', 'Dist_EMA50', 'Volume', 
        'MACD', 'MACD_Signal', 
        'BB_Upper', 'BB_Lower', 'ROC', 'CCI'
    ]
    
    X = df_train[vstupy]
    y = df_train['Target_Return']
    
    # Trénink silnějšího modelu
    model = RandomForestRegressor(
        n_estimators=100,      # Více stromů pro stabilitu
        max_depth=10,          # Hlubší stromy pro jemnější vzory
        min_samples_split=5,   # Prevence overfittingu na šumu
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X, y)
    
    budouci_svicky = []
    # Historie pro iterativní simulaci (včetně pomocných sloupců pro výpočet indikátorů)
    sim_cols = vstupy + ['ATR_Sim', 'SMA_20', 'std_dev', 'EMA_20', 'EMA_50']
    # Ujistíme se, že máme v sim_history všechny sloupce, i ty co nejsou ve vstupech modelu
    sim_history = df.tail(50).copy()
    
    posledni_cas = df['Cas'].iloc[-1]
    avg_vol = df_train['Target_Return'].std()

    for i in range(1, steps + 1):
        # Příprava vstupu pro model (pouze sloupce z 'vstupy')
        stav = sim_history[vstupy].tail(1)
        
        # Predikce (log return -> konverze na normal return)
        pred_log_ret = model.predict(stav)[0]
        
        # Tlumení predikce v čase (nejistota roste s časem)
        pred_zmena = (np.exp(pred_log_ret) - 1) * (0.95 ** (i / 10))
        pred_zmena = np.clip(pred_zmena, -0.02, 0.02) # Hard limit na jednu svíčku
        
        # Jemná korekce podle RSI (Mean Reversion)
        curr_rsi = sim_history['RSI'].iloc[-1]
        if curr_rsi > 80: pred_zmena -= 0.002
        if curr_rsi < 20: pred_zmena += 0.002
        
        # Realističtější šum: Kombinujeme fixní volatilitu (aby byly vidět výkyvy i na začátku)
        # a rostoucí nejistotu do budoucna.
        # Faktor 0.8 * avg_vol zajistí, že šum bude dost silný na vytvoření "červených svíček"
        noise_scale = 0.8 + (i / steps) * 0.4 
        noise = np.random.normal(0, avg_vol * noise_scale)
        
        posledni_cena = sim_history['Close'].iloc[-1]
        nova_cena = posledni_cena * (1 + pred_zmena + noise)
        
        # Časování (Přidáno 5m)
        if timeframe == '5m': delta = timedelta(minutes=i*5)
        elif timeframe == '15m': delta = timedelta(minutes=i*15)
        elif timeframe == '1h': delta = timedelta(hours=i)
        elif timeframe == '4h': delta = timedelta(hours=i*4)
        else: delta = timedelta(days=i)
        
        atr = sim_history['ATR_Sim'].iloc[-1]
        
        # Přidání svíčky do predikce
        budouci_svicky.append({
            'Cas': posledni_cas + delta,
            'Open': posledni_cena,
            'High': max(posledni_cena, nova_cena) + (atr * 0.4),
            'Low': min(posledni_cena, nova_cena) - (atr * 0.4),
            'Close': nova_cena
        })
        
        # --- Update Indikátorů pro další krok (Dynamická simulace) ---
        last_row = sim_history.iloc[-1]
        
        # EMA
        new_ema20 = (nova_cena * (2/21)) + (last_row['EMA_20'] * (1 - (2/21)))
        new_ema50 = (nova_cena * (2/51)) + (last_row['EMA_50'] * (1 - (2/51)))
        
        # Distances (NOVÉ - nutné aktualizovat pro další krok)
        new_dist20 = (nova_cena - new_ema20) / new_ema20
        new_dist50 = (nova_cena - new_ema50) / new_ema50
        
        # MACD
        new_ema12 = (nova_cena * (2/13)) + (last_row['Close'] * (1 - (2/13))) 
        new_ema26 = (nova_cena * (2/27)) + (last_row['Close'] * (1 - (2/27))) 
        new_macd = new_ema12 - new_ema26
        new_macd_sig = (new_macd * (2/10)) + (last_row['MACD_Signal'] * (1 - (2/10)))
        
        # RSI & RSI Slope
        delta_price = nova_cena - posledni_cena
        # Zjednodušený update RSI (plný výpočet je v loopu náročný)
        new_rsi = np.clip(last_row['RSI'] + (delta_price * 0.5 * (100/last_row['Close'])), 10, 90)
        new_rsi_slope = new_rsi - sim_history['RSI'].iloc[-3] if len(sim_history) >= 3 else 0
        
        # BB
        new_sma20 = ((last_row['SMA_20'] * 19) + nova_cena) / 20
        new_std = last_row['std_dev'] 
        new_bb_upper = new_sma20 + (new_std * 2)
        new_bb_lower = new_sma20 - (new_std * 2)
        
        new_row_dict = {
            'Close': nova_cena, 
            'Volume': last_row['Volume'],
            'RSI': new_rsi,
            'RSI_Slope': new_rsi_slope,
            'EMA_20': new_ema20,
            'EMA_50': new_ema50,
            'Dist_EMA20': new_dist20,
            'Dist_EMA50': new_dist50,
            'MACD': new_macd,
            'MACD_Signal': new_macd_sig,
            'BB_Upper': new_bb_upper,
            'BB_Lower': new_bb_lower,
            'SMA_20': new_sma20,
            'std_dev': new_std,
            'ROC': last_row['ROC'], 
            'CCI': last_row['CCI'], 
            'ATR_Sim': atr
        }
        
        # Převod na DataFrame a spojení
        sim_history = pd.concat([sim_history, pd.DataFrame([new_row_dict])], ignore_index=True)

    pred_df = pd.DataFrame(budouci_svicky)
    
    # Logika pro obchodní setup
    min_idx = pred_df['Low'].idxmin()
    limit_buy = pred_df.loc[min_idx, 'Low']
    buy_time = pred_df.loc[min_idx, 'Cas']
    
    post_buy_df = pred_df.loc[min_idx:]
    if not post_buy_df.empty:
        max_idx = post_buy_df['High'].idxmax()
        take_profit = post_buy_df.loc[max_idx, 'High']
        sell_time = post_buy_df.loc[max_idx, 'Cas']
    else:
        take_profit = limit_buy * 1.01
        sell_time = buy_time + timedelta(hours=24)
    
    potencial = ((take_profit / limit_buy) - 1) * 100
    
    now = datetime.now()
    def fmt_time(dt):
        diff = dt - now
        total_sec = diff.total_seconds()
        if total_sec < 0: return "Právě teď"
        h = int(total_sec // 3600)
        m = int((total_sec % 3600) // 60)
        return f"za {h}h {m}m"

    setup = {
        'Limit_Buy': limit_buy, 
        'Buy_Time': buy_time, 'Buy_Rel': fmt_time(buy_time),
        'Take_Profit': take_profit, 
        'Sell_Time': sell_time, 'Sell_Rel': fmt_time(sell_time),
        'Stop_Loss': limit_buy * 0.98, 
        'Potencial': potencial
    }
    return pred_df, setup

def posli_discord_alert(webhook_url, zprava, image_bytes=None):
    """Odesílá zprávu a případně obrázek na Discord."""
    if webhook_url and "discord.com" in webhook_url:
        try:
            if image_bytes:
                # Odeslání s obrázkem (multipart)
                files = {
                    "file": ("chart.png", image_bytes, "image/png")
                }
                data = {
                    "content": zprava
                }
                requests.post(webhook_url, data=data, files=files)
            else:
                # Odeslání pouze textu
                requests.post(webhook_url, json={"content": zprava})
            return True
        except: return False
    return False

# Bez cache pro progress bar (aby se vždy zobrazil průběh)
def skenuj_top_15(seznam, tf, steps):
    """Skenuje Top 15 a hledá nejlepší zisk."""
    results = []
    # Progress bar v hlavním okně
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    start_time = time.time()
    total = len(seznam)
    
    for i, coin in enumerate(seznam):
        # Výpočet času
        elapsed = time.time() - start_time
        if i > 0:
            avg_time = elapsed / i
            remaining = int(avg_time * (total - i))
            time_msg = f"Zbývá cca {remaining} s"
        else:
            time_msg = "Výpočet..."

        status_text.text(f"Skenuji {coin} ({i+1}/{total})... {time_msg}")
        progress_bar.progress((i + 1) / total)
        
        data = nacti_data(coin, tf)
        if data is not None:
            df = vypocitej_indicators(data)
            _, setup = generuj_ai_setup(df, tf, steps)
            if setup:
                results.append({'coin': coin, 'potencial': setup['Potencial'], 'setup': setup})
    
    progress_bar.empty()
    status_text.empty()
    
    if not results: return None
    return sorted(results, key=lambda x: x['potencial'], reverse=True)[0]

# --- 4. HLAVNÍ LOGIKA APLIKACE ---

# Načtení seznamu Top 15 (s fixním limitem objemu 10M, bez slideru)
with st.spinner("Aktualizuji seznam Top 15..."):
    # Posíláme default 10 pro objem, protože slider byl odstraněn
    top_15 = ziskej_top_15_pary(10, sort_criteria)

# Vložení seznamu do Sidebaru
user_choice = coin_selector.selectbox("Ruční výběr (Top 15)", ["-- Doporučení skeneru --"] + top_15)

# Spuštění skeneru
best_opp = skenuj_top_15(top_15, selected_tf, prediction_steps)

# Určení aktivní mince
if user_choice == "-- Doporučení skeneru --":
    if best_opp:
        active_coin = best_opp['coin']
        st.markdown(f"""
        <div style="background-color:#1e293b; padding:15px; border-radius:10px; border: 2px solid #6366f1; margin-bottom:20px; text-align:center">
            <h3 style="margin:0; color:#818cf8">💎 TIP SKENERU: {active_coin}</h3>
            <p style="margin:0; color:#cbd5e1">Očekávaný zisk: <b>+{best_opp['potencial']:.2f}%</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        active_coin = 'BTC/USDT'
else:
    active_coin = user_choice

st.title(f"📊 Analýza: {active_coin}")

# Detailní analýza a vykreslení
with st.spinner(f"Počítám detailní plán pro {active_coin}..."):
    df = nacti_data(active_coin, selected_tf)
    
    if df is not None:
        df = vypocitej_indicators(df)
        df_future, setup = generuj_ai_setup(df, selected_tf, prediction_steps)
        
        if not df_future.empty:
            curr_price = df['Close'].iloc[-1]
            
            # 1. Nejprve vytvoříme graf (potřebujeme ho pro obrázek)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_width=[0.2, 0.8])
            
            fig.add_trace(go.Candlestick(
                x=df['Cas'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
                name="Historie", opacity=0.4
            ), row=1, col=1)
            
            fig.add_trace(go.Candlestick(
                x=df_future['Cas'], open=df_future['Open'], high=df_future['High'], low=df_future['Low'], close=df_future['Close'], 
                name="AI Predikce", increasing_line_color='#00ffcc', decreasing_line_color='#ff00ff'
            ), row=1, col=1)

            fig.add_hline(y=setup['Limit_Buy'], line_dash="dash", line_color="#10b981", row=1, col=1)
            fig.add_hline(y=setup['Take_Profit'], line_dash="dot", line_color="#00ffcc", row=1, col=1)
            
            fig.add_trace(go.Scatter(x=[setup['Buy_Time']], y=[setup['Limit_Buy']], mode='markers+text', marker=dict(symbol='triangle-up', size=15, color='#10b981'), text=[f"NÁKUP"], textposition="bottom center", name="Vstup"), row=1, col=1)
            fig.add_trace(go.Scatter(x=[setup['Sell_Time']], y=[setup['Take_Profit']], mode='markers+text', marker=dict(symbol='triangle-down', size=15, color='#00ffcc'), text=[f"EXIT (+{setup['Potencial']:.2f}%)"], textposition="top center", name="Výstup"), row=1, col=1)

            if show_ema20: fig.add_trace(go.Scatter(x=df['Cas'], y=df['EMA_20'], name="EMA 20", line=dict(color='yellow', width=1)), row=1, col=1)
            if show_ema50: fig.add_trace(go.Scatter(x=df['Cas'], y=df['EMA_50'], name="EMA 50", line=dict(color='orange', width=1)), row=1, col=1)
            
            # Zobrazení Bollinger Bands (pokud je zaškrtnuto v menu)
            if show_bb:
                fig.add_trace(go.Scatter(x=df['Cas'], y=df['BB_Upper'], name="BB Upper", line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df['Cas'], y=df['BB_Lower'], name="BB Lower", line=dict(color='gray', width=1, dash='dot'), fill='tonexty'), row=1, col=1)

            colors = ['#ef5350' if df['Open'].iloc[i] > df['Close'].iloc[i] else '#26a69a' for i in range(len(df))]
            fig.add_trace(go.Bar(x=df['Cas'], y=df['Volume'], name="Volume", marker_color=colors, opacity=0.4), row=2, col=1)
            
            fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=750, margin=dict(l=10, r=10, t=30, b=10), hovermode='x unified', dragmode='pan')
            fig.update_xaxes(range=[df['Cas'].iloc[-1] - timedelta(days=2), df_future['Cas'].iloc[-1]])

            # Bod 3: Notifikace při zisku > nastavené %
            if discord_on and setup['Potencial'] > discord_min_profit and discord_url:
                # Použití chytrého formátování i pro Discord
                fmt_buy = format_price(setup['Limit_Buy'])
                fmt_tp = format_price(setup['Take_Profit'])
                fmt_sl = format_price(setup['Stop_Loss'])
                
                msg = f"🚀 **SUPER SIGNAL (>{discord_min_profit}%): {active_coin}**\n" \
                      f"Vstup: {fmt_buy}\n" \
                      f"Cíl: {fmt_tp}\n" \
                      f"Stop Loss: {fmt_sl}\n" \
                      f"Zisk: **{setup['Potencial']:.2f}%**\n" \
                      f"Timeframe: {selected_tf}"
                
                # Generování obrázku grafu (pro přílohu)
                img_bytes = None
                try:
                    # Pokus o vygenerování statického obrázku (vyžaduje kaleido)
                    img_bytes = fig.to_image(format="png", width=1000, height=600, scale=1)
                except Exception as e:
                    # Pokud selže (chybí kaleido), nevadí, pošleme bez obrázku
                    pass

                posli_discord_alert(discord_url, msg, img_bytes)
                st.toast(f"Odeslán alert na Discord pro {active_coin}!")

            # Boxy s plánem (HTML/CSS pro absolutní kontrolu vzhledu)
            c1, c2, c3, c4 = st.columns(4)
            
            # Formátování hodnot pro zobrazení
            fmt_buy = format_price(setup['Limit_Buy'])
            fmt_tp = format_price(setup['Take_Profit'])
            fmt_sl = format_price(setup['Stop_Loss'])
            fmt_curr = format_price(curr_price)
            
            # Společný styl pro karty
            card_style = "padding: 15px; border-radius: 10px; text-align: center; color: white; margin-bottom: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);"
            
            with c1:
                bg = "#15803d" if setup["Limit_Buy"] < curr_price else "#1e3a8a" # Zelená pro limit, Modrá pro market
                lbl = "🎁 LIMIT VSTUP" if setup["Limit_Buy"] < curr_price else "⚡ MARKET VSTUP"
                st.markdown(f"""
                <div style="background-color: {bg}; {card_style}">
                    <div style="font-size: 14px; opacity: 0.8; font-weight: bold;">{lbl}</div>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">{fmt_buy}</div>
                    <div style="font-size: 13px;">🕒 {setup['Buy_Rel']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                st.markdown(f"""
                <div style="background-color: #a16207; {card_style}">
                    <div style="font-size: 14px; opacity: 0.8; font-weight: bold;">🎯 CÍL (TP)</div>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">{fmt_tp}</div>
                    <div style="font-size: 13px;">🕒 {setup['Sell_Rel']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c3:
                st.markdown(f"""
                <div style="background-color: #7f1d1d; {card_style}">
                    <div style="font-size: 14px; opacity: 0.8; font-weight: bold;">🛑 STOP LOSS</div>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">{fmt_sl}</div>
                    <div style="font-size: 13px;">Ochrana kapitálu</div>
                </div>
                """, unsafe_allow_html=True)
                
            with c4:
                st.markdown(f"""
                <div style="background-color: #3b82f6; {card_style}">
                    <div style="font-size: 14px; opacity: 0.8; font-weight: bold;">📈 POTENCIÁL</div>
                    <div style="font-size: 24px; font-weight: bold; margin: 5px 0;">+{setup['Potencial']:.2f}%</div>
                    <div style="font-size: 13px;">Aktuální: {fmt_curr}</div>
                </div>
                """, unsafe_allow_html=True)

            # Zobrazení grafu
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Chyba: AI model nemohl vygenerovat predikci.")
    else:
        st.error(f"Nepodařilo se stáhnout data pro {active_coin}.")

# --- 5. AUTOMATICKÁ AKTUALIZACE (Bod 4) ---
if refresh_min > 0:
    # Placeholder pro odpočet v sidebaru
    countdown_placeholder = st.sidebar.empty()
    total_seconds = refresh_min * 60
    
    # Smyčka pro odpočet (umožňuje přerušení při interakci uživatele)
    for i in range(total_seconds, 0, -1):
        mins, secs = divmod(i, 60)
        countdown_placeholder.info(f"🔄 Aktualizace za: {mins:02d}:{secs:02d}")
        time.sleep(1)
        
    st.rerun()

st.caption(f"Poslední aktualizace: {datetime.now().strftime('%H:%M:%S')}")