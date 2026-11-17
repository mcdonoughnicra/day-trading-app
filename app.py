import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Day Trading Scanner",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Add mobile-friendly CSS
st.markdown("""
<style>
    /* Better mobile layout */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    /* Make tables scroll horizontally on small screens */
    .stDataFrame {
        overflow-x: auto;
    }
    /* Larger touch targets for buttons */
    .stButton > button {
        width: 100%;
        min-height: 48px;
    }
    @media (max-width: 768px) {
        .stSidebar {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Day Trading Scanner: ML + Technicals + Social Sentiment")
st.caption("Created by Craig McDonough")
st.caption("Ranks stocks by machine-learning probability of upward move (trained on technical features) blended with heuristic signals and Reddit social attention. **Not financial advice.**")

# Disclaimer
with st.expander("⚠️ Important Disclaimer & Data Notes", expanded=False):
    st.warning("""
    **This is NOT financial advice.** Trading stocks involves substantial risk of loss. Past performance does not guarantee future results.
    
    The ranking combines:
    - **ML Probability**: Random Forest classifier trained on each stock's own historical technical features (RSI, MACD, SMA ratios, volume, returns, volatility) to predict next-day positive return. This is an in-sample style estimate and can overfit; it is not a calibrated true probability of future moves.
    - **Heuristic Score**: Rule-based confluence of common day-trading signals.
    - **Social Attention**: Reddit mention volume/momentum via ApeWisdom (attention, not directional sentiment).
    
    Combined score weights ML probability heavily. Models are trained fresh on each scan using ~2 years of daily data.
    
    **Data sources (free / near-real-time):**
    - Price & technicals: Yahoo Finance (yfinance). During market hours often near-real-time or lightly delayed for liquid names; not institutional-grade zero-delay data.
    - Social: ApeWisdom public API.
    
    For true zero-delay live data suitable for high-frequency day trading, use a paid feed (Polygon.io, Alpaca SIP, Finnhub premium, etc.). Always verify with your broker and do your own research.
    
    Market is closed on weekends/holidays. Data shown will be the latest available session.
    """)

# Sidebar controls
st.sidebar.header("Scan Settings")

default_tickers = "AAPL, MSFT, NVDA, TSLA, AMD, AMZN, GOOGL, META, SPY, QQQ, PLTR, COIN, MSTR, SMCI, AVGO, ARM, SOXL, TQQQ"
ticker_input = st.sidebar.text_area("Tickers to scan (comma-separated)", value=default_tickers, height=100)

use_social_top = st.sidebar.checkbox("Also include top Reddit-mentioned stocks", value=True)
top_n_social = st.sidebar.slider("How many top social tickers to add", 5, 30, 15)

scan_button = st.sidebar.button("🔍 Scan for Highest Probability Upside", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption("Optional: Finnhub API key for alternative quote source (free key at finnhub.io)")
finnhub_key = st.sidebar.text_input("Finnhub API Key (optional)", type="password", value="")

# Helper functions
@st.cache_data(ttl=300)
def get_social_data():
    try:
        r = requests.get("https://apewisdom.io/api/v1.0/filter/all-stocks", timeout=15)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        social = {}
        for item in results:
            ticker = item.get("ticker", "").upper()
            if ticker:
                social[ticker] = {
                    "mentions": int(item.get("mentions", 0) or 0),
                    "mentions_24h_ago": int(item.get("mentions_24h_ago", 0) or 0),
                    "upvotes": int(item.get("upvotes", 0) or 0),
                    "rank": int(item.get("rank", 999) or 999),
                }
        return social
    except Exception as e:
        st.warning(f"Could not fetch social data: {e}")
        return {}

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - macd_signal
    return macd, macd_signal, hist

def engineer_features(hist):
    """Create technical features for ML."""
    df = hist.copy()
    close = df["Close"]
    volume = df["Volume"]
    
    df["rsi"] = compute_rsi(close)
    _, _, macd_hist = compute_macd(close)
    df["macd_hist"] = macd_hist
    df["sma20_ratio"] = close / close.rolling(20).mean()
    df["sma50_ratio"] = close / close.rolling(50).mean()
    df["vol_ratio"] = volume / volume.rolling(20).mean()
    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["volatility"] = close.pct_change().rolling(10).std()
    
    return df.dropna()

def get_ml_probability(ticker):
    """Train RandomForest on historical technicals to predict next-day up move; return proba for current features."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y", interval="1d")
        if hist.empty or len(hist) < 100:
            return None, None
        
        feats = engineer_features(hist)
        if len(feats) < 60:
            return None, None
        
        # Target: next day positive return
        feats["target"] = (feats["Close"].shift(-1) > feats["Close"]).astype(int)
        feats = feats.dropna()
        
        feature_cols = ["rsi", "macd_hist", "sma20_ratio", "sma50_ratio", "vol_ratio", "ret_1d", "ret_5d", "volatility"]
        X = feats[feature_cols].iloc[:-1]  # exclude last for training
        y = feats["target"].iloc[:-1]
        
        if len(X) < 50 or y.nunique() < 2:
            return None, None
        
        # Train/test split for accuracy estimate
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
        
        model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42, n_jobs=1)
        model.fit(X_train, y_train)
        
        # Accuracy on held-out
        acc = accuracy_score(y_test, model.predict(X_test)) if len(X_test) > 0 else None
        
        # Predict on current (last) features
        current_features = feats[feature_cols].iloc[[-1]]
        proba = model.predict_proba(current_features)[0][1]  # prob of class 1 (up)
        
        return round(float(proba) * 100, 1), round(float(acc) * 100, 1) if acc else None
    except Exception:
        return None, None

def get_stock_analysis(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Daily history for indicators
        hist = stock.history(period="90d", interval="1d")
        if hist.empty or len(hist) < 30:
            return None
        
        # Current / latest price
        info = stock.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
        prev_close = info.get("previousClose") or hist["Close"].iloc[-2] if len(hist) > 1 else current_price
        change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
        
        # Intraday if available (for open market)
        try:
            intraday = stock.history(period="1d", interval="1m")
            if not intraday.empty:
                current_price = intraday["Close"].iloc[-1]
                day_open = intraday["Open"].iloc[0]
                change_pct = ((current_price - day_open) / day_open * 100) if day_open else change_pct
        except Exception:
            pass
        
        close = hist["Close"]
        volume = hist["Volume"]
        
        rsi = compute_rsi(close).iloc[-1]
        macd_line, macd_sig, macd_hist = compute_macd(close)
        macd_hist_val = macd_hist.iloc[-1]
        macd_cross_bull = macd_line.iloc[-1] > macd_sig.iloc[-1] and macd_line.iloc[-2] <= macd_sig.iloc[-2] if len(macd_line) > 1 else False
        
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        avg_vol_20 = volume.rolling(20).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / avg_vol_20 if avg_vol_20 > 0 else 1
        
        # Optional Finnhub live quote
        finnhub_price = None
        if finnhub_key:
            try:
                fh = requests.get(f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_key}", timeout=5)
                if fh.status_code == 200:
                    q = fh.json()
                    finnhub_price = q.get("c")
                    if finnhub_price:
                        current_price = finnhub_price
            except Exception:
                pass
        
        return {
            "ticker": ticker,
            "price": round(float(current_price), 2),
            "change_pct": round(float(change_pct), 2),
            "rsi": round(float(rsi), 1) if not np.isnan(rsi) else None,
            "macd_hist": round(float(macd_hist_val), 4) if not np.isnan(macd_hist_val) else 0,
            "macd_bull_cross": macd_cross_bull,
            "above_sma20": current_price > sma20 if not np.isnan(sma20) else False,
            "above_sma50": current_price > sma50 if not np.isnan(sma50) else False,
            "vol_ratio": round(float(vol_ratio), 2),
            "hist": hist,
            "sma20": sma20,
            "sma50": sma50,
            "float_shares": info.get("floatShares") or info.get("sharesOutstanding"),
        }
    except Exception as e:
        return None

def score_and_reasons(analysis, social_info, ml_prob=None):
    if not analysis:
        return 0, []
    
    score = 0
    reasons = []
    
    # ML component (heavily weighted)
    if ml_prob is not None:
        score += ml_prob * 0.6  # 60% weight
        reasons.append(f"ML up-prob {ml_prob}%")
    
    # Technical points (heuristic)
    if analysis["above_sma20"]:
        score += 12
        reasons.append("Above SMA20")
    if analysis["above_sma50"]:
        score += 8
        reasons.append("Above SMA50")
    
    rsi = analysis["rsi"]
    if rsi is not None:
        if 40 <= rsi <= 70:
            score += 10
            reasons.append(f"RSI healthy ({rsi})")
        elif rsi < 35:
            score += 8
            reasons.append(f"RSI oversold bounce potential ({rsi})")
        elif rsi > 75:
            score -= 4
            reasons.append(f"RSI extended ({rsi})")
    
    if analysis["macd_hist"] > 0:
        score += 12
        reasons.append("MACD histogram positive")
    if analysis["macd_bull_cross"]:
        score += 8
        reasons.append("Recent MACD bullish cross")
    
    if analysis["vol_ratio"] > 1.3:
        score += 8
        reasons.append(f"Volume elevated ({analysis['vol_ratio']}x)")
    elif analysis["vol_ratio"] > 1.0:
        score += 4
    
    if analysis["change_pct"] > 0:
        score += 6
        reasons.append(f"Positive day change ({analysis['change_pct']}%)")
    
    # Social points
    mentions = social_info.get("mentions", 0)
    mentions_ago = social_info.get("mentions_24h_ago", 0)
    if mentions > 50:
        score += 10
        reasons.append(f"High Reddit mentions ({mentions})")
    elif mentions > 20:
        score += 5
        reasons.append(f"Notable Reddit mentions ({mentions})")
    
    if mentions > mentions_ago * 1.5 and mentions > 10:
        score += 6
        reasons.append("Mentions surging")
    
    # Cap roughly 0-100
    score = max(0, min(100, score))
    return round(score, 1), reasons

# Main logic
if "results" not in st.session_state:
    st.session_state.results = None

if scan_button:
    with st.spinner("Scanning stocks with live/near-live data..."):
        social = get_social_data()
        
        # Build ticker list
        user_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        tickers = list(dict.fromkeys(user_tickers))  # unique preserve order
        
        if use_social_top and social:
            top_social = sorted(social.items(), key=lambda x: x[1]["mentions"], reverse=True)[:top_n_social]
            for t, _ in top_social:
                if t not in tickers and len(t) <= 5:  # reasonable ticker length
                    tickers.append(t)
        
        results = []
        progress = st.progress(0)
        status = st.empty()
        for i, ticker in enumerate(tickers[:100]):  # limit to avoid long scans
            status.text(f"Analyzing {ticker} ({i+1}/{min(len(tickers),100)})...")
            analysis = get_stock_analysis(ticker)
            if analysis:
                social_info = social.get(ticker, {})
                ml_prob, ml_acc = get_ml_probability(ticker)
                score, reasons = score_and_reasons(analysis, social_info, ml_prob)
                results.append({
                    "Ticker": ticker,
                    "Combined Score": score,
                    "ML Prob %": ml_prob if ml_prob is not None else "—",
                    "ML Acc %": ml_acc if ml_acc is not None else "—",
                    "Price": analysis["price"],
                    "Change %": analysis["change_pct"],
                    "RSI": analysis["rsi"],
                    "MACD Hist": analysis["macd_hist"],
                    "Vol Ratio": analysis["vol_ratio"],
                    "Mentions": social_info.get("mentions", 0),
                    "Float (M)": round(analysis["float_shares"] / 1_000_000, 1) if analysis.get("float_shares") else "—",
                    "Signals": " | ".join(reasons[:5]) if reasons else "—",
                    "_analysis": analysis,
                    "_reasons": reasons,
                })
            progress.progress((i + 1) / min(len(tickers), 100))
        
        progress.empty()
        status.empty()
        if results:
            df = pd.DataFrame(results)
            df = df.sort_values("Combined Score", ascending=False).reset_index(drop=True)
            st.session_state.results = df
            st.session_state.scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            st.error("No data retrieved. Check tickers or try again.")
            st.session_state.results = None

# Display results
if st.session_state.results is not None:
    df = st.session_state.results
    st.success(f"Scan complete at {st.session_state.get('scan_time', 'now')}. Market status: Closed (weekend) — showing latest available data. Re-scan during market hours for fresher prices.")
    
    # Top picks
    high_prob = df[df["Combined Score"] >= 55]
    st.subheader(f"🔥 Highest Probability Upside Candidates ({len(high_prob)} with score ≥ 55)")
    
    if not high_prob.empty:
        display_df = high_prob[["Ticker", "Combined Score", "ML Prob %", "ML Acc %", "Price", "Change %", "RSI", "MACD Hist", "Vol Ratio", "Mentions", "Signals"]].copy()
        st.dataframe(
            display_df.style.background_gradient(subset=["Combined Score"], cmap="Greens").format({
                "Price": "${:.2f}",
                "Change %": "{:+.2f}%",
                "RSI": "{:.1f}",
                "MACD Hist": "{:.4f}",
                "Vol Ratio": "{:.2f}x",
            }),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No stocks currently meet the high-confluence threshold. Showing all ranked below.")
    
    st.subheader("Full Ranking")
    st.dataframe(
        df[["Ticker", "Combined Score", "ML Prob %", "ML Acc %", "Price", "Change %", "RSI", "MACD Hist", "Vol Ratio", "Mentions", "Float (M)", "Signals"]],
        width="stretch",
        hide_index=True,
    )
    
    # Low Float Movers section
    st.subheader("🚀 Highest Probability Low Float Movers")
    low_float = df[df["Float (M)"].apply(lambda x: isinstance(x, (int, float)) and x < 20)].copy()
    if not low_float.empty:
        low_float = low_float.sort_values("Combined Score", ascending=False)
        st.dataframe(
            low_float[["Ticker", "Combined Score", "ML Prob %", "Price", "Change %", "Float (M)", "Vol Ratio", "Signals"]].head(10),
            width="stretch",
            hide_index=True,
        )
        st.caption("Low float = under 20 million shares. These can move faster but are riskier.")
    else:
        st.info("No low-float stocks found in this scan (or float data unavailable).")
    
    # Chart for selected
    st.subheader("Detail Chart")
    selected = st.selectbox("Select ticker for chart & details", options=df["Ticker"].tolist())
    if selected:
        row = df[df["Ticker"] == selected].iloc[0]
        analysis = row["_analysis"]
        hist = analysis["hist"]
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name="Price"
        ), row=1, col=1)
        
        # SMAs
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(20).mean(), name="SMA20", line=dict(color="orange")), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(), name="SMA50", line=dict(color="blue")), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], name="Volume", marker_color="rgba(100,100,100,0.4)"), row=2, col=1)
        
        # Detect simple buy points (MACD bullish cross or RSI bounce from oversold)
        close = hist["Close"]
        rsi_series = compute_rsi(close)
        macd_line, macd_sig, _ = compute_macd(close)
        
        buy_indices = []
        buy_prices = []
        for i in range(1, len(hist)):
            # MACD bullish cross
            if macd_line.iloc[i] > macd_sig.iloc[i] and macd_line.iloc[i-1] <= macd_sig.iloc[i-1]:
                buy_indices.append(hist.index[i])
                buy_prices.append(close.iloc[i])
            # RSI crossing above 35 from below (bounce potential)
            elif rsi_series.iloc[i] > 35 and rsi_series.iloc[i-1] <= 35 and not np.isnan(rsi_series.iloc[i]):
                buy_indices.append(hist.index[i])
                buy_prices.append(close.iloc[i])
        
        # Add red upward arrows at buy points
        if buy_indices:
            fig.add_trace(go.Scatter(
                x=buy_indices,
                y=buy_prices,
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="red"),
                name="Potential Buy",
                hovertemplate="Potential Buy<br>%{x}<br>$%{y:.2f}<extra></extra>"
            ), row=1, col=1)
        
        fig.update_layout(height=600, title=f"{selected} — Combined Score: {row['Combined Score']} | ML Prob: {row['ML Prob %']}%", xaxis_rangeslider_visible=False)
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.write("**Signals contributing to score:**")
        for r in row["_reasons"]:
            st.write(f"- {r}")
        
        st.write(f"**Latest price:** ${analysis['price']} ({analysis['change_pct']:+.2f}%)")
        st.write(f"**RSI:** {analysis['rsi']} | **MACD Hist:** {analysis['macd_hist']} | **Vol ratio:** {analysis['vol_ratio']}x")

else:
    st.info("Click **Scan for Highest Probability Upside** in the sidebar to analyze stocks using technical indicators and Reddit social attention.")

st.markdown("---")
st.caption("Built with free/public data sources. For production day trading use professional real-time feeds and your own risk management. Always do your own research.")
