"""ACCT 3343 short (SELL) stock screener.

Run with: streamlit run app.py
The app is designed to work with a saved universe.csv in its repository.
It can also attempt to refresh quote ratios directly from Yahoo Finance.
"""

from datetime import datetime, timezone
from io import BytesIO
import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Short Thesis | Valuation Screener", page_icon="📉", layout="wide")

METRICS = ["trailing_pe", "forward_pe", "price_to_book", "ev_to_ebitda", "debt_to_equity", "return_on_equity", "revenue_growth", "altman_z"]
ALIASES = {
    "symbol": "ticker", "Ticker": "ticker", "Symbol": "ticker", "company": "name",
    "shortName": "name", "longName": "name", "sectorName": "sector",
    "trailingPE": "trailing_pe", "forwardPE": "forward_pe",
    "priceToBook": "price_to_book", "enterpriseToEbitda": "ev_to_ebitda",
    "debtToEquity": "debt_to_equity", "returnOnEquity": "return_on_equity",
    "revenueGrowth": "revenue_growth", "marketCap": "market_cap",
    "regularMarketPrice": "price", "regularMarketChangePercent": "change_pct",
    "Altman_Z": "altman_z", "Altman Z": "altman_z",
}


def normalize(df):
    df = df.rename(columns=ALIASES).copy()
    df = df.loc[:, ~df.columns.duplicated()]
    if "ticker" not in df:
        raise ValueError("The CSV must contain a 'ticker' or 'symbol' column.")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].str.fullmatch(r"[A-Z0-9.\-^]+", na=False)]
    df = df.drop_duplicates("ticker")
    for col in METRICS + ["market_cap", "price", "change_pct"]:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for col in ("name", "sector"):
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("Unknown").astype(str)
    return df.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_csv():
    return normalize(pd.read_csv("universe.csv"))


@st.cache_data(ttl=3600, show_spinner="Requesting Yahoo Finance screener pages...")
def yahoo_screen(limit):
    import yfinance as yf
    from yfinance import EquityQuery

    query = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        EquityQuery("gte", ["intradaymarketcap", 3e8]),
    ])
    rows = []
    for offset in range(0, limit, 250):
        page = yf.screen(query, offset=offset, size=min(250, limit-offset), sortField="intradaymarketcap", sortAsc=False)
        quotes = page.get("quotes", [])
        rows.extend(quotes)
        if len(quotes) == 0:
            break
    if not rows:
        raise RuntimeError("Yahoo did not return any companies. Please use the CSV data instead.")
    return normalize(pd.DataFrame(rows))


def read_uploaded(data):
    return normalize(pd.read_csv(BytesIO(data)))


def compute_flags(df, pe_limit, pb_limit, ebitda_limit, min_signals, z_cutoff, use_z, use_debt, de_limit):
    df = df.copy()
    criteria = {
        "High P/E": df["trailing_pe"] > pe_limit,
        "High P/B": df["price_to_book"] > pb_limit,
        "High EV/EBITDA": df["ev_to_ebitda"] > ebitda_limit,
    }
    available = {
        "High P/E": df["trailing_pe"].notna() & (df["trailing_pe"] > 0),
        "High P/B": df["price_to_book"].notna() & (df["price_to_book"] > 0),
        "High EV/EBITDA": df["ev_to_ebitda"].notna() & (df["ev_to_ebitda"] > 0),
    }
    if use_debt:
        criteria["High D/E"] = df["debt_to_equity"] > de_limit
        available["High D/E"] = df["debt_to_equity"].notna() & (df["debt_to_equity"] >= 0)
    if use_z:
        criteria["Weak Z-score"] = df["altman_z"] < z_cutoff
        available["Weak Z-score"] = df["altman_z"].notna()
    for key, condition in criteria.items():
        df[key] = condition & available[key]
    df["signals"] = sum(df[key].astype(int) for key in criteria)
    df["evaluated"] = sum(mask.astype(int) for mask in available.values())
    df["flag_rate"] = np.where(df["evaluated"] > 0, df["signals"] / df["evaluated"], np.nan)
    df["screen_result"] = np.select(
        [df["evaluated"] < min_signals, df["signals"] >= min_signals],
        ["Insufficient data", "Potential SELL"], default="Not flagged",
    )
    df["why_flagged"] = df.apply(
        lambda r: ", ".join(k for k in criteria if r[k]) or "No triggered valuation threshold", axis=1
    )
    return df.sort_values(["signals", "flag_rate", "market_cap"], ascending=[False, False, False], na_position="last")


st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1450px}
.hero{padding:25px 30px;background:linear-gradient(120deg,#111d30,#263653);color:white;border-radius:15px;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem}.hero p{margin:10px 0 0;color:#cbd5e1}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Short Thesis · US Stock Screener</h1><p>Find potentially overvalued stocks using transparent, adjustable accounting-based thresholds.</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Universe", ["Saved CSV", "Yahoo live screener", "Upload my CSV"], index=0)
    limit = st.select_slider("Yahoo universe size", options=[600, 750, 900, 1000], value=900, disabled=source != "Yahoo live screener")
    upload = st.file_uploader("Upload CSV", type="csv") if source == "Upload my CSV" else None
    st.caption("The live screener needs internet access and Yahoo availability. Financial metrics may be incomplete.")
    st.divider()
    st.header("2 · Valuation thresholds")
    pe_limit = st.slider("Trailing P/E above", 5.0, 100.0, 30.0, 1.0)
    pb_limit = st.slider("Price / book above", 0.5, 30.0, 5.0, 0.5)
    ev_limit = st.slider("EV / EBITDA above", 3.0, 50.0, 20.0, 1.0)
    min_signals = st.slider("Minimum triggered signals", 1, 3, 2)
    st.divider()
    st.header("3 · Optional risk checks")
    use_debt = st.checkbox("Include high debt / equity", value=False)
    de_limit = st.slider("Debt / equity above", 10.0, 500.0, 150.0, 10.0, disabled=not use_debt)
    use_z = st.checkbox("Include low Altman Z", value=False)
    z_cutoff = st.slider("Altman Z below", 0.0, 4.0, 1.81, 0.01, disabled=not use_z)
    st.caption("Yahoo may express debt/equity as a percentage (150 = 1.5×). Z-score is used only if your CSV includes calculated values.")

try:
    if source == "Saved CSV":
        data = load_csv()
        provenance = "Saved universe.csv · values as provided (not automatically refreshed)"
    elif source == "Yahoo live screener":
        data = yahoo_screen(limit)
        provenance = f"Yahoo Finance screener · retrieved {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · cache up to 1 hour"
    elif upload is not None:
        data = read_uploaded(upload.getvalue())
        provenance = "Uploaded user-provided data"
    else:
        st.info("Upload a CSV containing at least a ticker column and valuation ratios.")
        st.stop()
except Exception as exc:
    st.error(f"Unable to load this data source: {exc}")
    st.stop()

result = compute_flags(data, pe_limit, pb_limit, ev_limit, min_signals, z_cutoff, use_z, use_debt, de_limit)
with st.sidebar:
    st.header("4 · Narrow the list")
    sectors = sorted(s for s in result["sector"].dropna().unique() if s and s != "Unknown")
    selected_sectors = st.multiselect("Sectors", sectors)
    only_sell = st.checkbox("Show only potential SELLs", value=True)
    search = st.text_input("Ticker or company name")

view = result.copy()
if selected_sectors:
    view = view[view["sector"].isin(selected_sectors)]
if only_sell:
    view = view[view["screen_result"] == "Potential SELL"]
if search:
    view = view[view["ticker"].str.contains(search, case=False, regex=False) | view["name"].str.contains(search, case=False, regex=False)]

st.caption(provenance)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Companies analyzed", f"{len(data):,}")
c2.metric("Potential SELL candidates", f"{sum(result['screen_result'].eq('Potential SELL')):,}")
c3.metric("With enough ratio data", f"{sum(result['evaluated'] >= min_signals):,}")
c4.metric("Shown after filters", f"{len(view):,}")

if len(data) < 600:
    st.warning("The current universe has fewer than 600 companies. Use the Yahoo live screener or upload a complete 600–1,000-stock dataset for your submission.")

coverage = pd.DataFrame({"Metric": ["Trailing P/E", "Price / book", "EV / EBITDA", "Debt / equity", "Altman Z"],
                         "Available": [int(result[c].notna().sum()) for c in ["trailing_pe", "price_to_book", "ev_to_ebitda", "debt_to_equity", "altman_z"]]})
with st.expander("Data coverage and methodology", expanded=False):
    st.dataframe(coverage, hide_index=True, use_container_width=True)
    st.markdown("""
    **Screening logic:** A positive trailing P/E above the cutoff, positive price/book above the cutoff, and positive EV/EBITDA above the cutoff each count as one possible overvaluation signal. Companies need at least the chosen number of observed metrics and triggered signals to be flagged. Negative multiples and missing values are **not** treated as zero or counted as overvaluation.

    **Interpretation:** High multiples are *relative warning signs*, not a proof of overpricing or a short-sale recommendation. Growth prospects and sector differences matter. Debt/equity and Altman Z indicate financial risk rather than valuation; the latter should be calculated from appropriately comparable company statements and is generally unsuitable for financial firms. Yahoo-provided fields can have reporting-date, unit, or coverage limitations.
    """)

st.subheader("Screening results")
columns = ["ticker", "name", "sector", "screen_result", "signals", "evaluated", "trailing_pe", "price_to_book", "ev_to_ebitda", "debt_to_equity", "altman_z", "why_flagged"]
st.dataframe(view[columns], use_container_width=True, hide_index=True, height=430,
             column_config={"trailing_pe": st.column_config.NumberColumn("P/E", format="%.1f"),
                            "price_to_book": st.column_config.NumberColumn("P/B", format="%.1f"),
                            "ev_to_ebitda": st.column_config.NumberColumn("EV/EBITDA", format="%.1f"),
                            "debt_to_equity": st.column_config.NumberColumn("D/E (Yahoo %)", format="%.1f"),
                            "altman_z": st.column_config.NumberColumn("Altman Z", format="%.2f")})
st.download_button("Download filtered results (CSV)", view.to_csv(index=False).encode(), "short_screen_results.csv", "text/csv")

left, right = st.columns([1.2, 1])
with left:
    chart = result[result["evaluated"] >= min_signals].groupby("sector", dropna=False).agg(
        universe=("ticker", "count"), flagged=("screen_result", lambda s: int((s == "Potential SELL").sum()))
    ).reset_index().sort_values("flagged", ascending=False).head(15)
    if len(chart):
        fig = px.bar(chart, x="flagged", y="sector", orientation="h", title="SELL flags by sector", hover_data=["universe"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Flagged companies", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Inspect one company")
    options = view["ticker"].tolist()
    if options:
        picked = st.selectbox("Ticker", options)
        row = view[view["ticker"] == picked].iloc[0]
        st.markdown(f"**{row['name']}** · {row['sector']}")
        st.markdown(f"**{row['screen_result']}** — {int(row['signals'])} / {int(row['evaluated'])} observed signals triggered")
        st.write(row["why_flagged"])
        st.caption("A SELL flag means this company deserves additional research, not that its intrinsic value has been established.")
    else:
        st.info("No stocks currently match. Lower the thresholds or remove filters.")

st.caption("Educational financial statement screener for ACCT 3343 · Prices and ratios may be delayed; independently verify any investment conclusion.")
