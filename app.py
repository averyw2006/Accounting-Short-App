"""ACCT 3343 short (SELL) stock screener.

Run with: streamlit run app.py
The app is designed to work with a saved universe.csv in its repository.
It can also attempt to refresh quote ratios directly from Yahoo Finance.
"""

from datetime import datetime, timezone
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Short Thesis | Valuation Screener", page_icon="📉", layout="wide")

METRICS = ["trailing_pe", "forward_pe", "price_to_book", "ev_to_ebitda", "debt_to_equity", "return_on_equity", "revenue_growth", "altman_z", "price_to_sales", "profit_margin", "operating_margin", "earnings_growth", "free_cash_flow", "total_cash", "total_debt"]
ALIASES = {
    "symbol": "ticker", "Ticker": "ticker", "Symbol": "ticker", "company": "name",
    "shortName": "name", "longName": "name", "sectorName": "sector",
    "trailingPE": "trailing_pe", "forwardPE": "forward_pe",
    "priceToBook": "price_to_book", "enterpriseToEbitda": "ev_to_ebitda",
    "debtToEquity": "debt_to_equity", "returnOnEquity": "return_on_equity",
    "revenueGrowth": "revenue_growth", "marketCap": "market_cap",
    "priceToSalesTrailing12Months": "price_to_sales", "profitMargins": "profit_margin",
    "operatingMargins": "operating_margin", "earningsGrowth": "earnings_growth",
    "freeCashflow": "free_cash_flow", "totalCash": "total_cash", "totalDebt": "total_debt",
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


VALUATION_FACTORS = [
    ("trailing_pe", "pe", "P/E"),
    ("price_to_book", "pb", "P/B"),
    ("ev_to_ebitda", "ev", "EV/EBITDA"),
    ("price_to_sales", "ps", "P/S"),
]


def compute_flags(df, premium_limit, pe_limit, pb_limit, use_absolute, min_signals, min_peers,
                  weights, min_weighted_score, z_cutoff, use_z, use_debt, de_limit):
    """Four distinct peer-relative multiples, with separate risk annotations.

    Benchmarks are calculated on the full unfiltered universe so a sidebar
    sector selection never changes the underlying peer definition.
    """
    df = df.copy()
    for metric, prefix, _ in VALUATION_FACTORS:
        usable = df[metric].where(df[metric] > 0)
        group = usable.groupby(df["sector"])
        df[f"{prefix}_peer_count"] = group.transform("count")
        df[f"{prefix}_sector_median"] = group.transform("median").where(df[f"{prefix}_peer_count"] >= min_peers)
        df[f"{prefix}_premium_pct"] = (usable / df[f"{prefix}_sector_median"] - 1) * 100
    weighted_earned = pd.Series(0.0, index=df.index)
    weighted_available = pd.Series(0.0, index=df.index)
    df["signals"] = 0
    df["evaluated"] = 0
    for metric, prefix, _ in VALUATION_FACTORS:
        available = df[f"{prefix}_premium_pct"].notna()
        triggered = available & (df[f"{prefix}_premium_pct"] >= premium_limit)
        if use_absolute and prefix == "pe":
            triggered &= df[metric] >= pe_limit
        if use_absolute and prefix == "pb":
            triggered &= df[metric] >= pb_limit
        df[f"{prefix}_signal"] = triggered
        if weights[prefix] > 0:
            df["signals"] += triggered.astype(int)
            df["evaluated"] += available.astype(int)
            weighted_earned += triggered.astype(float) * weights[prefix]
            weighted_available += available.astype(float) * weights[prefix]
    df["weighted_score_pct"] = np.where(weighted_available > 0, 100 * weighted_earned / weighted_available, np.nan)
    df["earnings_yield_pct"] = np.where(df["trailing_pe"] > 0, 100 / df["trailing_pe"], np.nan)
    df["implied_eps_growth_pct"] = np.where(
        (df["trailing_pe"] > 0) & (df["forward_pe"] > 0),
        (df["trailing_pe"] / df["forward_pe"] - 1) * 100, np.nan,
    )
    df["risk_notes"] = ""
    if use_debt:
        df["risk_notes"] += np.where(df["debt_to_equity"] > de_limit, "High debt/equity; ", "")
    if use_z:
        df["risk_notes"] += np.where((df["altman_z"] < z_cutoff) & df["altman_z"].notna() & ~df["sector"].eq("Financials"), "Low Altman Z; ", "")
    df["risk_notes"] = df["risk_notes"].str.rstrip("; ")
    # A same-fundamentals, peer-multiple reference price, not intrinsic value.
    pe_valid = df["pe_premium_pct"].notna()
    pb_valid = df["pb_premium_pct"].notna()
    df["pe_peer_price"] = np.where(pe_valid & (df["price"] > 0),
                                   df["price"] * df["pe_sector_median"] / df["trailing_pe"], np.nan)
    df["pb_peer_price"] = np.where(pb_valid & (df["price"] > 0),
                                   df["price"] * df["pb_sector_median"] / df["price_to_book"], np.nan)
    prices = df[["pe_peer_price", "pb_peer_price"]]
    df["peer_reference_price"] = prices.mean(axis=1, skipna=True)
    df.loc[prices.notna().sum(axis=1) < min_signals, "peer_reference_price"] = np.nan
    df["reference_gap_pct"] = (df["price"] / df["peer_reference_price"] - 1) * 100
    premiums = pd.DataFrame({prefix: df[f"{prefix}_premium_pct"].clip(lower=-100, upper=500) for _,prefix,_ in VALUATION_FACTORS})
    df["average_premium_pct"] = premiums.mean(axis=1)
    df["screen_result"] = np.select(
        [df["evaluated"] < min_signals, (df["signals"] >= min_signals) & (df["weighted_score_pct"] >= min_weighted_score)],
        ["Insufficient data", "Potential SELL"], default="Not flagged",
    )
    df["why_flagged"] = df.apply(lambda r: ", ".join(
        f"{label} above sector median" for _, prefix, label in VALUATION_FACTORS if weights[prefix] > 0 and r[f"{prefix}_signal"]
    ) or "Does not meet the selected peer-valuation rules", axis=1)
    return df.sort_values(["signals", "weighted_score_pct", "evaluated", "average_premium_pct", "market_cap"],
                          ascending=[False, False, False, False, False], na_position="last")


st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1450px}
.hero{padding:25px 30px;background:linear-gradient(120deg,#111d30,#263653);color:white;border-radius:15px;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem}.hero p{margin:10px 0 0;color:#cbd5e1}
</style>""", unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>Short Thesis · US Stock Screener</h1><p>903-company universe · Four valuation methods · Peer-relative short-candidate research</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Universe", ["Saved CSV", "Yahoo live screener", "Upload my CSV"], index=0)
    limit = st.select_slider("Yahoo universe size", options=[600, 750, 900, 1000], value=900, disabled=source != "Yahoo live screener")
    upload = st.file_uploader("Upload CSV", type="csv") if source == "Upload my CSV" else None
    st.caption("The live screener needs internet access and Yahoo availability. Financial metrics may be incomplete.")
    st.divider()
    st.header("2 · Peer-valuation rules")
    premium_limit = st.slider("Premium above sector median (%)", 0, 200, 35, 5,
                              help="A 35% cutoff requires a company's multiple to exceed its sector median by at least 35%.")
    min_signals = st.slider("Minimum valuation signals", 1, 4, 2)
    min_weighted_score = st.slider("Minimum weighted signal score (%)", 25, 100, 65, 5)
    min_peers = st.slider("Minimum companies per peer benchmark", 3, 30, 5)
    st.markdown("**Importance weights** (0 = exclude the measure)")
    weights = {
        "pe": st.slider("P/E weight", 0, 4, 2),
        "pb": st.slider("P/B weight", 0, 4, 2),
        "ev": st.slider("EV/EBITDA weight", 0, 4, 1),
        "ps": st.slider("P/S weight", 0, 4, 1),
    }
    use_absolute = st.checkbox("Also require absolute multiple cutoffs", value=False)
    pe_limit = st.slider("P/E at least", 5.0, 100.0, 30.0, 1.0, disabled=not use_absolute)
    pb_limit = st.slider("P/B at least", 0.5, 30.0, 5.0, 0.5, disabled=not use_absolute)
    st.caption("Sector medians are recalculated from the full available universe. Negative and missing multiples are excluded.")
    st.divider()
    st.header("3 · Optional risk annotations")
    use_debt = st.checkbox("Annotate high debt / equity", value=False)
    de_limit = st.slider("Debt / equity above", 10.0, 500.0, 150.0, 10.0, disabled=not use_debt)
    use_z = st.checkbox("Annotate low Altman Z", value=False)
    z_cutoff = st.slider("Altman Z below", 0.0, 4.0, 1.81, 0.01, disabled=not use_z)
    st.caption("Risk indicators are shown separately; they do not count as overvaluation signals. Yahoo D/E is often a percent (150 = 1.5×). Z-score requires uploaded statement data.")

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

if not any(weights.values()):
    st.error("Choose a positive weight for at least one valuation measure.")
    st.stop()
result = compute_flags(data, premium_limit, pe_limit, pb_limit, use_absolute, min_signals, min_peers,
                       weights, min_weighted_score, z_cutoff, use_z, use_debt, de_limit)
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

benchmark_counts = {label: int(result[f"{prefix}_premium_pct"].notna().sum()) for _,prefix,label in VALUATION_FACTORS}
st.caption("Usable sector benchmarks: " + " · ".join(f"{label}: {n:,}" for label,n in benchmark_counts.items()))
if sum(benchmark_counts[label] > 0 and weights[prefix] > 0 for _,prefix,label in VALUATION_FACTORS) < min_signals:
    st.warning("Fewer valuation methods have usable peer benchmarks than your minimum-signal setting. Lower the minimum or upload more complete statement data.")

coverage = pd.DataFrame({"Metric": ["Trailing P/E", "Forward P/E", "Price / book", "EV / EBITDA", "Debt / equity", "Return on equity", "Revenue growth", "Free cash flow", "Altman Z", "Share price"],
                         "Available": [int(result[c].notna().sum()) for c in ["trailing_pe", "forward_pe", "price_to_book", "ev_to_ebitda", "debt_to_equity", "return_on_equity", "revenue_growth", "free_cash_flow", "altman_z", "price"]]})
with st.expander("Data coverage and methodology", expanded=False):
    st.dataframe(coverage, hide_index=True, width="stretch")
    st.markdown("""
    **Four distinct valuation signals (not four ratios derived from the same earnings denominator):**

    1. P/E premium = `(company P/E ÷ median positive sector P/E − 1) × 100`.
    2. P/B premium = `(company P/B ÷ median positive sector P/B − 1) × 100`.
    3. EV/EBITDA premium = `(company EV/EBITDA ÷ median positive sector EV/EBITDA − 1) × 100`.
    4. P/S premium = `(company P/S ÷ median positive sector P/S − 1) × 100`.

    A metric only counts if its company's positive multiple and a sector median computed from at least the chosen number of peers exist. Each signal fires when its premium meets the threshold. A potential SELL needs both the selected count of triggered signals **and** the minimum weighted score. The score is `100 × (sum of weights for triggered available signals ÷ sum of weights for all available signals)`; unavailable metrics are omitted from the denominator. A zero weight excludes a metric entirely. The optional absolute cutoffs add an extra hurdle for P/E and P/B. **Missing, zero, or negative multiples are never assigned fictitious zero values.**

    **Other accounting measures:** Earnings yield = `100 ÷ trailing P/E`. A forward earnings change proxy = `(trailing P/E ÷ forward P/E − 1) × 100`, which is meaningful only when both ratios use the same price basis and consistent earnings horizons. A peer reference share price is the average of `share price × sector-median multiple ÷ company multiple` across available and valid P/E and P/B measures. This is a rough **relative valuation reference**, not a discounted cash-flow or intrinsic valuation.

    **Limitations:** Higher sector-relative multiples can be warranted by expected growth, higher profitability, risk differences, or accounting choices. Sector categories are broad and not always ideal peer groups. The saved universe supplements broad quote-ratio coverage with deeper information for a **non-random subset of candidates and large firms**. These extra metrics must not be treated as representative of the full universe. Debt/equity and Altman Z are **risk annotations**, not evidence of overpricing; original Altman Z is not appropriate for financial firms. Ratios and prices may be delayed or drawn from differing reporting periods. This exercise produces research candidates, **not trading advice**.

    Sources and methods: [CFA Institute — market-based valuation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/market-based-valuation-price-enterprise-value-multiples) · [CFA Institute — valuation basics](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/equity-valuation-concepts-basic-tools) · Data: Yahoo Finance snapshot and S&P index listings.
    """)

st.subheader("Ranked screening results")
st.caption("Ranked by number of triggered signals, weighted score, and data coverage. Peer reference prices are estimates, not price targets.")
columns = ["ticker", "name", "sector", "screen_result", "signals", "evaluated", "weighted_score_pct", "trailing_pe", "pe_premium_pct", "price_to_book", "pb_premium_pct", "ev_to_ebitda", "ev_premium_pct", "price_to_sales", "ps_premium_pct", "reference_gap_pct", "risk_notes"]
st.dataframe(view[columns], width="stretch", hide_index=True, height=430,
             column_config={"trailing_pe": st.column_config.NumberColumn("P/E", format="%.1f"),
                            "weighted_score_pct": st.column_config.NumberColumn("Weighted score", format="%.0f%%"),
                            "pe_premium_pct": st.column_config.NumberColumn("P/E premium", format="%.1f%%"),
                            "price_to_book": st.column_config.NumberColumn("P/B", format="%.1f"),
                            "pb_premium_pct": st.column_config.NumberColumn("P/B premium", format="%.1f%%"),
                            "ev_to_ebitda": st.column_config.NumberColumn("EV/EBITDA", format="%.1f"),
                            "ev_premium_pct": st.column_config.NumberColumn("EV premium", format="%.1f%%"),
                            "price_to_sales": st.column_config.NumberColumn("P/S", format="%.1f"),
                            "ps_premium_pct": st.column_config.NumberColumn("P/S premium", format="%.1f%%"),
                            "reference_gap_pct": st.column_config.NumberColumn("Above peer reference", format="%.1f%%")})
st.download_button("Download filtered results (CSV)", view.to_csv(index=False).encode(), "short_screen_results.csv", "text/csv")

left, right = st.columns([1.2, 1])
with left:
    chart = result[result["evaluated"] >= min_signals].groupby("sector", dropna=False).agg(
        universe=("ticker", "count"), flagged=("screen_result", lambda s: int((s == "Potential SELL").sum()))
    ).reset_index().sort_values("flagged", ascending=False).head(15)
    if len(chart):
        chart["flagged_pct"] = (chart["flagged"] / chart["universe"] * 100).round(1)
        fig = px.bar(chart, x="flagged_pct", y="sector", orientation="h", title="Flagged share of measurable sector (%)", hover_data=["flagged", "universe"])
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Flagged companies", yaxis_title="")
        fig.update_xaxes(title="Flagged share (%)")
        st.plotly_chart(fig, width="stretch")
with right:
    st.subheader("Inspect one company")
    options = view["ticker"].tolist()
    if options:
        picked = st.selectbox("Ticker", options)
        row = view[view["ticker"] == picked].iloc[0]
        st.markdown(f"**{row['name']}** · {row['sector']}")
        st.markdown(f"**{row['screen_result']}** — {int(row['signals'])} / {int(row['evaluated'])} observed peer-valuation signals triggered; **weighted score {row['weighted_score_pct']:.0f}%**")
        st.write(row["why_flagged"])
        for col, prefix, label in VALUATION_FACTORS:
            sector_col, premium_col = f"{prefix}_sector_median", f"{prefix}_premium_pct"
            if pd.notna(row[premium_col]):
                st.write(f"**{label}:** {row[col]:.1f} vs sector median {row[sector_col]:.1f} → {row[premium_col]:+.1f}% premium")
            else:
                st.write(f"**{label}:** {row[col]:.1f}× but too few comparable peers" if pd.notna(row[col]) else f"**{label}:** data unavailable")
        if pd.notna(row["reference_gap_pct"]):
            st.metric("Market price above peer-multiple reference", f"{row['reference_gap_pct']:+.1f}%")
            st.caption(f"Observed share price ${row['price']:,.2f}; approximate peer reference ${row['peer_reference_price']:,.2f}. Same-company fundamentals, sector median multiples; not an intrinsic value estimate.")
        if pd.notna(row["earnings_yield_pct"]):
            st.write(f"**Earnings yield:** {row['earnings_yield_pct']:.2f}% (inverse of P/E)")
        if pd.notna(row["implied_eps_growth_pct"]):
            st.write(f"**Forward vs trailing earnings proxy:** {row['implied_eps_growth_pct']:+.1f}% (not an actual realized growth rate)")
        detail_items = [("EV/EBITDA", "ev_to_ebitda", "multiple"),
                        ("Price/sales", "price_to_sales", "multiple"),
                        ("Return on equity", "return_on_equity", "pct"),
                        ("Revenue growth", "revenue_growth", "pct"),
                        ("Profit margin", "profit_margin", "pct"),
                        ("Debt/equity", "debt_to_equity", "de"),
                        ("Free cash flow", "free_cash_flow", "money")]
        details = []
        for label, key, unit in detail_items:
            val=row[key]
            if pd.isna(val):
                continue
            formatted = (f"{val:.1f}×" if unit == "multiple" else
                         f"{val * 100:.1f}%" if unit == "pct" else
                         f"{val / 100:.2f}×" if unit == "de" else
                         f"${val / 1e9:,.2f}B")
            details.append(f"**{label}:** {formatted}")
        if details:
            st.markdown("**Additional financial context**")
            for item in details:
                st.markdown(item)
            st.caption("Additional ratios cover only selected stocks; figures may have different reporting periods. Revenue growth and ROE can support a high valuation, rather than a short thesis.")
            if pd.notna(row['revenue_growth']) and row['revenue_growth'] > 0.25:
                st.info("Counterargument: strong reported revenue growth may help justify an above-peer valuation. Investigate whether growth is sustainable.")
            if pd.notna(row['revenue_growth']) and row['revenue_growth'] < 0.05 and row['signals'] == 2:
                st.warning("Thesis-supporting observation: both valuation premiums are high while reported revenue growth is below 5%. Check the reporting period and any one-time effects.")
        if row["risk_notes"]:
            st.warning(f"Additional risk annotation: {row['risk_notes']}")
        st.caption("A short candidate requires further work on growth expectations, earnings quality, financing cost, and short-sale risk. A high multiple alone does not establish overvaluation.")
    else:
        st.info("No stocks currently match. Lower the thresholds or remove filters.")

with st.expander("What would strengthen a real SELL case?", expanded=False):
    st.markdown("""
    - **Valuation:** Is the company's multiple expensive even after comparing it with similar peers and its expected growth?
    - **Accounting quality:** Are operating cash flows keeping up with reported net income? Are receivables or inventories growing unusually fast?
    - **Financial stability:** What are the leverage, interest coverage, and relevant financial-distress signals?
    - **Catalyst:** What specific event could cause the market to lower the valuation, and over what time horizon?
    - **Counterargument:** Could exceptional profitability or growth justify the premium? What would disprove the short thesis?

    These questions require company financial statements and additional research beyond quote multiples. This app identifies candidates for that next stage.
    """)
st.caption("ACCT 3343 educational screener · Yahoo-sourced ratios are a dated snapshot · No personalized investment recommendation")
