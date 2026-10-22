"""ACCT 3343 | Short Analysis stock screener.

Deployment: streamlit run app.py. Read methodology in the app before interpreting flags.
Saved universe.csv is a dated, INCOMPLETE snapshot. Live financial statement enrichment
uses Yahoo Finance on request, never fabricates observations, and can be exported.
"""
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ACCT 3343 | Accounting & Short Analysis", page_icon="📉", layout="wide")
DATA_PATH = Path(__file__).with_name("universe.csv")
NUMBERS = ["market_cap", "price", "trailing_pe", "forward_pe", "price_to_book", "ev_to_ebitda",
           "price_to_sales", "debt_to_equity", "return_on_equity", "revenue_growth", "profit_margin",
           "net_income", "operating_cash_flow", "total_assets", "current_assets", "current_liabilities",
           "total_liabilities", "total_equity", "gross_profit", "total_revenue", "ebit", "working_capital",
           "retained_earnings", "total_debt", "cash", "shares_outstanding", "market_value_equity",
           "sta", "gpa", "book_to_market", "altman_z", "free_cash_flow", "price_to_sales",
           "operating_margin", "earnings_growth", "change_pct", "total_cash"]
META = ["ticker", "name", "sector", "exchange", "quote_type", "statement_asof", "info_asof_utc",
        "verification_status", "statement_status", "statement_error"]
ALIASES = {"symbol": "ticker", "Ticker": "ticker", "shortName": "name", "longName": "name",
           "marketCap": "market_cap", "trailingPE": "trailing_pe", "forwardPE": "forward_pe",
           "priceToBook": "price_to_book", "enterpriseToEbitda": "ev_to_ebitda",
           "priceToSalesTrailing12Months": "price_to_sales", "operatingCashFlow": "operating_cash_flow",
           "netIncome": "net_income", "totalAssets": "total_assets", "grossProfit": "gross_profit",
           "totalRevenue": "total_revenue", "freeCashflow": "free_cash_flow"}
EXCLUDED_SECTORS = {"Financials", "Financial Services", "Real Estate"}


def normalize(frame):
    d = frame.rename(columns=ALIASES).copy()
    d = d.loc[:, ~d.columns.duplicated()]
    if "ticker" not in d.columns:
        raise ValueError("Data must include ticker.")
    d["ticker"] = d["ticker"].astype("string").str.strip().str.upper()
    d = d[d["ticker"].str.fullmatch(r"[A-Z][A-Z0-9.\-]{0,12}", na=False)]
    d = d.drop_duplicates("ticker", keep="first").reset_index(drop=True)
    for c in NUMBERS:
        if c not in d:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    for c in META:
        if c not in d:
            d[c] = pd.NA
        d[c] = d[c].astype("string")
    d["sector"] = d["sector"].fillna("Unknown")
    d["name"] = d["name"].fillna(d["ticker"])
    return d


@st.cache_data(show_spinner=False)
def load_saved():
    return normalize(pd.read_csv(DATA_PATH))


def safe_ratio(top, bottom, positive=False):
    a = pd.to_numeric(top, errors="coerce")
    b = pd.to_numeric(bottom, errors="coerce")
    valid = (b > 0) if positive else (b != 0)
    return (a / b.where(valid)).replace([np.inf, -np.inf], np.nan)


def compute_accounting(source):
    """Annual accounting factors, with null propagation and economic denominator guards."""
    d = normalize(source)
    assets = d["total_assets"].where(d["total_assets"] > 0)
    d["sta"] = (d["net_income"] - d["operating_cash_flow"]) / assets
    d["gpa"] = d["gross_profit"] / assets
    book = d["total_equity"].where(d["total_equity"] > 0)
    market = d["market_cap"].where(d["market_cap"] > 0)
    d["book_to_market"] = book / market
    # Original public-manufacturing Altman Z: 1.2 WC/TA + 1.4 RE/TA +
    # 3.3 EBIT/TA + 0.6 market cap / total liabilities + revenue / assets.
    # This is shown only for non-financial, non-real-estate sectors as context.
    liabilities = d["total_liabilities"].where(d["total_liabilities"] > 0)
    parts = [1.2 * d["working_capital"] / assets,
             1.4 * d["retained_earnings"] / assets,
             3.3 * d["ebit"] / assets,
             0.6 * market / liabilities,
             d["total_revenue"] / assets]
    z = sum(parts)
    d["altman_z"] = z.where(~d["sector"].isin(EXCLUDED_SECTORS))
    return d


def sector_relative(d, column, min_peers=5, positive=False):
    values = d[column].where(d[column] > 0) if positive else d[column]
    counts = values.groupby(d["sector"]).transform("count")
    means = values.groupby(d["sector"]).transform("mean").where(counts >= min_peers)
    medians = values.groupby(d["sector"]).transform("median").where(counts >= min_peers)
    demean = values - means
    return demean, medians, counts


def screen(data, opts):
    """Industry-benchmark denominator is the filtered screening population."""
    d = compute_accounting(data)
    for k in ["sta", "gpa", "book_to_market"]:
        dm, med, ct = sector_relative(d, k, opts["min_peers"], False)
        d[k + "_demeaned"] = dm
        d[k + "_sector_median"] = med
        d[k + "_peers"] = ct
    for k in ["trailing_pe", "price_to_book", "ev_to_ebitda", "price_to_sales"]:
        _, med, ct = sector_relative(d, k, opts["min_peers"], True)
        d[k + "_sector_median"] = med
        d[k + "_peers"] = ct
        d[k + "_premium_pct"] = 100 * (safe_ratio(d[k].where(d[k] > 0), med, True) - 1)
    # SELL analyses: abnormal accruals, excess peer-multiples, distressed balance sheet.
    # The distress and CFO checks only evaluate when raw statement data is available.
    d["flag_sta"] = d["sta_demeaned"].gt(opts["sta_excess"])
    d["flag_valuation"] = (d["trailing_pe_premium_pct"].gt(opts["premium"]) |
                           d["price_to_book_premium_pct"].gt(opts["premium"]))
    d["flag_distress"] = d["altman_z"].lt(opts["z_limit"])
    # Positive earnings with no matching cash, or nonpositive CFO, count as cash concerns.
    d["flag_cash"] = d["net_income"].gt(0) & (
        d["operating_cash_flow"].lt(d["net_income"] * opts["cash_ratio"]))
    mapping = {"flag_sta": d["sta_demeaned"].notna(),
               "flag_valuation": d["trailing_pe_premium_pct"].notna() |
                                 d["price_to_book_premium_pct"].notna(),
               "flag_distress": d["altman_z"].notna(),
               "flag_cash": d["net_income"].gt(0) & d["operating_cash_flow"].notna()}
    enabled = [c for c in mapping if c in opts["sell_methods"]]
    d["sell_flags"] = sum(d[c].astype(int) for c in enabled) if enabled else 0
    d["sell_coverage"] = sum(mapping[c].astype(int) for c in enabled) if enabled else 0
    # Cannot substantiate an accounting-driven short using only a price multiple.
    accounting_available = (d["sta"].notna() | d["operating_cash_flow"].notna() |
                            d["altman_z"].notna())
    d["sell_ready"] = accounting_available & (d["sell_coverage"] >= opts["min_sell_coverage"])
    d["sell_candidate"] = (d["sell_ready"] &
                           d["sell_flags"].ge(opts["min_red_flags"]))
    d["valuation_watch"] = d["flag_valuation"] & ~d["sell_candidate"]
    # BUY = profitability and book-value screening, NOT the inverse of SELL STA/Z/valuation.
    d["buy_ready"] = d["gpa_demeaned"].notna() & d["book_to_market_demeaned"].notna()
    d["buy_candidate"] = (d["buy_ready"] &
                          d["gpa_demeaned"].ge(opts["min_gpa_excess"]) &
                          d["book_to_market_demeaned"].ge(opts["min_bm_excess"]))
    d["buy_rank"] = (d["gpa_demeaned"].rank(pct=True, na_option="keep") +
                     d["book_to_market_demeaned"].rank(pct=True, na_option="keep")) / 2
    d["label"] = np.select([d["sell_candidate"], d["buy_candidate"], d["valuation_watch"]],
                           ["SELL candidate", "BUY candidate", "Valuation watch"], default="Unflagged")
    d["sell_explanation"] = d.apply(lambda r: "; ".join(
        label for key, label in [("flag_sta", "elevated industry-adjusted accruals"),
                                ("flag_valuation", "high peer-relative multiples"),
                                ("flag_distress", "low Altman Z"),
                                ("flag_cash", "weak operating-cash conversion")]
        if key in enabled and bool(r[key])) or "No selected SELL red flags", axis=1)
    return d


def _pick(statement, *labels):
    if statement is None or statement.empty:
        return np.nan
    # Use the latest published annual reporting column, not annualized quarterly values.
    for label in labels:
        if label in statement.index:
            v = pd.to_numeric(statement.loc[label].iloc[0], errors="coerce")
            if pd.notna(v) and np.isfinite(v):
                return float(v)
    return np.nan


@st.cache_data(ttl=900, show_spinner=False)
def fetch_one(symbol):
    """Best-effort Yahoo annual data fetch; API failures never imply delisting.

    The 15-minute cache permits retrying transient Yahoo blocks. Only explicit
    EQUITY quote metadata is considered an equity verification.
    """
    import yfinance as yf
    response = {"ticker": symbol, "verification_status": "Not verified",
                "statement_status": "Unavailable", "statement_error": pd.NA,
                "info_asof_utc": datetime.now(timezone.utc).isoformat()}
    stock = yf.Ticker(symbol)
    info = {}
    errors = []
    # get_info() commonly fails on Streamlit Cloud when Yahoo throttles a request.
    # Treat that as an API issue, not as a statement that 3M/AbbVie is delisted.
    try:
        info = stock.get_info() or {}
    except Exception as exc:
        errors.append(f"Quote lookup: {type(exc).__name__}: {exc}")
    typ = str(info.get("quoteType") or "").upper()
    # yfinance returns sparse information for some valid listed companies.
    # Alternative metadata can verify the equity type without relying on info.
    if not typ:
        try:
            metadata = stock.get_history_metadata() or {}
            typ = str(metadata.get("instrumentType") or metadata.get("quoteType") or "").upper()
        except Exception as exc:
            errors.append(f"History metadata: {type(exc).__name__}: {exc}")
    if typ == "EQUITY":
        response["verification_status"] = "Yahoo equity verified"
    elif typ and typ not in {"UNKNOWN", "NONE"}:
        response["verification_status"] = f"Reported non-equity type: {typ}"
        response["statement_error"] = "; ".join(errors)[:350] if errors else pd.NA
        return response
    # Empty metadata is NOT evidence of a fake ticker; try annual statements.
    response.update({"name": info.get("longName") or info.get("shortName") or symbol,
                     "sector": info.get("sector") or pd.NA,
                     "exchange": info.get("exchange") or pd.NA,
                     "quote_type": typ or pd.NA,
                     "market_cap": info.get("marketCap"),
                     "price": info.get("currentPrice") if info.get("currentPrice") is not None else info.get("regularMarketPrice"),
                     "trailing_pe": info.get("trailingPE"),
                     "forward_pe": info.get("forwardPE"),
                     "price_to_book": info.get("priceToBook"),
                     "price_to_sales": info.get("priceToSalesTrailing12Months"),
                     "ev_to_ebitda": info.get("enterpriseToEbitda")})
    if not info:
        try:
            fi = stock.fast_info
            price = fi.get("last_price")
            mcap = fi.get("market_cap")
            if price is not None:
                response["price"] = price
            if mcap is not None:
                response["market_cap"] = mcap
        except Exception as exc:
            errors.append(f"Fast quote: {type(exc).__name__}: {exc}")

    try:
        bs, inc, cf = stock.balance_sheet, stock.income_stmt, stock.cashflow
        if any(x is None or x.empty for x in (bs, inc, cf)):
            errors.append("One or more Yahoo annual statements are empty")
        else:
            # Find the latest matching annual period instead of mixing fiscal years.
            date_cols = []
            for statement in (bs, inc, cf):
                date_cols.append({pd.Timestamp(col).normalize(): col for col in statement.columns})
            common = set(date_cols[0]) & set(date_cols[1]) & set(date_cols[2])
            if not common:
                errors.append("No shared fiscal year across the three statements")
            else:
                asof = max(common)
                bs, inc, cf = (frame[[columns[asof]]] for frame, columns in
                               zip((bs, inc, cf), date_cols))
                response["statement_asof"] = str(asof.date())
                response.update({
                    "net_income": _pick(inc, "Net Income", "Net Income Common Stockholders"),
                    "operating_cash_flow": _pick(cf, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
                    "total_assets": _pick(bs, "Total Assets"),
                    "total_liabilities": _pick(bs, "Total Liabilities Net Minority Interest", "Total Liabilities"),
                    "total_equity": _pick(bs, "Stockholders Equity", "Total Equity Gross Minority Interest"),
                    "current_assets": _pick(bs, "Current Assets"),
                    "current_liabilities": _pick(bs, "Current Liabilities"),
                    "gross_profit": _pick(inc, "Gross Profit"),
                    "total_revenue": _pick(inc, "Total Revenue", "Operating Revenue"),
                    "ebit": _pick(inc, "EBIT", "Operating Income"),
                    "retained_earnings": _pick(bs, "Retained Earnings"),
                    "working_capital": _pick(bs, "Working Capital"),
                    "total_debt": _pick(bs, "Total Debt")})
                if pd.isna(response["working_capital"]):
                    x, y = response["current_assets"], response["current_liabilities"]
                    if pd.notna(x) and pd.notna(y):
                        response["working_capital"] = x - y
                response["statement_status"] = (
                    "Annual statements retrieved"
                    if all(pd.notna(response[k]) for k in ("total_assets", "net_income", "operating_cash_flow"))
                    else "Partial annual statements")
                if typ != "EQUITY":
                    response["verification_status"] = "Statements found; equity type unconfirmed"
    except Exception as exc:
        errors.append(f"Annual statements: {type(exc).__name__}: {exc}")
    if errors:
        response["statement_error"] = "; ".join(errors)[:500]
        if response["verification_status"] == "Not verified":
            response["verification_status"] = "Yahoo lookup unavailable / inconclusive"
    return response


def merge_fetch(base, rows):
    d = normalize(base).set_index("ticker")
    for r in rows:
        symbol = r["ticker"]
        if symbol not in d.index:
            continue
        # Preserve previously verified status on transient failures, and never
        # discard usable financial fields merely because quote type is absent.
        previous = d.at[symbol, "verification_status"]
        if (previous == "Yahoo equity verified" and
                r.get("verification_status") != "Yahoo equity verified"):
            r = {**r, "verification_status": previous}
        for k, value in r.items():
            if k == "ticker" or pd.isna(value):
                continue
            if k not in d.columns:
                d[k] = np.nan if k in NUMBERS else pd.NA
            d.at[symbol, k] = value
    return normalize(d.reset_index())


def pretty(x, percent=False, money=False):
    if pd.isna(x):
        return "N/A"
    if money:
        return f"${x/1e9:,.2f}B"
    return f"{x*100:+.2f}%" if percent else f"{x:,.2f}"




def display_table(rows, columns, names, percent_columns=(), decimals=2):
    """Readable formatting only: the underlying screen retains full precision."""
    view = rows.loc[:, columns].copy().rename(columns=names)
    for original in percent_columns:
        label = names.get(original, original)
        if label in view:
            view[label] = pd.to_numeric(view[label], errors="coerce") * 100
    numeric = view.select_dtypes(include="number").columns
    for label in numeric:
        if label not in ("Red flags", "Signals observed", "STA peers", "GPA peers", "B/M peers"):
            view[label] = view[label].round(decimals)
    st.dataframe(view, hide_index=True, use_container_width=True)


def explain_company(row):
    """Draw conclusions only from observable evidence; distinguish clues and claims."""
    reasons = []
    if pd.notna(row["sta_demeaned"]):
        reasons.append(f"STA is {row['sta'] * 100:.2f}% versus an estimated sector mean of "
                       f"{(row['sta'] - row['sta_demeaned']) * 100:.2f}% "
                       f"({row['sta_peers']:.0f} usable sector observations).")
    for field, label in [("trailing_pe_premium_pct", "P/E"), ("price_to_book_premium_pct", "P/B")]:
        if pd.notna(row[field]):
            reasons.append(f"{label} trades {row[field]:+.1f}% versus its positive-multiple sector median.")
    if pd.notna(row["altman_z"]):
        reasons.append(f"Original Altman Z is {row['altman_z']:.2f}; it indicates distress exposure, not overvaluation.")
    if pd.notna(row["net_income"]) and pd.notna(row["operating_cash_flow"]):
        reasons.append(f"Annual net income is {pretty(row['net_income'], money=True)} and operating cash flow "
                       f"is {pretty(row['operating_cash_flow'], money=True)}.")
    if row["sell_candidate"]:
        headline = f"**Potential SELL research candidate — {int(row['sell_flags'])} selected red flags.** "
        conclusion = ("The combination of observed signals warrants further scrutiny of earnings quality, "
                      "valuation and/or financial strength. It is not proof the shares are overpriced. ")
        counter = "Counterargument: growth prospects, one-off accounting items, and industry differences may justify some signals."
    elif row["buy_candidate"]:
        headline = "**Potential BUY research candidate — independent quality/value screen.** "
        conclusion = (f"Gross profitability is {row['gpa']*100:.2f}% of assets "
                      f"({row['gpa_demeaned']*100:+.2f} percentage points relative to sector mean), "
                      f"while book-to-market is {row['book_to_market']:.2f} "
                      f"({row['book_to_market_demeaned']:+.2f} relative to sector mean). "
                      "Both are observable above the selected benchmarks. ")
        counter = "Counterargument: apparent value can reflect deteriorating fundamentals or accounting differences."
    elif row["valuation_watch"]:
        headline = "**Preliminary valuation watch — NOT a confirmed SELL.** "
        conclusion = "Peer multiples look elevated, but available accounting evidence is not sufficient for a short thesis. "
        counter = "Counterargument: superior expected growth or profitability may warrant a valuation premium."
    else:
        headline = "**No supported BUY/SELL classification under current filters.** "
        conclusion = "This is not a recommendation to hold: the evidence is mixed, unavailable, or outside the selected thresholds. "
        counter = "Revisit the choice of sector peers and data coverage before drawing an investment conclusion."
    text = headline + conclusion + " ".join(reasons) + " " + counter
    if pd.notna(row["statement_asof"]):
        text += f" Annual financial statements are dated {row['statement_asof']}; market prices may be from a different date."
    return text

st.markdown("""<style>.block-container{padding-top:1.6rem;max-width:1520px}
[data-testid='stMetric']{border:1px solid rgba(120,120,120,.18);padding:12px;border-radius:10px}
</style>""", unsafe_allow_html=True)
st.title("ACCT 3343 · Accounting-Based Short Analysis")
st.caption("Industry-adjusted earnings quality, valuation, distress risk, and an independent profitability/value BUY comparison")

try:
    base = load_saved()
except Exception as exc:
    st.error(f"Cannot open universe.csv: {exc}")
    st.stop()

if "enriched" not in st.session_state:
    st.session_state.enriched = {}

with st.sidebar:
    st.header("Universe and financial statements")
    incoming = st.file_uploader("Optional: upload an enriched universe.csv", type="csv")
    if incoming is not None:
        try:
            base = normalize(pd.read_csv(BytesIO(incoming.getvalue())))
        except Exception as exc:
            st.error(f"Invalid uploaded CSV: {exc}")
    st.caption("Saved rows are historical. Explicit Yahoo EQUITY metadata confirms a listing; API failures and missing quote types remain unverified, never automatically invalid.")
    available = base["ticker"].tolist()
    chosen = st.multiselect("Tickers for Yahoo statement verification", available,
                            default=available[:min(10, len(available))], max_selections=75)
    if st.button("Verify and retrieve annual statements", type="primary", use_container_width=True):
        if not chosen:
            st.warning("Choose at least one ticker to verify.")
        else:
            bar = st.progress(0)
            for i, symbol in enumerate(chosen, 1):
                st.session_state.enriched[symbol] = fetch_one(symbol)
                bar.progress(i / len(chosen), text=f"Retrieved {i} of {len(chosen)}")
            st.success("Requests completed. Check the audit results; attempts are not proof of successful retrieval.")
    if st.button("Retry failed Yahoo lookups", use_container_width=True):
        fetch_one.clear()
        st.info("Yahoo cache cleared. Select the tickers and press Verify again.")
    if st.button("Clear this session's fetched records"):
        st.session_state.enriched = {}
        st.rerun()
    st.divider()
    st.header("Screening population")
    sector_options = sorted(base["sector"].dropna().unique().tolist())
    sector_selected = st.multiselect("Sectors", sector_options,
        default=[s for s in sector_options if s not in EXCLUDED_SECTORS])
    min_cap = st.slider("Minimum market cap ($ billions)", 0.0, 100.0, 0.0, step=0.5)
    include_unknown_cap = st.checkbox("Include unknown market caps", value=False)
    text_filter = st.text_input("Company/ticker search")
    st.divider()
    st.header("SELL: red-flag criteria")
    min_peers = st.slider("Minimum sector peers per measure", 2, 30, 5)
    sta_excess = st.slider("STA above sector mean by", 0.00, 0.30, 0.04, 0.01,
                           help="STA=(net income−operating cash flow)/total assets; demeaned STA = STA − mean(STA in same sector).")
    premium = st.slider("Valuation premium over sector median (%)", 0, 200, 35, 5)
    z_limit = st.slider("Altman Z below (distress)", 0.5, 4.0, 1.81, 0.05)
    cash_ratio = st.slider("Cash flow / positive net income below", 0.0, 1.5, 0.8, 0.05)
    options_labels = {"flag_sta": "High sector-demeaned STA", "flag_valuation": "Rich sector-relative valuation",
                      "flag_distress": "Low Altman Z", "flag_cash": "Low operating-cash conversion"}
    methods = st.multiselect("Active SELL signals", list(options_labels), default=list(options_labels),
                             format_func=lambda x: options_labels[x])
    red_flags = st.slider("Minimum red flags", 1, max(1, len(methods)), min(2, max(1, len(methods))))
    coverage_min = st.slider("Minimum observable SELL signals", 1, max(1, len(methods)), min(2, max(1, len(methods))))
    st.divider()
    st.header("BUY: independent ranking")
    gpa_excess = st.slider("GPA minus sector mean at least", -0.20, 0.30, 0.02, 0.01)
    bm_excess = st.slider("B/M minus sector mean at least", -1.00, 2.00, 0.0, 0.05)
    only_verified = st.checkbox("Show only Yahoo-verified companies", value=False)

combined = merge_fetch(base, list(st.session_state.enriched.values())) if st.session_state.enriched else base
if incoming is not None:
    st.caption("Using uploaded CSV, with any successful session lookups merged by ticker.")
else:
    st.caption("Using repository universe.csv, with any successful session lookups merged by ticker. Saved snapshot prices are not live.")

population = combined[combined["sector"].isin(sector_selected)].copy()
cap_mask = population["market_cap"].ge(min_cap * 1e9)
if include_unknown_cap:
    cap_mask |= population["market_cap"].isna()
population = population[cap_mask]
if only_verified:
    population = population[population["verification_status"].eq("Yahoo equity verified")]
if text_filter:
    mask = (population["ticker"].str.contains(text_filter, case=False, regex=False, na=False) |
            population["name"].str.contains(text_filter, case=False, regex=False, na=False))
    population = population[mask]
opts = {"min_peers": min_peers, "sta_excess": sta_excess, "premium": premium,
        "z_limit": z_limit, "cash_ratio": cash_ratio, "sell_methods": methods,
        "min_red_flags": red_flags, "min_sell_coverage": coverage_min,
        "min_gpa_excess": gpa_excess, "min_bm_excess": bm_excess}
result = screen(population, opts)

if not methods:
    st.warning("Enable a SELL signal to generate accounting-based SELL candidates.")
if (combined["verification_status"] == "Yahoo equity verified").sum() == 0:
    st.warning("**Verification pending:** the saved 903-ticker list has not been checked against current exchange listings. Use Yahoo statement verification and export the results before presenting ticker validity as confirmed.")
if result["sta"].notna().sum() == 0:
    st.info("**Raw financial statement fields are missing from this saved dataset.** The accounting-driven SELL and BUY lists will be limited or empty until you retrieve Yahoo statements or upload an enriched CSV. Valuation-watch names are preliminary only.")

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Rows in saved universe", f"{len(base):,}")
k2.metric("Population after filters", f"{len(result):,}")
k3.metric("Verified equities", f"{combined['verification_status'].eq('Yahoo equity verified').sum():,}")
k4.metric("SELL candidates", f"{result['sell_candidate'].sum():,}")
k5.metric("BUY candidates", f"{result['buy_candidate'].sum():,}")

sell_tab, buy_tab, exploration_tab, methodology_tab, quality_tab = st.tabs([
    "SELL candidates", "BUY candidates", "Charts & company conclusions", "Formulas & methodology", "Data audit & export"])

with sell_tab:
    st.subheader("Accounting-supported potential SELL candidates")
    st.caption("Requires financial-statement coverage and at least the selected number of red flags. A rich multiple alone cannot qualify.")
    cols = ["ticker", "name", "sector", "sell_flags", "sell_coverage", "sta", "sta_demeaned",
            "trailing_pe_premium_pct", "price_to_book_premium_pct", "altman_z", "operating_cash_flow",
            "net_income", "sell_explanation", "verification_status"]
    sells = result[result["sell_candidate"]].sort_values(["sell_flags", "sta_demeaned"], ascending=False)
    display_table(sells, ["ticker", "name", "sector", "sell_flags", "sell_coverage",
        "sta", "sta_demeaned", "sta_peers", "trailing_pe_premium_pct",
        "price_to_book_premium_pct", "altman_z", "sell_explanation"],
        {"ticker": "Ticker", "name": "Company", "sector": "Sector", "sell_flags": "Red flags",
         "sell_coverage": "Signals observed", "sta": "STA (%)", "sta_demeaned": "Adjusted STA (pp)",
         "sta_peers": "STA peers", "trailing_pe_premium_pct": "P/E premium (%)",
         "price_to_book_premium_pct": "P/B premium (%)", "altman_z": "Altman Z",
         "sell_explanation": "Evidence"}, percent_columns=["sta", "sta_demeaned"])
    if len(sells) and (sells["sta_peers"].fillna(0) < 10).any():
        st.warning("Some flagged firms have fewer than 10 usable sector STA peers. Interpret sector averages cautiously.")
    st.subheader("Preliminary valuation watchlist · NOT accounting-confirmed shorts")
    watch = result[result["valuation_watch"]].sort_values("trailing_pe_premium_pct", ascending=False)
    display_table(watch, ["ticker", "name", "sector", "trailing_pe_premium_pct",
        "price_to_book_premium_pct", "sta", "statement_status"],
        {"ticker": "Ticker", "name": "Company", "sector": "Sector",
         "trailing_pe_premium_pct": "P/E premium (%)", "price_to_book_premium_pct": "P/B premium (%)",
         "sta": "STA (%)", "statement_status": "Statement status"}, percent_columns=["sta"])
    st.download_button("Download SELL and watchlist observations", result.to_csv(index=False).encode(),
                       "acct3343_short_screen.csv", "text/csv")

with buy_tab:
    st.subheader("Independent BUY methodology · profitability + balance-sheet value")
    st.caption("BUY is NOT low STA or high Altman Z. It independently requires above-benchmark gross profitability and book-to-market.")
    buys = result[result["buy_candidate"]].sort_values("buy_rank", ascending=False)
    display_table(buys, ["ticker", "name", "sector", "gpa", "gpa_demeaned", "gpa_peers",
        "book_to_market", "book_to_market_demeaned", "book_to_market_peers", "buy_rank", "statement_asof"],
        {"ticker": "Ticker", "name": "Company", "sector": "Sector", "gpa": "Gross profit/assets (%)",
         "gpa_demeaned": "GPA above sector (pp)", "gpa_peers": "GPA peers",
         "book_to_market": "Book-to-market", "book_to_market_demeaned": "B/M minus sector mean",
         "book_to_market_peers": "B/M peers", "buy_rank": "Rank (0–1)", "statement_asof": "Fiscal year end"},
        percent_columns=["gpa", "gpa_demeaned"])
    if len(buys) and ((buys["gpa_peers"].fillna(0) < 10) | (buys["book_to_market_peers"].fillna(0) < 10)).any():
        st.warning("Some BUY candidates have fewer than 10 usable sector observations. Sector-relative results are provisional.")

with exploration_tab:
    x1,x2 = st.columns(2)
    with x1:
        chart_data = result.dropna(subset=["trailing_pe_premium_pct", "sta_demeaned"])
        if len(chart_data):
            fig = px.scatter(chart_data, x="trailing_pe_premium_pct", y="sta_demeaned", color="label",
                             hover_name="ticker", hover_data=["sector", "name"],
                             title="Peer valuation premium vs sector-demeaned STA")
            fig.add_hline(y=sta_excess, line_dash="dash")
            fig.add_vline(x=premium, line_dash="dash")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Valuation-vs-STA scatter requires annual cash-flow statements.")
    with x2:
        bars = result.groupby("sector", dropna=False).agg(companies=("ticker", "size"),
            sell=("sell_candidate", "sum"), buy=("buy_candidate", "sum"),
            watch=("valuation_watch", "sum")).reset_index()
        if len(bars):
            fig = px.bar(bars, x="sector", y=["sell", "buy", "watch"], barmode="group",
                         title="Screen outcomes by sector", hover_data=["companies"])
            st.plotly_chart(fig, use_container_width=True)
    st.subheader("Company-specific investment conclusion")
    if len(result):
        ticker = st.selectbox("Select an included company", result["ticker"].tolist())
        r = result.loc[result["ticker"] == ticker].iloc[0]
        st.markdown(f"### {r['name']} ({ticker}) · {r['sector']}")
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("STA", pretty(r["sta"], percent=True))
        d2.metric("Industry-adjusted STA", pretty(r["sta_demeaned"], percent=True))
        d3.metric("Gross profitability", pretty(r["gpa"], percent=True))
        d4.metric("Book-to-market", pretty(r["book_to_market"]))
        st.markdown(explain_company(r))
        if pd.notna(r["sta_demeaned"]):
            st.caption(f"STA sector benchmark: {(r['sta'] - r['sta_demeaned'])*100:.2f}% "
                       f"using {int(r['sta_peers'])} observations. "
                       "Comparisons change when universe filters change.")
        st.write("**Evidence:**", r["sell_explanation"])
        st.write(f"Reported net income: {pretty(r['net_income'], money=True)} · "
                 f"Operating cash flow: {pretty(r['operating_cash_flow'], money=True)} · "
                 f"Assets: {pretty(r['total_assets'], money=True)}")
        st.write(f"P/E sector premium: {pretty(r['trailing_pe_premium_pct'])}% · "
                 f"P/B sector premium: {pretty(r['price_to_book_premium_pct'])}% · "
                 f"Altman Z: {pretty(r['altman_z'])}")
        st.caption(f"Annual statement date: {r['statement_asof'] if pd.notna(r['statement_asof']) else 'not supplied'} · "
                   f"Verification: {r['verification_status'] if pd.notna(r['verification_status']) else 'not checked'}")
        st.write("**Risks and counterarguments:** accounting timing differences, sector heterogeneity, "
                 "high expected growth, stale market data, annual filing lag, short-sale borrowing cost, "
                 "and potentially asymmetric losses. These are research hypotheses, not price targets.")
    else:
        st.info("No companies satisfy the universe filters.")

with methodology_tab:
    st.markdown("""
    ### How to interpret this app
    **Potential SELL** combines observed earnings-quality, relative-valuation, cash-conversion and/or
    distress-warning signals. **Potential BUY** is an independent profitability-plus-book-value
    screen; it is not the reverse of SELL. These are research candidates, not verified mispricings.
    Peer benchmarks reflect the currently filtered, available-data population, not all 903 tickers.
    Sector averages can be unstable when statement coverage is thin or accruals contain outliers.

    ### Course formulas and interpretations
    **SELL — four distinct warning signs:**
    1. **Scaled total accruals (STA)** = `(annual net income − annual operating cash flow) / annual total assets`. Higher positive STA may mean profits are not supported by cash. **Sector demeaning** = company STA minus the arithmetic mean STA among companies in its sector with usable annual statements; this helps compare earnings quality against the group rather than across unrelated industries. Sector means need the minimum number of valid peers.
    2. **Peer valuation** = `(positive company multiple / median positive sector multiple − 1) × 100`. A premium in **P/E or P/B** is an overvaluation clue, not proof of it; missing or negative multiples cannot be ranked as if positive.
    3. **Altman Z (original public manufacturing specification)** = `1.2×(working capital/assets) + 1.4×(retained earnings/assets) + 3.3×(EBIT/assets) + 0.6×(market equity/total liabilities) + sales/assets`. This is an **insolvency-risk indicator**, not a valuation model; applicability is limited outside manufacturing and it is not used for financials or real estate.
    4. **Cash conversion** is flagged if annual net income is positive and operating cash flow is below the user-set fraction of net income.

    **BUY — genuinely different financial statement analysis:** gross profitability on assets **GPA** = `annual gross profit / annual total assets`; book-to-market **B/M** = `positive common equity / market capitalization`. Both are compared with their **sector arithmetic means**. A BUY needs both excesses to exceed the selected thresholds, which must have real statement inputs. Sector-relative ranks alone are not buy or sell orders.

    **Decision rules:** every active SELL method has a true/false flag only when its required data is measurable. Require at least N observed signals, at least N triggered red flags, and **at least one accounting measure** for a SELL. A missing value never becomes zero and can never independently trigger a flag. The valuation watchlist is kept separate. BUY requires both GPA and B/M observations. Filters, peer groups, and conclusions recalculate at every Streamlit rerun.

    **Peer groups:** sector membership is a coarse comparison. Sector-relative means use only rows remaining after sector, size, and ticker filters. Low sample size may make all scores unavailable. Financials and Real Estate are excluded by default because their accounting structure can make these rules misleading. All lookups use latest **annual** statements (not TTM / quarterly annualization); market capitalization/price may be at a different date.

    **No fabricated company data:** the supplied repository CSV was a partial historical snapshot. The app does not silently backfill assets, cash flows, market caps, or outdated listings. A company only receives the label *Yahoo equity verified* after Yahoo confirms quote type EQUITY. Yahoo failures, delistings, stale tickers, and missing annual statements are shown in the audit. Verification is point-in-time and cannot guarantee continued listing.
    """)

with quality_tab:
    coverage_cols = ["market_cap", "price", "trailing_pe", "price_to_book", "ev_to_ebitda",
                     "net_income", "operating_cash_flow", "total_assets", "gross_profit", "total_equity",
                     "sta", "sta_demeaned", "altman_z", "gpa", "book_to_market"]
    counts = pd.DataFrame({"field": coverage_cols, "populated": [int(result[k].notna().sum()) for k in coverage_cols]})
    counts["missing"] = len(result) - counts["populated"]
    st.subheader("Completeness and provenance of the currently filtered universe")
    st.dataframe(counts, hide_index=True, use_container_width=True)
    st.caption("If lookups say unavailable/inconclusive, Yahoo may be blocking cloud requests. Review statement_error, retry after a short delay, or upload a CSV exported from a successful local/Colab run.")
    st.caption(f"Rows: {len(result)} · unique tickers: {result['ticker'].nunique()} · "
               f"verified in full loaded dataset: {combined['verification_status'].eq('Yahoo equity verified').sum()} · "
               f"statement rows: {combined['statement_status'].eq('Annual statements retrieved').sum()}")
    st.dataframe(combined[["ticker", "name", "sector", "verification_status", "statement_status",
                           "statement_asof", "statement_error"]], hide_index=True, use_container_width=True)
    st.download_button("Download enriched universe.csv for GitHub", combined.to_csv(index=False).encode(),
                       "universe.csv", mime="text/csv",
                       help="Replace the repository universe.csv with this file to persist fetched rows. The app cannot push changes to GitHub automatically.")
    st.caption("This download is the persistent data snapshot. Replacing it on GitHub preserves your existing deployment path: app.py at the repository root. No background data refresh occurs.")

st.caption("Educational analysis only · Financial data: historical saved CSV and optional Yahoo Finance requests · Not personalized investment advice")
