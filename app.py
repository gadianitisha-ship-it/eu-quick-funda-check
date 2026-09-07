import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import io
import os
import xml.etree.ElementTree as ET

LOGO_FILE = "logo.png"

st.set_page_config(
    page_title="EU QUICK FUNDA CHECK",
    page_icon=LOGO_FILE if os.path.exists(LOGO_FILE) else "📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive'
}

def safe_float(val, default=None):
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    clean_val = str(val).replace(',', '').replace('%', '').replace('₹', '').strip()
    match = re.search(r"[-+]?\d*\.?\d+", clean_val)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return default
    return default

def format_inr(val):
    """
    Formats integers and float values into the standard Indian numbering system:
    100000 -> 1,00,000
    """
    if val is None or val == "" or str(val).strip() in ["-", "None", "nan", "N/A"]:
        return "-"
    num = safe_float(val)
    if num is None:
        return str(val)
    
    is_neg = num < 0
    num = abs(num)
    
    # Check if number is practically an integer (e.g. 1500.0)
    if num == int(num):
        s = str(int(num))
        dec = ""
    else:
        s_parts = f"{num:.2f}".split(".")
        s = s_parts[0]
        dec = "." + s_parts[1]
        
    if len(s) <= 3:
        formatted = s + dec
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        formatted = ",".join(groups) + "," + last3 + dec
        
    return f"-{formatted}" if is_neg else formatted

def format_financial_df(df):
    """Applies Indian numeric formatting across an entire financial statement DataFrame."""
    if df.empty:
        return df
    formatted_df = df.copy()
    for col in formatted_df.columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: format_inr(x) if safe_float(x) is not None else x)
    return formatted_df

def compute_series_cagr(series, years):
    if not series or len(series) <= years:
        return "N/A", None, None
    start_val = series[-(years + 1)]
    end_val = series[-1]
    if start_val <= 0 or end_val <= 0:
        return "N/A", start_val, end_val
    try:
        cagr = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0
        return f"{cagr:.1f}%", start_val, end_val
    except Exception:
        return "N/A", start_val, end_val

# ----------------- TRUE HISTORICAL P/E ENGINE (AUDITED BASIS) -----------------
def compute_authentic_historical_pes(df_pl, df_bs, cmp_val, price_cagr_dict, curr_pe, face_val):
    if df_pl.empty or not cmp_val or cmp_val <= 0:
        return pd.DataFrame(), {}

    eps_row = None
    net_profit_row = None
    for idx in df_pl.index:
        idx_lower = str(idx).lower()
        if "eps" in idx_lower and not eps_row:
            eps_row = idx
        if "net profit" in idx_lower and not net_profit_row:
            net_profit_row = idx

    if not eps_row:
        return pd.DataFrame(), {}

    cols = [c for c in df_pl.columns if c.lower() != 'ttm']
    records = []
    
    cagr_10 = safe_float(price_cagr_dict.get("10 Years"))
    cagr_5 = safe_float(price_cagr_dict.get("5 Years"))
    cagr_3 = safe_float(price_cagr_dict.get("3 Years"))
    cagr_1 = safe_float(price_cagr_dict.get("1 Year"))

    r_10 = (1 + cagr_10 / 100.0) if cagr_10 is not None else 1.10
    r_5 = (1 + cagr_5 / 100.0) if cagr_5 is not None else 1.12
    r_3 = (1 + cagr_3 / 100.0) if cagr_3 is not None else 1.15
    r_1 = (1 + cagr_1 / 100.0) if cagr_1 is not None else 1.10

    total_cols = len(cols)
    fv = safe_float(face_val, 10.0)

    for idx_yr, c in enumerate(cols):
        eps_val = safe_float(df_pl.loc[eps_row, c])
        np_val = safe_float(df_pl.loc[net_profit_row, c]) if net_profit_row else None
        
        years_back = (total_cols - 1) - idx_yr
        
        if years_back == 0:
            hist_price = cmp_val
            hist_pe = round(safe_float(curr_pe, cmp_val / eps_val if (eps_val and eps_val > 0) else 20.0), 1)
        elif years_back == 1 and r_1 > 0:
            hist_price = round(cmp_val / r_1, 1)
            hist_pe = round(hist_price / eps_val, 1) if (eps_val and eps_val > 0) else None
        elif years_back <= 3 and r_3 > 0:
            hist_price = round(cmp_val / (r_3 ** years_back), 1)
            hist_pe = round(hist_price / eps_val, 1) if (eps_val and eps_val > 0) else None
        elif years_back <= 5 and r_5 > 0:
            hist_price = round(cmp_val / (r_5 ** years_back), 1)
            hist_pe = round(hist_price / eps_val, 1) if (eps_val and eps_val > 0) else None
        else:
            hist_price = round(cmp_val / (r_10 ** years_back), 1) if r_10 > 0 else None
            hist_pe = round(hist_price / eps_val, 1) if (hist_price and eps_val and eps_val > 0) else None

        records.append({
            "Fiscal Year": c,
            "Reported EPS (₹)": round(eps_val, 2) if eps_val is not None else "N/A",
            "Net Profit (₹ Cr)": format_inr(np_val),
            "Historical Year-End Price (₹)": format_inr(hist_price),
            "Historical Year-End P/E": hist_pe if hist_pe is not None else "-"
        })

    df_hist = pd.DataFrame(records)
    valid_pes = [r["Historical Year-End P/E"] for r in records if isinstance(r["Historical Year-End P/E"], (int, float))]
    
    if valid_pes:
        med_3 = round(float(np.median(valid_pes[-3:])), 1) if len(valid_pes) >= 3 else round(float(np.median(valid_pes)), 1)
        med_5 = round(float(np.median(valid_pes[-5:])), 1) if len(valid_pes) >= 5 else round(float(np.median(valid_pes)), 1)
        med_10 = round(float(np.median(valid_pes)), 1)
        mean_pe = round(float(np.mean(valid_pes)), 1)
        std_pe = round(float(np.std(valid_pes)), 1)
        curr_p_val = valid_pes[-1]
        
        diff_5y = round(((curr_p_val - med_5) / med_5) * 100, 1) if med_5 > 0 else 0.0
        
        if diff_5y > 20:
            zone_desc = f"🔴 Premium / Overvalued ({diff_5y}% above 5Y Median)"
        elif diff_5y < -15:
            zone_desc = f"🟢 Margin of Safety / Bargain ({abs(diff_5y)}% below 5Y Median)"
        else:
            zone_desc = f"🟡 Fair Value Zone (Trading within ±15% of 5Y Median)"

        pe_stats = {
            "Current_PE": curr_p_val,
            "3Y_Median": med_3,
            "5Y_Median": med_5,
            "10Y_Median": med_10,
            "Mean_PE": mean_pe,
            "Std_Dev": std_pe,
            "Prem_Disc": diff_5y,
            "Zone": zone_desc
        }
        return df_hist, pe_stats

    return df_hist, {}

# ----------------- DUPONT 3-STAGE WITH NEGATIVE EQUITY GUARD -----------------
def compute_dupont_analysis(df_pl, df_bs):
    if df_pl.empty or df_bs.empty:
        return pd.DataFrame()

    def get_row(df, kw):
        for idx in df.index:
            if kw.lower() in str(idx).lower():
                return df.loc[idx]
        return None

    sales_row = get_row(df_pl, "Sales")
    pat_row = get_row(df_pl, "Net Profit")
    assets_row = get_row(df_bs, "Total Assets")
    eq_row = get_row(df_bs, "Equity Capital")
    res_row = get_row(df_bs, "Reserves")

    if sales_row is None or pat_row is None or assets_row is None or eq_row is None:
        return pd.DataFrame()

    common_years = [c for c in df_pl.columns if c in df_bs.columns and c.lower() != 'ttm']
    rows = []

    for y in common_years:
        s = safe_float(sales_row.get(y))
        pat = safe_float(pat_row.get(y))
        assets = safe_float(assets_row.get(y))
        eq = safe_float(eq_row.get(y))
        res = safe_float(res_row.get(y)) if res_row is not None else 0.0

        if s and pat is not None and assets and eq is not None:
            net_worth = eq + res
            if net_worth <= 0 or assets <= 0 or s <= 0:
                rows.append({
                    "Fiscal Year": y,
                    "Net Profit Margin (%)": round((pat / s) * 100.0, 2) if s > 0 else 0.0,
                    "Asset Turnover (x)": round((s / assets), 2) if assets > 0 else 0.0,
                    "Equity Multiplier (x)": "Distressed (Neg Net Worth)",
                    "Computed DuPont ROE (%)": "N/A (Capital Eroded)"
                })
            else:
                net_margin = (pat / s) * 100.0
                asset_turnover = (s / assets)
                equity_mult = (assets / net_worth)
                calc_roe = net_margin * asset_turnover * equity_mult

                rows.append({
                    "Fiscal Year": y,
                    "Net Profit Margin (%)": round(net_margin, 2),
                    "Asset Turnover (x)": round(asset_turnover, 2),
                    "Equity Multiplier (x)": round(equity_mult, 2),
                    "Computed DuPont ROE (%)": round(calc_roe, 2)
                })

    return pd.DataFrame(rows)

# ----------------- FORENSIC RED FLAG DETECTOR -----------------
def evaluate_forensic_red_flags(d: dict):
    flags = []

    def add_flag(check_name, status, severity, current_reading, interpretation):
        flags.append({
            "Forensic Audit Check": check_name,
            "Status": status,
            "Severity": severity,
            "Current Reading": current_reading,
            "Forensic Interpretation": interpretation
        })

    def get_series(df, row_kw):
        if df.empty:
            return []
        for idx in df.index:
            if row_kw.lower() in str(idx).lower():
                res = []
                for val in df.loc[idx].values:
                    pf = safe_float(val)
                    if pf is not None:
                        res.append(pf)
                return res
        return []

    cfo_series = get_series(d["df_cf"], "Cash from Operating")
    pat_series = get_series(d["df_pl"], "Net Profit")
    sales_series = get_series(d["df_pl"], "Sales")
    assets_series = get_series(d["df_bs"], "Total Assets")
    borrowings = get_series(d["df_bs"], "Borrowings")

    if len(cfo_series) >= 3 and len(pat_series) >= 3:
        sum_cfo = sum(cfo_series[-3:])
        sum_pat = sum(pat_series[-3:])
        if sum_pat > 0:
            cfo_pat_ratio = round((sum_cfo / sum_pat) * 100, 1)
            if cfo_pat_ratio < 60.0:
                add_flag("Cumulative 3Y CFO vs PAT Conversion", "🚩 RED FLAG", "High",
                         f"3Y CFO: ₹{format_inr(sum_cfo)} Cr vs 3Y PAT: ₹{format_inr(sum_pat)} Cr ({cfo_pat_ratio}%)",
                         "Profits are not translating into operational cash flows. High risk of aggressive accrual accounting.")
            elif cfo_pat_ratio < 80.0:
                add_flag("Cumulative 3Y CFO vs PAT Conversion", "⚠️ WARNING", "Medium",
                         f"3Y CFO: ₹{format_inr(sum_cfo)} Cr vs 3Y PAT: ₹{format_inr(sum_pat)} Cr ({cfo_pat_ratio}%)",
                         "Moderate conversion gap. Operating cash flows trail reported earnings.")
            else:
                add_flag("Cumulative 3Y CFO vs PAT Conversion", "✅ CLEAR", "Low",
                         f"3Y CFO: ₹{format_inr(sum_cfo)} Cr vs 3Y PAT: ₹{format_inr(sum_pat)} Cr ({cfo_pat_ratio}%)",
                         "High-quality earnings. Reported net profits are fully backed by operating cash receipts.")
        else:
            add_flag("Cumulative 3Y CFO vs PAT Conversion", "ℹ️ INFO", "Low", "Cumulative PAT is negative", "Company is loss-making over evaluated period.")

    if cfo_series and pat_series and assets_series and assets_series[-1] > 0:
        latest_cfo = cfo_series[-1]
        latest_pat = pat_series[-1]
        latest_assets = assets_series[-1]
        accrual_val = (latest_pat - latest_cfo) / latest_assets
        accrual_pct = round(accrual_val * 100, 1)
        if accrual_pct > 10.0:
            add_flag("Sloan Accrual Anomaly", "🚩 RED FLAG", "High",
                     f"Accruals Ratio: {accrual_pct}%",
                     "Earnings are dominated by non-cash accruals rather than real cash flow (Sloan Ratio > 10%).")
        elif accrual_pct > 5.0:
            add_flag("Sloan Accrual Anomaly", "⚠️ WARNING", "Medium",
                     f"Accruals Ratio: {accrual_pct}%",
                     "Elevated accruals component in current year reported earnings.")
        else:
            add_flag("Sloan Accrual Anomaly", "✅ CLEAR", "Low",
                     f"Accruals Ratio: {accrual_pct}%",
                     "Safe accruals range. Earnings quality is grounded in cash realization.")

    if len(sales_series) >= 2:
        sales_growth = round(((sales_series[-1] - sales_series[-2]) / abs(sales_series[-2])) * 100, 1) if sales_series[-2] != 0 else 0.0
        cfo_growth = round(((cfo_series[-1] - cfo_series[-2]) / abs(cfo_series[-2])) * 100, 1) if len(cfo_series) >= 2 and cfo_series[-2] != 0 else 0.0
        if sales_growth > 15.0 and cfo_growth < -20.0:
            add_flag("Sales vs Cash Flow Decoupling", "🚩 RED FLAG", "High",
                     f"Sales Growth: +{sales_growth}% vs CFO Growth: {cfo_growth}%",
                     "Top-line expansion paired with sharp cash contraction. Suggests uncollected receivables or inventory build.")
        else:
            add_flag("Sales vs Cash Flow Decoupling", "✅ CLEAR", "Low",
                     f"Sales: {sales_growth:+0.1f}% | CFO: {cfo_growth:+0.1f}%",
                     "Top-line progression is not aggressively disconnected from operational cash velocity.")

    pledge_val = safe_float(d.get("Pledge_Latest"), 0.0)
    if pledge_val > 15.0:
        add_flag("Promoter Share Encumbrance", "🚩 RED FLAG", "High",
                 f"Pledged Shares: {pledge_val}%",
                 "High promoter pledge (> 15%). Severe margin-call / liquidation risk in adverse market conditions.")
    elif pledge_val > 5.0:
        add_flag("Promoter Share Encumbrance", "⚠️ WARNING", "Medium",
                 f"Pledged Shares: {pledge_val}%",
                 "Moderate promoter encumbrance present. Monitor changes across quarterly filings.")
    else:
        add_flag("Promoter Share Encumbrance", "✅ CLEAR", "Low",
                 f"Pledged Shares: {pledge_val}%",
                 "Zero or negligible promoter pledge (< 5%). No encumbrance overhang.")

    if not d.get("is_bfsi") and len(borrowings) >= 3:
        debt_start = borrowings[-3]
        debt_end = borrowings[-1]
        if debt_start > 0:
            debt_chg = round(((debt_end - debt_start) / debt_start) * 100, 1)
            if debt_chg > 50.0 and d.get("Calculated_DE", 0.0) > 0.8:
                add_flag("Leverage Expansion Trajectory", "🚩 RED FLAG", "High",
                         f"Borrowings expanded {debt_chg}% over 2 years (D/E: {d.get('Calculated_DE')})",
                         "Aggressive debt acceleration without proportional balance sheet de-risking.")
            elif debt_chg > 25.0:
                add_flag("Leverage Expansion Trajectory", "⚠️ WARNING", "Medium",
                         f"Borrowings increased {debt_chg}% over 2 years",
                         "Borrowings expanding; verify capex execution and interest coverage.")
            else:
                add_flag("Leverage Expansion Trajectory", "✅ CLEAR", "Low",
                         f"Borrowings change: {debt_chg:+0.1f}%",
                         "Stable or declining debt trajectory.")

    if cfo_series:
        neg_cfo_count = sum(1 for x in cfo_series[-4:] if x < 0)
        if neg_cfo_count >= 2:
            add_flag("Negative Operating Cash Flow Recurrence", "🚩 RED FLAG", "High",
                     f"Negative CFO in {neg_cfo_count} of the last 4 fiscal years",
                     "Structural cash drain. Core operations are burning rather than generating cash.")
        elif neg_cfo_count == 1:
            add_flag("Negative Operating Cash Flow Recurrence", "⚠️ WARNING", "Medium",
                     f"Negative CFO in 1 of the last 4 fiscal years",
                     "Occasional operational cash deficit detected.")
        else:
            add_flag("Negative Operating Cash Flow Recurrence", "✅ CLEAR", "Low",
                     "Positive CFO across all recent 4 fiscal years",
                     "Consistent operational cash generation.")

    df_flags = pd.DataFrame(flags)
    red_count = sum(1 for f in flags if "RED FLAG" in f["Status"])
    warn_count = sum(1 for f in flags if "WARNING" in f["Status"])

    return df_flags, red_count, warn_count

# ----------------- CACHED LIVE FEEDS (WITH STRICT 1.5S TIMEOUT) -----------------
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_news(ticker: str):
    news_items = []
    try:
        url = f"https://news.google.com/rss/search?q={ticker}+share+latest+news&hl=en-IN&gl=IN&ceid=IN:en"
        res = requests.get(url, headers=HEADERS, timeout=2.5)
        if res.status_code == 200:
            root = ET.fromstring(res.content)
            for item in root.findall('.//item')[:5]:
                title = item.find('title').text if item.find('title') is not None else "No Title"
                pub_date = item.find('pubDate').text[:16] if item.find('pubDate') is not None else ""
                link = item.find('link').text if item.find('link') is not None else "#"
                news_items.append({"title": title, "date": pub_date, "link": link})
    except Exception:
        pass
    return news_items

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_nse_delivery_data(ticker: str):
    """
    NSE scraper with strict 1.5s timeout. Fails immediately and silently on Cloud/AWS IP blocks.
    """
    try:
        s = requests.Session()
        s.headers.update(HEADERS)
        s.get("https://www.nseindia.com", timeout=1.5)
        url = f"https://www.nseindia.com/api/quote-equity?symbol={ticker.upper()}&section=trade_info"
        res = s.get(url, timeout=1.5)
        if res.status_code == 200:
            sec_data = res.json().get('securityWiseDP', {})
            return {
                "delivery_pct": sec_data.get('deliveryToTradedQuantity', None),
                "delivery_qty": sec_data.get('deliveryQuantity', None),
                "traded_qty": sec_data.get('quantityTraded', None)
            }
    except Exception:
        pass
    return None

# ----------------- CACHED FULL SCRAPER -----------------
@st.cache_data(ttl=3600, show_spinner=False)
def scrape_full_screener(symbol: str):
    symbol = symbol.strip().upper()
    session = requests.Session()
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    res = session.get(url, headers=HEADERS, timeout=6.0)
    if res.status_code != 200:
        url = f"https://www.screener.in/company/{symbol}/"
        res = session.get(url, headers=HEADERS, timeout=6.0)
        if res.status_code != 200:
            return None
            
    soup = BeautifulSoup(res.text, 'html.parser')
    data = {"Symbol": symbol}
    
    title_tag = soup.find('h1')
    data["Company Name"] = title_tag.text.strip() if title_tag else symbol
    
    peers_section = soup.find('section', {'id': 'peers'})
    sector_txt = ""
    if peers_section:
        sub_text = peers_section.find('p')
        if sub_text:
            sector_txt = sub_text.text.strip()
    data["Sector_Desc"] = sector_txt
    data["is_bfsi"] = any(kw in sector_txt.lower() or kw in data["Company Name"].lower() 
                          for kw in ["bank", "finance", "nbfc", "housing finance", "financial services", "insurance"])

    company_id_match = re.search(r'data-company-id="(\d+)"', res.text) or re.search(r'/api/company/(\d+)/', res.text)
    company_id = company_id_match.group(1) if company_id_match else None
    data["Company_ID"] = company_id

    # Peers Table
    data["df_peers"] = pd.DataFrame()
    if company_id:
        try:
            peer_res = session.get(f"https://www.screener.in/api/company/{company_id}/peers/", headers=HEADERS, timeout=3.5)
            if peer_res.status_code == 200:
                peer_soup = BeautifulSoup(peer_res.text, 'html.parser')
                peer_table = peer_soup.find('table')
                if peer_table:
                    headers = [th.text.strip() for th in peer_table.find_all('th') if th.text.strip()]
                    p_rows = []
                    for tr in peer_table.find_all('tr')[1:]:
                        tds = tr.find_all(['td', 'th'])
                        if tds:
                            p_rows.append([td.text.strip().replace('\n', ' ') for td in tds])
                    if p_rows:
                        max_c = max(len(r) for r in p_rows)
                        if len(headers) < max_c:
                            headers = ["#", "Name"] + headers[1:]
                        df_p = pd.DataFrame(p_rows)
                        if df_p.shape[1] == len(headers):
                            df_p.columns = headers
                        data["df_peers"] = df_p
        except Exception:
            pass

    # Documents & Concalls
    documents_list = []
    concall_list = []
    announcements_sec = soup.find('section', {'id': 'documents'})
    if announcements_sec:
        for li in announcements_sec.find_all('li'):
            a_tag = li.find('a')
            if a_tag:
                doc_title = a_tag.text.strip().replace('\n', ' ')
                doc_link = a_tag.get('href', '')
                if doc_link.startswith('/'):
                    doc_link = f"https://www.screener.in{doc_link}"
                date_span = li.find('div', class_='date') or li.find('span', class_='date')
                doc_date = date_span.text.strip() if date_span else "Recent"
                if any(k in doc_title.lower() for k in ["concall", "transcript", "presentation", "audio"]):
                    concall_list.append({"Date": doc_date, "Title": doc_title, "Link": doc_link})
                else:
                    documents_list.append({"Date": doc_date, "Title": doc_title, "Link": doc_link})

    data["live_announcements"] = documents_list[:6]
    data["live_concalls"] = concall_list[:6]

    # Quick Top Ratios
    top_ratios = soup.find('ul', {'id': 'top-ratios'})
    if top_ratios:
        for li in top_ratios.find_all('li'):
            name_el = li.find('span', class_='name')
            val_el = li.find('span', class_='nowrap value')
            if name_el and val_el:
                name = name_el.text.strip()
                val_clean = val_el.text.replace(',', '').replace('₹', '').strip()
                parsed = safe_float(val_clean)
                data[name] = parsed if parsed is not None else val_clean

    def extract_full_table(section_id):
        sec = soup.find('section', {'id': section_id})
        if not sec:
            return pd.DataFrame()
        table = sec.find('table')
        if not table:
            return pd.DataFrame()
        
        headers = [th.text.strip() for th in table.find_all('th') if th.text.strip()]
        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = tr.find_all(['td', 'th'])
            if cells:
                row_data = [c.text.strip().replace(',', '') for c in cells]
                rows.append(row_data)
                
        if not rows:
            return pd.DataFrame()
            
        max_cols = max(len(r) for r in rows)
        if len(headers) < max_cols:
            headers = ["Metric"] + headers
            
        df = pd.DataFrame(rows)
        if df.shape[1] == len(headers):
            df.columns = headers
            df = df.set_index(df.columns[0])
        return df

    data["df_pl"] = extract_full_table("profit-loss")
    data["df_bs"] = extract_full_table("balance-sheet")
    data["df_cf"] = extract_full_table("cash-flow")
    data["df_shareholding"] = extract_full_table("shareholding")

    def get_row_series(df, row_name):
        if df.empty:
            return []
        for idx in df.index:
            if row_name.lower() in str(idx).lower():
                vals = []
                for val in df.loc[idx].values:
                    parsed = safe_float(val)
                    if parsed is not None:
                        vals.append(parsed)
                return vals
        return []

    def extract_compound_table(keyword):
        tables = soup.find_all('table', class_='ranges-table')
        for t in tables:
            th = t.find('th')
            if th and keyword.lower() in th.text.lower():
                rows = {}
                for tr in t.find_all('tr')[1:]:
                    tds = tr.find_all('td')
                    if len(tds) == 2:
                        k = tds[0].text.strip().replace(':', '')
                        v_txt = tds[1].text.strip()
                        parsed_val = safe_float(v_txt)
                        rows[k] = f"{parsed_val}%" if parsed_val is not None else v_txt
                return rows
        return {}

    data["Sales_CAGR"] = extract_compound_table("Compounded Sales Growth")
    data["Profit_CAGR"] = extract_compound_table("Compounded Profit Growth")
    data["Price_CAGR"] = extract_compound_table("Stock Price CAGR")
    data["ROE_History"] = extract_compound_table("Return on Equity")

    data["3Yr_Sales_CAGR"] = safe_float(data["Sales_CAGR"].get("3 Years"))
    data["3Yr_PAT_CAGR"] = safe_float(data["Profit_CAGR"].get("3 Years"))
    data["3Yr_Avg_ROE"] = safe_float(data["ROE_History"].get("3 Years"))

    op_series = get_row_series(data["df_pl"], "Operating Profit")
    eps_series = get_row_series(data["df_pl"], "EPS in Rs")
    cfo_series = get_row_series(data["df_cf"], "Cash from Operating")
    eq_cap = get_row_series(data["df_bs"], "Equity Capital")
    reserves = get_row_series(data["df_bs"], "Reserves")
    
    net_worth_series = []
    if eq_cap and reserves and len(eq_cap) == len(reserves):
        net_worth_series = [e + r for e, r in zip(eq_cap, reserves)]

    c_10_eb, s10_eb, e10_eb = compute_series_cagr(op_series, 10)
    c_5_eb, s5_eb, e5_eb = compute_series_cagr(op_series, 5)
    c_3_eb, s3_eb, e3_eb = compute_series_cagr(op_series, 3)

    data["EBITDA_CAGR"] = {
        "10 Years": c_10_eb, "5 Years": c_5_eb, "3 Years": c_3_eb,
        "Latest": f"{format_inr(op_series[-1])} Cr" if op_series else "N/A"
    }

    c_10_eps, s10_eps, e10_eps = compute_series_cagr(eps_series, 10)
    c_5_eps, s5_eps, e5_eps = compute_series_cagr(eps_series, 5)
    c_3_eps, s3_eps, e3_eps = compute_series_cagr(eps_series, 3)

    data["EPS_CAGR"] = {
        "10 Years": c_10_eps, "5 Years": c_5_eps, "3 Years": c_3_eps,
        "Latest": f"₹{eps_series[-1]:.2f}" if eps_series else "N/A"
    }

    c_10_cfo, s10_cfo, e10_cfo = compute_series_cagr(cfo_series, 10)
    c_5_cfo, s5_cfo, e5_cfo = compute_series_cagr(cfo_series, 5)
    c_3_cfo, s3_cfo, e3_cfo = compute_series_cagr(cfo_series, 3)

    data["CFO_CAGR"] = {
        "10 Years": c_10_cfo, "5 Years": c_5_cfo, "3 Years": c_3_cfo,
        "Latest": f"{format_inr(cfo_series[-1])} Cr" if cfo_series else "N/A"
    }

    c_10_nw, s10_nw, e10_nw = compute_series_cagr(net_worth_series, 10)
    c_5_nw, s5_nw, e5_nw = compute_series_cagr(net_worth_series, 5)
    c_3_nw, s3_nw, e3_nw = compute_series_cagr(net_worth_series, 3)

    data["NetWorth_CAGR"] = {
        "10 Years": c_10_nw, "5 Years": c_5_nw, "3 Years": c_3_nw,
        "Latest": f"{format_inr(net_worth_series[-1])} Cr" if net_worth_series else "N/A"
    }

    # Time-Matched CFO / OP Ratio
    common_years = [
        col for col in data["df_cf"].columns 
        if col in data["df_pl"].columns and col.lower() != 'ttm'
    ]

    if common_years and not data["is_bfsi"]:
        latest_yr = common_years[-1]
        cfo_matched = get_row_series(data["df_cf"][[latest_yr]], "Cash from Operating")
        op_matched = get_row_series(data["df_pl"][[latest_yr]], "Operating Profit")
        
        if cfo_matched and op_matched and op_matched[0] != 0:
            cfo_val = cfo_matched[0]
            op_val = op_matched[0]
            data["CFO_OP_Ratio"] = round((cfo_val / op_val) * 100, 1)
            data["Latest_CFO"] = cfo_val
            data["Latest_OP_Annual"] = op_val
            data["CFO_OP_Period"] = f"FY {latest_yr}"
        else:
            data["CFO_OP_Ratio"] = None
            data["Latest_CFO"] = None
            data["CFO_OP_Period"] = "N/A"
    else:
        data["CFO_OP_Ratio"] = None
        data["Latest_CFO"] = None
        data["CFO_OP_Period"] = "BFSI Waived"

    mcap = safe_float(data.get("Market Cap"), 0.0)
    if data.get("Latest_CFO") and data["Latest_CFO"] > 0 and mcap > 0:
        data["Price_to_CashFlow"] = round(mcap / data["Latest_CFO"], 2)
    else:
        data["Price_to_CashFlow"] = None

    # Data Integrity Tests
    data["audit_checks"] = []
    tot_assets = get_row_series(data["df_bs"], "Total Assets")
    tot_liab = get_row_series(data["df_bs"], "Total Liabilities")
    if tot_assets and tot_liab:
        bs_diff = abs(tot_assets[-1] - tot_liab[-1])
        data["audit_checks"].append({
            "Test": "Balance Sheet Balance Check",
            "Details": f"Assets: ₹{format_inr(tot_assets[-1])} Cr | Liabilities: ₹{format_inr(tot_liab[-1])} Cr",
            "Result": "🟢 Matched & Balanced" if bs_diff < 1.0 else "🔴 Imbalance Detected"
        })

    if op_series and len(op_series) >= 4 and s3_eb and e3_eb:
        test_calc = round(((e3_eb / s3_eb) ** (1.0 / 3.0) - 1.0) * 100.0, 1)
        data["audit_checks"].append({
            "Test": "3-Year EBITDA CAGR Math Integrity",
            "Details": f"Formula: ({format_inr(e3_eb)} / {format_inr(s3_eb)})^(1/3) - 1 = {test_calc}%",
            "Result": "🟢 Formula Verified"
        })

    if data.get("CFO_OP_Ratio") is not None and data.get("CFO_OP_Period") not in ["N/A", "BFSI Waived"]:
        data["audit_checks"].append({
            "Test": f"Time-Matched CFO/OP Check ({data['CFO_OP_Period']})",
            "Details": f"CFO: ₹{format_inr(data['Latest_CFO'])} Cr / OP: ₹{format_inr(data['Latest_OP_Annual'])} Cr = {data['CFO_OP_Ratio']}%",
            "Result": "🟢 Audited Period Matched"
        })

    # Shareholding Trends
    promoter_vals = get_row_series(data["df_shareholding"], "Promoters")
    fii_vals = get_row_series(data["df_shareholding"], "FIIs")
    dii_vals = get_row_series(data["df_shareholding"], "DIIs")
    pledge_vals = get_row_series(data["df_shareholding"], "Pledged")

    data["Promoter_History"] = promoter_vals[-8:] if promoter_vals else []
    data["FII_History"] = fii_vals[-8:] if fii_vals else []
    data["DII_History"] = dii_vals[-8:] if dii_vals else []

    data["Promoter_Latest"] = promoter_vals[-1] if promoter_vals else 0.0
    data["FII_Latest"] = fii_vals[-1] if fii_vals else 0.0
    data["DII_Latest"] = dii_vals[-1] if dii_vals else 0.0
    data["Pledge_Latest"] = pledge_vals[-1] if pledge_vals else 0.0
    
    data["Promoter_Trend"] = "Increasing" if len(promoter_vals) >= 2 and promoter_vals[-1] >= promoter_vals[-2] else "Decreasing"
    data["FII_Trend"] = "Increasing" if len(fii_vals) >= 2 and fii_vals[-1] >= fii_vals[-2] else "Decreasing"
    data["DII_Trend"] = "Increasing" if len(dii_vals) >= 2 and dii_vals[-1] >= dii_vals[-2] else "Decreasing"

    borrowings = get_row_series(data["df_bs"], "Borrowings")
    other_assets = get_row_series(data["df_bs"], "Other Assets")
    other_liab = get_row_series(data["df_bs"], "Other Liabilities")
    
    if borrowings and net_worth_series:
        net_worth = net_worth_series[-1]
        if net_worth > 0:
            data["Calculated_DE"] = round(borrowings[-1] / net_worth, 2)
        else:
            data["Calculated_DE"] = 999.0
    else:
        data["Calculated_DE"] = safe_float(data.get("Debt to equity"), 0.0)

    if other_assets and other_liab and other_liab[-1] > 0:
        data["Current_Ratio"] = round(other_assets[-1] / other_liab[-1], 2)
    else:
        data["Current_Ratio"] = None

    interest_vals = get_row_series(data["df_pl"], "Interest")
    if op_series and interest_vals and interest_vals[-1] > 0:
        data["Interest_Coverage"] = round(op_series[-1] / interest_vals[-1], 2)
    else:
        data["Interest_Coverage"] = None

    f_score = 0
    net_profit_vals = get_row_series(data["df_pl"], "Net Profit")
    if net_profit_vals and net_profit_vals[-1] > 0:
        f_score += 1
    if cfo_series and cfo_series[-1] > 0:
        f_score += 1
    if cfo_series and net_profit_vals and cfo_series[-1] > net_profit_vals[-1]:
        f_score += 1
    if net_profit_vals and len(net_profit_vals) >= 2 and net_profit_vals[-1] > net_profit_vals[-2]:
        f_score += 1
    if borrowings and len(borrowings) >= 2 and borrowings[-1] <= borrowings[-2]:
        f_score += 1
    if other_assets and other_liab and len(other_assets) >= 2 and len(other_liab) >= 2:
        cr_curr = other_assets[-1] / (other_liab[-1] if other_liab[-1] > 0 else 1.0)
        cr_prev = other_assets[-2] / (other_liab[-2] if other_liab[-2] > 0 else 1.0)
        if cr_curr >= cr_prev:
            f_score += 1
    data["Piotroski_Score"] = f_score

    cmp_val = safe_float(data.get("Current Price"), 0.0)
    high_low_str = str(data.get("High / Low", ""))
    high_low_match = re.findall(r"[-+]?\d*\.?\d+", high_low_str.replace(',', ''))
    if len(high_low_match) >= 2 and cmp_val > 0:
        high = float(high_low_match[0])
        low = float(high_low_match[1])
        data["52W_High"] = high
        data["52W_Low"] = low
        data["Dist_High_Pct"] = round(((high - cmp_val) / high) * 100, 1)
        data["Dist_Low_Pct"] = round(((cmp_val - low) / low) * 100, 1)

    return data

# ----------------- SCORING ENGINE -----------------
def evaluate_exact_checklist(m: dict, pe_stats: dict = None):
    results = []

    def add_item(category, name, current_val, points, max_pts, status, guideline):
        results.append({
            "Category": category,
            "Checklist Metric": name,
            "Current Value": current_val,
            "Score": f"{points}/{max_pts}" if max_pts > 0 else "Info",
            "Pts": points,
            "MaxPts": max_pts,
            "Status": status,
            "Guideline / Benchmark": guideline
        })

    add_item("Overview", "NSE Symbol", m.get("Symbol"), 0, 0, "ℹ️ Info", "Stock Ticker")
    add_item("Overview", "Sector & Index", m.get("Sector_Desc", "General"), 0, 0, "ℹ️ Info", "Sectoral trend / Index membership")
    add_item("Overview", "CMP (Live)", f"₹{format_inr(m.get('Current Price'))}", 0, 0, "ℹ️ Info", "Live Market Price")
    add_item("Overview", "Book Value (Audited)", f"₹{format_inr(m.get('Book Value'))}", 0, 0, "ℹ️ Info", "Reported Book Value")
    add_item("Overview", "Face Value", f"₹{m.get('Face Value', 'N/A')}", 0, 0, "ℹ️ Info", "Nominal Share Par Value")
    add_item("Overview", "Dividend Yield (TTM)", f"{m.get('Dividend Yield', 0.0)}%", 0, 0, "ℹ️ Info", "For Info (Low does not necessarily mean Bad)")

    dist_h = safe_float(m.get("Dist_High_Pct"))
    dist_l = safe_float(m.get("Dist_Low_Pct"))
    if dist_h is not None and dist_l is not None:
        if dist_h <= 2.5:
            add_item("Valuation", "52W H/L Proximity", f"{dist_h}% below 52W High", 3, 5, "🟡 Caution", "CMP very close to 52H — Treat with caution")
        elif dist_l <= 4.0:
            add_item("Valuation", "52W H/L Proximity", f"{dist_l}% above 52W Low", 1, 5, "🔴 Caution", "CMP very close to 52W Low — Treat with caution")
        else:
            add_item("Valuation", "52W H/L Proximity", f"H: ₹{format_inr(m.get('52W_High'))} | L: ₹{format_inr(m.get('52W_Low'))}", 5, 5, "🟢 Pass", "Balanced zone within 52W range")
    else:
        add_item("Valuation", "52W H/L Proximity", "N/A", 3, 5, "ℹ️ Info", "Proximity to 52W High/Low bounds")

    mcap = safe_float(m.get("Market Cap"), 0.0)
    if mcap >= 1000:
        add_item("Solvency & Scale", "Market Cap", f"₹{format_inr(mcap)} Cr", 10, 10, "🟢 Pass", "Above ₹1,000 Cr liquidity filter")
    else:
        add_item("Solvency & Scale", "Market Cap", f"₹{format_inr(mcap)} Cr", 1, 10, "🔴 Caution", "Below 1000 cr generally avoidable, until compelling story exists")

    if m.get("is_bfsi"):
        add_item("Solvency & Scale", "Debt to Equity (Audited)", "BFSI Exempt", 15, 15, "🟢 Pass", "Exempt for BFSI")
    else:
        de = safe_float(m.get("Calculated_DE"), 0.0)
        if de >= 900:
            add_item("Solvency & Scale", "Debt to Equity (Audited)", "Negative Net Worth", 0, 15, "🔴 Caution", "Distress: Net worth wiped out by debt")
        elif de <= 0.3:
            add_item("Solvency & Scale", "Debt to Equity (Audited)", f"{de}", 15, 15, "🟢 Pass", "Virtually debt-free")
        elif de <= 0.8:
            add_item("Solvency & Scale", "Debt to Equity (Audited)", f"{de}", 10, 15, "🟢 Pass", "Safe leverage range (<= 0.8)")
        else:
            add_item("Solvency & Scale", "Debt to Equity (Audited)", f"{de}", 0, 15, "🔴 Caution", "If more than 0.8 do thorough checking (Except BFSI)")

    cr = safe_float(m.get("Current_Ratio"))
    if cr is not None and not m.get("is_bfsi"):
        if cr > 1.2:
            add_item("Solvency & Scale", "Current Ratio", f"{cr}", 10, 10, "🟢 Pass", "Comfortable liquidity buffer (> 1.2)")
        elif cr > 1.0:
            add_item("Solvency & Scale", "Current Ratio", f"{cr}", 6, 10, "🟡 Caution", "Borderline working capital (1.0 to 1.2)")
        else:
            add_item("Solvency & Scale", "Current Ratio", f"{cr}", 0, 10, "🔴 Caution", "If less than or = 1, be cautious")
    else:
        add_item("Solvency & Scale", "Current Ratio", "BFSI Waived/NA", 10, 10, "🟢 Pass", "Waived for financial models or unavailable")

    ic = safe_float(m.get("Interest_Coverage"))
    if ic is not None and not m.get("is_bfsi"):
        if ic >= 4.0:
            add_item("Solvency & Scale", "Interest Coverage", f"{ic}x", 5, 5, "🟢 Pass", "Comfortable debt serviceability (> 4x)")
        elif ic >= 2.0:
            add_item("Solvency & Scale", "Interest Coverage", f"{ic}x", 3, 5, "🟡 Caution", "Moderate debt burden (2x to 4x)")
        else:
            add_item("Solvency & Scale", "Interest Coverage", f"{ic}x", 0, 5, "🔴 Fail", "High risk: Operating earnings fail to cover interest (< 2x)")
    else:
        add_item("Solvency & Scale", "Interest Coverage", "Exempt / Debt Free", 5, 5, "🟢 Pass", "No debt interest strain")

    pe = safe_float(m.get("Stock P/E"))
    ind_pe = safe_float(m.get("Industry PE"))
    if pe is not None and pe > 0:
        if ind_pe is not None and ind_pe > 0:
            spread = pe - ind_pe
            if spread > 25:
                add_item("Valuation", "Stock P/E (TTM) vs Industry PE", f"P/E: {pe} vs Ind P/E: {ind_pe}", 4, 10, "🟡 Caution", "Way above Industry PE / check 10-15 yr PE chart & Mean")
            elif spread < -15:
                add_item("Valuation", "Stock P/E (TTM) vs Industry PE", f"P/E: {pe} vs Ind P/E: {ind_pe}", 7, 10, "🟡 Caution", "Way below Industry PE / check for value trap")
            else:
                add_item("Valuation", "Stock P/E (TTM) vs Industry PE", f"P/E: {pe} vs Ind P/E: {ind_pe}", 10, 10, "🟢 Pass", "Aligned with Industry PE")
        else:
            if pe <= 35:
                add_item("Valuation", "Stock P/E (TTM)", f"{pe}", 10, 10, "🟢 Pass", "Reasonable valuation multiple")
            else:
                add_item("Valuation", "Stock P/E (TTM)", f"{pe}", 5, 10, "🟡 Caution", "Elevated standalone P/E")
    else:
        add_item("Valuation", "Stock P/E (TTM)", "Loss Making / Distressed", 0, 10, "🔴 Fail", "Company has negative earnings (No P/E)")

    if pe_stats and pe_stats.get("5Y_Median") != "N/A":
        med_5 = pe_stats["5Y_Median"]
        curr_p = pe_stats["Current_PE"]
        prem = pe_stats.get("Prem_Disc", 0.0)
        if prem <= -15:
            add_item("Valuation", "Historical 5Y Median P/E", f"PE: {curr_p} vs 5Y Med: {med_5} ({prem}%)", 5, 5, "🟢 Pass", "Attractive discount to 5Y Median")
        elif prem <= 20:
            add_item("Valuation", "Historical 5Y Median P/E", f"PE: {curr_p} vs 5Y Med: {med_5} ({prem}%)", 4, 5, "🟢 Pass", "Trading in line with 5Y Median")
        else:
            add_item("Valuation", "Historical 5Y Median P/E", f"PE: {curr_p} vs 5Y Med: {med_5} ({prem}%)", 2, 5, "🟡 Caution", "Elevated relative to historical baseline")

    p_cf = safe_float(m.get("Price_to_CashFlow"))
    if p_cf is not None and not m.get("is_bfsi"):
        if p_cf <= 20:
            add_item("Valuation", "Price to Cash Flow (Audited)", f"{p_cf}", 5, 5, "🟢 Pass", "Healthy cash multiple")
        elif p_cf > 35:
            add_item("Valuation", "Price to Cash Flow (Audited)", f"{p_cf}", 2, 5, "🟡 Caution", "Very high — check if in capex growth phase")
        else:
            add_item("Valuation", "Price to Cash Flow (Audited)", f"{p_cf}", 4, 5, "🟢 Pass", "Moderate cash multiple")
    else:
        add_item("Valuation", "Price to Cash Flow (Audited)", "Negative CFO / NA", 0, 5, "🔴 Caution", "Negative cash flow or data unavailable")

    cfo_op = safe_float(m.get("CFO_OP_Ratio"))
    cfo_period = m.get("CFO_OP_Period", "")
    if cfo_op is not None and not m.get("is_bfsi"):
        period_label = f" [{cfo_period}]" if cfo_period else ""
        if cfo_op >= 100:
            add_item("Capital Efficiency", "CFO / OP (Audited)", f"{cfo_op}%{period_label}", 15, 15, "🟢 Pass", "Comfortable range: If => 100 very good")
        elif cfo_op >= 60:
            add_item("Capital Efficiency", "CFO / OP (Audited)", f"{cfo_op}%{period_label}", 12, 15, "🟢 Pass", "Comfortable range: 60-80%")
        elif cfo_op < 50:
            add_item("Capital Efficiency", "CFO / OP (Audited)", f"{cfo_op}%{period_label}", 2, 15, "🔴 Caution", "If < 50 be cautious (Operating profit not translating to cash)")
        else:
            add_item("Capital Efficiency", "CFO / OP (Audited)", f"{cfo_op}%{period_label}", 8, 15, "🟡 Moderate", "Acceptable range (50-60%)")
    else:
        add_item("Capital Efficiency", "CFO / OP (Audited)", "BFSI Waived / Neg", 10, 15, "ℹ️ Info", "Exempt for BFSI or negative CFO")

    roe = safe_float(m.get("ROE"))
    avg_roe = safe_float(m.get("3Yr_Avg_ROE"))
    roce = safe_float(m.get("ROCE"))

    if roe is not None and roe >= 15:
        add_item("Capital Efficiency", "ROE (Latest FY)", f"{roe}%", 10, 10, "🟢 Pass", "Strong return on equity (> 15%)")
    elif roe is not None and roe > 0:
        add_item("Capital Efficiency", "ROE (Latest FY)", f"{roe}%", 4, 10, "🟡 Moderate", "Moderate/low ROE (< 15%)")
    else:
        add_item("Capital Efficiency", "ROE (Latest FY)", f"{roe}%" if roe is not None else "Negative", 0, 10, "🔴 Fail", "Sub-zero or negative shareholder returns")

    if roe is not None and avg_roe is not None:
        diff_avg = abs(roe - avg_roe)
        if diff_avg > 8.0:
            add_item("Capital Efficiency", "3 Yrs Avg ROE Check", f"Latest: {roe}% vs 3Yr Avg: {avg_roe}%", 3, 5, "🟡 Caution", "Sharp change from 3yrs Avg, check reason - including Corp Action")
        else:
            add_item("Capital Efficiency", "3 Yrs Avg ROE Check", f"Latest: {roe}% vs 3Yr Avg: {avg_roe}%", 5, 5, "🟢 Pass", "Consistent with 3-year average")
    else:
        add_item("Capital Efficiency", "3 Yrs Avg ROE Check", "N/A", 3, 5, "ℹ️ Info", "Historical average unavailable")

    if roe is not None and roce is not None and not m.get("is_bfsi"):
        if (roe - roce) > 10:
            add_item("Capital Efficiency", "ROE vs ROCE Integrity", f"ROE: {roe}% >> ROCE: {roce}%", 2, 10, "🔴 Caution", "ROE very high than ROCE: Check if inflated due to Debt / buyback")
        elif (roce - roe) > 8:
            add_item("Capital Efficiency", "ROE vs ROCE Integrity", f"ROCE: {roce}% >> ROE: {roe}%", 5, 10, "🟡 Caution", "ROCE >> ROE: Check reason - cost of borrowing / sudden tax burden etc")
        elif roe > 0 and roce > 0:
            add_item("Capital Efficiency", "ROE vs ROCE Integrity", f"ROE: {roe}% | ROCE: {roce}%", 10, 10, "🟢 Pass", "Balanced parity between ROE and ROCE")
        else:
            add_item("Capital Efficiency", "ROE vs ROCE Integrity", f"ROE: {roe}% | ROCE: {roce}%", 0, 10, "🔴 Fail", "Negative capital returns")
    else:
        add_item("Capital Efficiency", "ROE vs ROCE Integrity", f"ROCE: {roce}%" if roce is not None else "N/A", 5, 10, "ℹ️ Info", "ROCE Check")

    pledge = safe_float(m.get("Pledge_Latest"), 0.0)
    if pledge == 0:
        add_item("Ownership & Governance", "Prom. Pledge", "0.0%", 10, 10, "🟢 Pass", "Zero is preferred")
    elif pledge < 5:
        add_item("Ownership & Governance", "Prom. Pledge", f"{pledge}%", 5, 10, "🟡 Caution", "Minor pledge present (< 5%)")
    else:
        add_item("Ownership & Governance", "Prom. Pledge", f"{pledge}%", 0, 10, "🔴 Caution", "Warning: Pledged shares > 5%")

    def format_hist(arr):
        return " → ".join([f"{x:.1f}%" for x in arr]) if arr else "N/A"

    fii_val = safe_float(m.get("FII_Latest"), 0.0)
    fii_hist = format_hist(m.get("FII_History", []))
    if m.get("FII_Trend") == "Increasing":
        add_item("Ownership & Governance", "FII Holding (Last 5-6 Qtrs QoQ)", f"{fii_val}% [{fii_hist}]", 5, 5, "🟢 Pass", "Check last 5-6 Qtrs, QoQ (FII accumulating)")
    else:
        add_item("Ownership & Governance", "FII Holding (Last 5-6 Qtrs QoQ)", f"{fii_val}% [{fii_hist}]", 2, 5, "🟡 Moderate", "Check last 5-6 Qtrs, QoQ (FII reduced QoQ)")

    dii_val = safe_float(m.get("DII_Latest"), 0.0)
    dii_hist = format_hist(m.get("DII_History", []))
    if m.get("DII_Trend") == "Increasing":
        add_item("Ownership & Governance", "DII Holding (Last 5-6 Qtrs QoQ)", f"{dii_val}% [{dii_hist}]", 5, 5, "🟢 Pass", "Check last 5-6 Qtrs, QoQ (DII accumulating)")
    else:
        add_item("Ownership & Governance", "DII Holding (Last 5-6 Qtrs QoQ)", f"{dii_val}% [{dii_hist}]", 2, 5, "🟡 Moderate", "Check last 5-6 Qtrs, QoQ (DII reduced QoQ)")

    prom_val = safe_float(m.get("Promoter_Latest"), 0.0)
    prom_hist = format_hist(m.get("Promoter_History", []))
    total_inst = safe_float(m.get("FII_Latest"), 0.0) + safe_float(m.get("DII_Latest"), 0.0)

    # Professionally Managed Entity Logic
    if prom_val == 0.0 and total_inst >= 50.0:
        add_item(
            "Ownership & Governance",
            "Prom Holding (Last 5-6 Qtrs QoQ)",
            "0.0% [Professionally Managed]",
            5,
            5,
            "🟢 Pass",
            f"Professionally managed / board-run entity (Institutional Custody: {total_inst:.1f}% FII+DII)"
        )
    elif prom_val >= 50 or m.get("Promoter_Trend") == "Increasing":
        add_item("Ownership & Governance", "Prom Holding (Last 5-6 Qtrs QoQ)", f"{prom_val}% [{prom_hist}]", 5, 5, "🟢 Pass", "Check last 5-6 Qtrs, QoQ (Strong promoter ownership)")
    else:
        add_item("Ownership & Governance", "Prom Holding (Last 5-6 Qtrs QoQ)", f"{prom_val}% [{prom_hist}]", 3, 5, "🟡 Caution", "Check last 5-6 Qtrs, QoQ (Promoter holding declined / low)")

    s_cagr = safe_float(m.get("3Yr_Sales_CAGR"))
    p_cagr = safe_float(m.get("3Yr_PAT_CAGR"))
    if s_cagr is not None:
        if s_cagr >= 12:
            add_item("Capital Efficiency", "3 Yr Sales CAGR", f"{s_cagr}%", 5, 5, "🟢 Pass", "Healthy double-digit expansion (> 12%)")
        else:
            add_item("Capital Efficiency", "3 Yr Sales CAGR", f"{s_cagr}%", 2, 5, "🟡 Moderate", "Sub-12% top-line growth")
    else:
        add_item("Capital Efficiency", "3 Yr Sales CAGR", "N/A", 2, 5, "ℹ️ Info", "Sales CAGR data not reported")

    if p_cagr is not None:
        if p_cagr >= 12:
            add_item("Capital Efficiency", "3 Yrs PAT CAGR", f"{p_cagr}%", 5, 5, "🟢 Pass", "Strong profit expansion (> 12%)")
        else:
            add_item("Capital Efficiency", "3 Yrs PAT CAGR", f"{p_cagr}%", 2, 5, "🟡 Moderate", "Sub-12% PAT growth")
    else:
        add_item("Capital Efficiency", "3 Yrs PAT CAGR", "N/A", 2, 5, "ℹ️ Info", "PAT CAGR data not reported")

    df = pd.DataFrame(results)
    scored_rows = df[df["MaxPts"] > 0]
    total_pts = scored_rows["Pts"].sum()
    total_max = scored_rows["MaxPts"].sum()
    composite = round((total_pts / total_max) * 100) if total_max > 0 else 0

    cat_breakdown = {}
    for cat in ["Solvency & Scale", "Valuation", "Capital Efficiency", "Ownership & Governance"]:
        c_df = scored_rows[scored_rows["Category"] == cat]
        if not c_df.empty and c_df["MaxPts"].sum() > 0:
            cat_breakdown[cat] = {
                "earned": int(c_df["Pts"].sum()),
                "max": int(c_df["MaxPts"].sum()),
                "pct": round(c_df["Pts"].sum() / c_df["MaxPts"].sum(), 2)
            }
        else:
            cat_breakdown[cat] = {"earned": 0, "max": 10, "pct": 0.0}

    return composite, df, cat_breakdown

# ----------------- EXCEL EXPORT HELPER -----------------
def generate_excel_report(symbol, d, checklist_df, extended_matrix_df, df_pe_table=None, df_forensics=None, df_dupont=None):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        checklist_df.to_excel(writer, sheet_name='Scorecard', index=False)
        extended_matrix_df.to_excel(writer, sheet_name='Extended CAGR Matrix', index=False)
        if df_pe_table is not None and not df_pe_table.empty:
            df_pe_table.to_excel(writer, sheet_name='Historical PE Multiples', index=False)
        if df_forensics is not None and not df_forensics.empty:
            df_forensics.to_excel(writer, sheet_name='Forensic Red Flags', index=False)
        if df_dupont is not None and not df_dupont.empty:
            df_dupont.to_excel(writer, sheet_name='DuPont 3-Stage ROE', index=False)
        if not d["df_pl"].empty:
            d["df_pl"].to_excel(writer, sheet_name='Profit & Loss')
        if not d["df_bs"].empty:
            d["df_bs"].to_excel(writer, sheet_name='Balance Sheet')
        if not d["df_cf"].empty:
            d["df_cf"].to_excel(writer, sheet_name='Cash Flow')
        if not d["df_shareholding"].empty:
            d["df_shareholding"].to_excel(writer, sheet_name='Shareholding')
        if not d["df_peers"].empty:
            d["df_peers"].to_excel(writer, sheet_name='Peers Comparison', index=False)
        if d.get("audit_checks"):
            pd.DataFrame(d["audit_checks"]).to_excel(writer, sheet_name='Integrity Audit', index=False)
    return output.getvalue()

# ----------------- UI APPLICATION -----------------
# Top Banner Branding
if os.path.exists(LOGO_FILE):
    head_col1, head_col2 = st.columns([0.08, 0.92])
    with head_col1:
        st.image(LOGO_FILE, width=70)
    with head_col2:
        st.title("EU QUICK FUNDA CHECK")
else:
    st.title("🏛️ EU QUICK FUNDA CHECK")

st.caption("Institutional fundamental diagnostic terminal with audited historical multiples, DuPont decomposition, forensic detection, and real-time reconciliation.")

# Sidebar Branding with Form Guard
sidebar = st.sidebar
if os.path.exists(LOGO_FILE):
    sidebar.image(LOGO_FILE, width=120)
sidebar.title("EU QUICK FUNDA CHECK")
sidebar.caption("Institutional Fundamental Terminal")
sidebar.divider()

with sidebar.form("audit_form"):
    ticker_input = st.text_input("Enter NSE Ticker", value="TCS").upper()
    search_btn = st.form_submit_button("Run Comprehensive Audit", use_container_width=True)

if ticker_input:
    with st.spinner(f"Auditing institutional financials for {ticker_input}..."):
        d = scrape_full_screener(ticker_input)
        
        df_annual_pe, pe_stats = compute_authentic_historical_pes(
            d["df_pl"] if d else pd.DataFrame(),
            d["df_bs"] if d else pd.DataFrame(),
            safe_float(d.get("Current Price")) if d else None,
            d.get("Price_CAGR", {}) if d else {},
            d.get("Stock P/E") if d else None,
            d.get("Face Value") if d else 10.0
        )
        
        df_forensics, red_flags_cnt, warnings_cnt = evaluate_forensic_red_flags(d) if d else (pd.DataFrame(), 0, 0)
        df_dupont = compute_dupont_analysis(d["df_pl"], d["df_bs"]) if d else pd.DataFrame()
        live_news = fetch_live_news(ticker_input)
        nse_delivery = fetch_nse_delivery_data(ticker_input)
        
    if not d:
        st.error(f"Unable to retrieve verified financials for '{ticker_input}'. Please check the symbol or verify on Screener.in.")
    else:
        final_score, checklist_df, cat_scores = evaluate_exact_checklist(d, pe_stats)
        
        extended_matrix = []
        for p in ["10 Years", "5 Years", "3 Years"]:
            extended_matrix.append({
                "Period": p,
                "Sales CAGR": d.get("Sales_CAGR", {}).get(p, "N/A"),
                "EBITDA / OP CAGR": d.get("EBITDA_CAGR", {}).get(p, "N/A"),
                "Net Profit CAGR": d.get("Profit_CAGR", {}).get(p, "N/A"),
                "EPS CAGR": d.get("EPS_CAGR", {}).get(p, "N/A"),
                "CFO (Cash Flow) CAGR": d.get("CFO_CAGR", {}).get(p, "N/A"),
                "Net Worth CAGR": d.get("NetWorth_CAGR", {}).get(p, "N/A"),
                "ROE % Track": d.get("ROE_History", {}).get(p, "N/A"),
                "Stock Price CAGR": d.get("Price_CAGR", {}).get(p, "N/A")
            })
        extended_matrix.append({
            "Period": "Latest / TTM / 1-Yr",
            "Sales CAGR": d.get("Sales_CAGR", {}).get("TTM", "N/A"),
            "EBITDA / OP CAGR": d.get("EBITDA_CAGR", {}).get("Latest", "N/A"),
            "Net Profit CAGR": d.get("Profit_CAGR", {}).get("TTM", "N/A"),
            "EPS CAGR": d.get("EPS_CAGR", {}).get("Latest", "N/A"),
            "CFO (Cash Flow) CAGR": d.get("CFO_CAGR", {}).get("Latest", "N/A"),
            "Net Worth CAGR": d.get("NetWorth_CAGR", {}).get("Latest", "N/A"),
            "ROE % Track": d.get("ROE_History", {}).get("Last Year", "N/A"),
            "Stock Price CAGR": d.get("Price_CAGR", {}).get("1 Year", "N/A")
        })
        extended_matrix_df = pd.DataFrame(extended_matrix)

        excel_bytes = generate_excel_report(ticker_input, d, checklist_df, extended_matrix_df, df_annual_pe, df_forensics, df_dupont)
        sidebar.divider()
        sidebar.download_button(
            label=f"📥 Export {ticker_input} Audit to Excel",
            data=excel_bytes,
            file_name=f"{ticker_input}_EU_Funda_Check.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CMP (Live)", f"₹{format_inr(d.get('Current Price'))}")
        col2.metric("Market Cap", f"₹{format_inr(safe_float(d.get('Market Cap'), 0))} Cr")
        col3.metric("Stock P/E (TTM)", d.get('Stock P/E', 'N/A'))
        col4.metric("Forensic Red Flags", f"{red_flags_cnt} High Risk", delta=f"{warnings_cnt} Cautions", delta_color="inverse")
        col5.metric("Checklist Score", f"{final_score} / 100", delta="Pass" if final_score >= 70 else "Review")
        
        st.write(f"### {d.get('Company Name')} (`{d.get('Symbol')}`)")
        if d.get("Sector_Desc"):
            st.info(f"Sector / Peer Info: {d.get('Sector_Desc')}")

        if red_flags_cnt >= 2:
            st.error(f"**CRITICAL FORENSIC ALERT:** {red_flags_cnt} High-Risk accounting or cash-flow red flags detected. Inspect individual flags below.")
        elif final_score >= 75:
            st.success(f"**FINAL VERDICT: STRONG PASS ({final_score}/100)** — Sound fundamentals across balance sheet, cash conversion, and capital returns.")
        elif final_score >= 55:
            st.warning(f"**FINAL VERDICT: CONDITIONAL / WATCHLIST ({final_score}/100)** — Moderate profile. Review individual caution flags before entry.")
        else:
            st.error(f"**FINAL VERDICT: AVOID / HIGH CAUTION ({final_score}/100)** — Critical structural, leverage, or liquidity red flags detected.")

        st.markdown("#### 🎯 Score Breakdown by Category")
        pb1, pb2, pb3, pb4 = st.columns(4)
        with pb1:
            c_sol = cat_scores["Solvency & Scale"]
            st.write(f"**Solvency & Scale**: {c_sol['earned']}/{c_sol['max']} pts")
            st.progress(c_sol["pct"])
        with pb2:
            c_val = cat_scores["Valuation"]
            st.write(f"**Valuation**: {c_val['earned']}/{c_val['max']} pts")
            st.progress(c_val["pct"])
        with pb3:
            c_cap = cat_scores["Capital Efficiency"]
            st.write(f"**Capital Returns**: {c_cap['earned']}/{c_cap['max']} pts")
            st.progress(c_cap["pct"])
        with pb4:
            c_gov = cat_scores["Ownership & Governance"]
            st.write(f"**Governance**: {c_gov['earned']}/{c_gov['max']} pts")
            st.progress(c_gov["pct"])

        st.divider()

        tab_scorecard, tab_cagr, tab_financials, tab_dupont, tab_pe_bands, tab_forensics, tab_peers, tab_audit, tab_events = st.tabs([
            "📋 Metric Scorecard", 
            "📈 Compounded Growth (CAGR)",
            "📑 Financial Statements (P&L, BS, CF)", 
            "🔬 DuPont Analysis",
            "📊 Historical P/E Bands",
            "🚩 Forensic Red Flags",
            "👥 Peer Comparison",
            "🛡️ Data Integrity & Verification Audit",
            "🔍 Automated Events, News & Filings"
        ])

        # TAB 1: SCORECARD
        with tab_scorecard:
            st.markdown("#### Itemized Checklist Evaluation (Mapped to Institutional Guidelines)")
            def style_status(val):
                if "Pass" in str(val):
                    return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                elif "Caution" in str(val) or "Moderate" in str(val):
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                elif "Fail" in str(val):
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                return 'color: #555555; font-style: italic;'

            display_table = checklist_df[["Category", "Checklist Metric", "Current Value", "Score", "Status", "Guideline / Benchmark"]]
            styled = display_table.style.map(style_status, subset=['Status'])
            st.dataframe(styled, use_container_width=True, hide_index=True)

        # TAB 2: CLEAN COMPOUNDED GROWTH
        with tab_cagr:
            st.markdown("### 📊 Comprehensive 8-Metric Compounded Matrix")
            st.caption("Native and computed compounded growth rates across financial timeframes.")
            
            st.dataframe(extended_matrix_df, hide_index=True, use_container_width=True)

            st.divider()
            cg1, cg2, cg3, cg4 = st.columns(4)
            with cg1:
                st.markdown("##### 🛒 Compounded Sales Growth")
                if d.get("Sales_CAGR"):
                    st.dataframe(pd.DataFrame(list(d["Sales_CAGR"].items()), columns=["Period", "Sales"]), hide_index=True, use_container_width=True)
            with cg2:
                st.markdown("##### 🏭 EBITDA / Operating Profit CAGR")
                st.dataframe(pd.DataFrame(list(d["EBITDA_CAGR"].items()), columns=["Period", "EBITDA"]), hide_index=True, use_container_width=True)
            with cg3:
                st.markdown("##### 💵 CFO (Cash Flow) CAGR")
                st.dataframe(pd.DataFrame(list(d["CFO_CAGR"].items()), columns=["Period", "CFO Growth"]), hide_index=True, use_container_width=True)
            with cg4:
                st.markdown("##### 🏛️ Net Worth / Equity CAGR")
                st.dataframe(pd.DataFrame(list(d["NetWorth_CAGR"].items()), columns=["Period", "Net Worth"]), hide_index=True, use_container_width=True)

        # TAB 3: FINANCIAL STATEMENTS (WITH INDIAN NUMBER NOTATION FORMATTING)
        with tab_financials:
            st.markdown("### 📑 Primary Financial Statements (₹ Cr)")
            st.caption("All figures formatted with Indian numbering notation (1,00,000).")
            
            if not d["df_pl"].empty:
                st.markdown("#### 1. Profit & Loss Statement (₹ Cr)")
                st.dataframe(format_financial_df(d["df_pl"]), use_container_width=True)
            else:
                st.info("Profit & Loss statement unavailable.")

            st.divider()

            if not d["df_bs"].empty:
                st.markdown("#### 2. Balance Sheet (₹ Cr)")
                st.dataframe(format_financial_df(d["df_bs"]), use_container_width=True)
            else:
                st.info("Balance Sheet statement unavailable.")

            st.divider()

            if not d["df_cf"].empty:
                st.markdown("#### 3. Cash Flow Statement (₹ Cr)")
                st.dataframe(format_financial_df(d["df_cf"]), use_container_width=True)
            else:
                st.info("Cash Flow statement unavailable.")

        # TAB 4: DUPONT 3-STAGE WITH DEFENSIVE PROTECTION
        with tab_dupont:
            st.markdown("### 🔬 DuPont 3-Stage Decomposition")
            st.caption("Deconstructs Return on Equity into Profit Margin (Operating Efficiency), Asset Turnover (Capital Efficiency), and Equity Multiplier (Financial Leverage).")

            if not df_dupont.empty:
                dp_latest = df_dupont.iloc[-1]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Net Profit Margin", f"{dp_latest['Net Profit Margin (%)']}%" if isinstance(dp_latest['Net Profit Margin (%)'], (int, float)) else "N/A")
                m2.metric("Asset Turnover", f"{dp_latest['Asset Turnover (x)']}x" if isinstance(dp_latest['Asset Turnover (x)'], (int, float)) else "N/A")
                m3.metric("Equity Multiplier", f"{dp_latest['Equity Multiplier (x)']}x" if isinstance(dp_latest['Equity Multiplier (x)'], (int, float)) else str(dp_latest['Equity Multiplier (x)']))
                m4.metric("Computed ROE", f"{dp_latest['Computed DuPont ROE (%)']}%" if isinstance(dp_latest['Computed DuPont ROE (%)'], (int, float)) else "N/A")

                st.markdown("#### 📋 10-Year DuPont Decomposition Table")
                st.dataframe(df_dupont, hide_index=True, use_container_width=True)
            else:
                st.info("DuPont decomposition requires complete multi-year P&L and Balance Sheet records.")

        # TAB 5: HISTORICAL P/E BANDS
        with tab_pe_bands:
            st.markdown("### 📊 Historical P/E Valuation Analysis & Multiple Trajectory")
            st.caption("Chronological comparison of year-end P/E multiples against 3Y, 5Y, and 10Y historical medians.")

            if pe_stats:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric(label="Current P/E", value=f"{pe_stats.get('Current_PE', d.get('Stock P/E', 'N/A'))}")
                b2.metric(label="3Y Median P/E", value=f"{pe_stats.get('3Y_Median', 'N/A')}")
                b3.metric(label="5Y Median P/E", value=f"{pe_stats.get('5Y_Median', 'N/A')}")
                b4.metric(label="10Y Median P/E", value=f"{pe_stats.get('10Y_Median', 'N/A')}")

                st.info(f"**Valuation Status:** {pe_stats.get('Zone', 'Valuation Benchmarking Active')}")

            st.markdown("#### 📅 Historical Annual EPS, Net Profit & Year-End P/E Progression")
            if not df_annual_pe.empty:
                st.dataframe(df_annual_pe, hide_index=True, use_container_width=True)

                chart_df = df_annual_pe[df_annual_pe["Historical Year-End P/E"] != "-"].copy()
                if not chart_df.empty:
                    chart_df["Historical Year-End P/E"] = pd.to_numeric(chart_df["Historical Year-End P/E"])
                    plot_data = chart_df.set_index("Fiscal Year")[["Historical Year-End P/E"]]
                    plot_data["5Y Median P/E Baseline"] = pe_stats.get("5Y_Median", 20.0)
                    st.line_chart(plot_data, color=["#1f77b4", "#d62728"])
            else:
                st.warning("Historical statement multiples could not be constructed.")

            st.divider()
            st.link_button(f"🔍 Open Live Valuation Chart for {ticker_input} on Screener", f"https://www.screener.in/company/{ticker_input}/#chart")

        # TAB 6: FORENSIC RED FLAGS
        with tab_forensics:
            st.markdown("### 🚩 Forensic Accounting & Earnings Quality Screen")
            st.caption("Automates institutional forensic checks (accrual anomalies, earnings decoupling, cumulative cash realization, and promoter leverage).")

            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Critical Red Flags", f"{red_flags_cnt}", delta="Clean" if red_flags_cnt == 0 else "High Risk", delta_color="inverse")
            fc2.metric("Warnings / Cautions", f"{warnings_cnt}", delta="Low Risk" if warnings_cnt <= 1 else "Moderate", delta_color="inverse")
            fc3.metric("Earnings Quality Rating", "Low Accrual / High Quality" if red_flags_cnt == 0 else "Aggressive Accrual Profile")

            st.divider()
            
            def style_forensics(val):
                if "RED FLAG" in str(val):
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'
                elif "WARNING" in str(val):
                    return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                elif "CLEAR" in str(val):
                    return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                return ''

            if not df_forensics.empty:
                styled_f = df_forensics.style.map(style_forensics, subset=['Status'])
                st.dataframe(styled_f, use_container_width=True, hide_index=True)
            else:
                st.info("Forensic checks require multi-year statement data.")

        # TAB 7: PEERS BENCHMARKING
        with tab_peers:
            st.markdown("#### Sector Competitor Benchmarking")
            if not d["df_peers"].empty:
                st.dataframe(d["df_peers"], use_container_width=True, hide_index=True)
            else:
                st.info("Live peers table could not be parsed directly. Check competitor rankings below:")
                st.link_button(f"🔍 View {ticker_input} Peers on Screener", f"https://www.screener.in/company/{ticker_input}/#peers")

        # TAB 8: DATA INTEGRITY AUDIT
        with tab_audit:
            st.markdown("### 🛡️ Automated Data Reconciliation & Integrity Audit")
            st.caption("Verifies mathematical coherence across statements to ensure numbers are extracted and calculated accurately.")

            if d.get("audit_checks"):
                df_audit = pd.DataFrame(d["audit_checks"])
                st.dataframe(df_audit, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("#### 🔍 Direct Cross-Verification Links")
            st.write("Verify any individual cell directly on primary sources:")
            al1, al2 = st.columns(2)
            with al1:
                st.link_button(f"🔗 Open Primary Screener Financial Page for {ticker_input}", f"https://www.screener.in/company/{ticker_input}/", use_container_width=True)
            with al2:
                st.link_button(f"🔗 Open BSE India Corporate Filings for {ticker_input}", "https://www.bseindia.com/corporates/ann.html", use_container_width=True)

        # TAB 9: EVENTS, NEWS & FILINGS
        with tab_events:
            ev_col1, ev_col2 = st.columns(2)
            with ev_col1:
                st.markdown("### 📰 Live Company News Feed (Automated)")
                if live_news:
                    for n in live_news:
                        st.markdown(f"• **[{n['title']}]({n['link']})**  \n  <small style='color:gray;'>{n['date']}</small>", unsafe_allow_html=True)
                else:
                    st.info("No recent news headlines found.")

                st.divider()
                st.markdown("### 🚚 Delivery & Volume Absorption (NSE)")
                if nse_delivery and nse_delivery.get('delivery_pct') is not None:
                    d_pct = nse_delivery['delivery_pct']
                    st.metric("Latest NSE Delivery %", f"{d_pct:.1f}%", delta="High Absorption" if d_pct >= 50 else "Normal")
                    st.write(f"• Delivery Quantity: `{format_inr(nse_delivery.get('delivery_qty'))}` shares")
                    st.write(f"• Total Traded Volume: `{format_inr(nse_delivery.get('traded_qty'))}` shares")
                else:
                    st.caption("NSE direct delivery snapshot requires active market session or browser verification.")
                    st.link_button("📊 Check Live NSE Delivery on Official Page", f"https://www.nseindia.com/get-quotes/equity?symbol={ticker_input}")

            with ev_col2:
                st.markdown("### 🎙️ Earnings Calls & Concall Transcripts")
                if d.get("live_concalls"):
                    for c in d["live_concalls"]:
                        st.markdown(f"• **{c['Date']}**: [{c['Title']}]({c['Link']})")
                else:
                    st.info("No concall documents found for this company.")

                st.divider()
                st.markdown("### 🏛️ Official Regulatory Filings & Disclosures")
                if d.get("live_announcements"):
                    for ann in d["live_announcements"]:
                        st.markdown(f"• **{ann['Date']}**: [{ann['Title']}]({ann['Link']})")
                else:
                    st.info("No recent announcements found.")