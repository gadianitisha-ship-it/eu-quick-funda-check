import requests
import json
from datetime import datetime
import time

# Flagship tickers used to pull the macro Sector P/E from NSE
SECTOR_MAP = {
    "IT": "TCS",
    "PHARMA": "SUNPHARMA",
    "BFSI": "HDFCBANK",
    "FMCG": "ITC",
    "AUTO": "MARUTI",
    "GENERAL": "RELIANCE"
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.nseindia.com/'
}

def get_sector_pes():
    session = requests.Session()
    session.headers.update(HEADERS)
    
    # Hit main page to generate session cookies
    try:
        session.get("https://www.nseindia.com", timeout=5)
    except:
        pass

    results = {
        "last_updated": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data": {}
    }

    for sector, symbol in SECTOR_MAP.items():
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
        try:
            res = session.get(url, timeout=5)
            if res.status_code == 200:
                meta = res.json().get('metadata', {})
                sector_pe = meta.get('pdSectorPe') or meta.get('sectorPe')
                if sector_pe:
                    results["data"][sector] = float(sector_pe)
        except Exception as e:
            print(f"Failed to fetch {sector}: {e}")
        
        # Polite delay to prevent NSE blocking the GitHub runner
        time.sleep(2)

    # Write the results to a JSON file
    with open("sector_pe.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully updated sector_pe.json: {results['data']}")

if __name__ == "__main__":
    get_sector_pes()
