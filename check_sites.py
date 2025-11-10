import requests
from bs4 import BeautifulSoup
import time
from collections import defaultdict

def query_arquivo_statistics():
    """
    Query Arquivo.pt to find the most archived Portuguese websites between 2010-2020,
    checking coverage per year for each site.
    """
    
    # Well-known Portuguese news and media sites to check
    portuguese_sites = [
        "publico.pt",
        "dn.pt",  # Diário de Notícias
        "jn.pt",  # Jornal de Notícias
        "expresso.pt",  # Expresso
        "cmjornal.pt",  # Correio da Manhã
        "record.pt",  # Record (sports)
        "abola.pt",  # A Bola (sports)
        "rtp.pt",  # RTP (public broadcaster)
        "tvi.pt",  # TVI (TV network)
        "sapo.pt",  # SAPO portal
        "xl.pt",  # XL
        "ionline.pt",  # i online
        "observador.pt",  # Observador
        "tsf.pt",  # TSF Radio
        "sol.sapo.pt",  # SOL
        "visao.sapo.pt",  # Visão magazine
    ]
    
    years = range(2010, 2021)  # 2010-2020
    
    print("="*80)
    print("Checking Arquivo.pt coverage per year for major Portuguese sites (2010-2020)")
    print("="*80)
    
    api_endpoint = "https://arquivo.pt/textsearch"
    
    # Store results per site per year
    site_year_data = defaultdict(dict)
    
    for site in portuguese_sites:
        print(f"\n{'='*80}")
        print(f"Checking {site}")
        print(f"{'='*80}")
        
        for year in years:
            try:
                # Query for this site for specific year
                params = {
                    "q": "",  # Common search term
                    "siteSearch": site,
                    "from": f"{year}0101000000",
                    "to": f"{year}1231235959",
                    "maxItems": 1,  # Just check if results exist
                    "prettyPrint": "false"
                }
                
                print(f"  {year}: ", end='', flush=True)
                response = requests.get(api_endpoint, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    total_items = data.get("estimated_nr_results", 0)
                    
                    site_year_data[site][year] = total_items
                    
                    if total_items > 0:
                        print(f"✓ ~{total_items:,} results")
                    else:
                        print(f"✗ No results")
                else:
                    print(f"✗ Error {response.status_code}")
                    site_year_data[site][year] = 0
                
                # Respect rate limits
                time.sleep(0.4)
                
            except Exception as e:
                print(f"✗ Error: {str(e)[:50]}")
                site_year_data[site][year] = 0
                continue
    
    # Calculate total per site and sort
    site_totals = []
    for site, year_data in site_year_data.items():
        total = sum(year_data.values())
        years_with_data = sum(1 for count in year_data.values() if count > 0)
        site_totals.append({
            'site': site,
            'total': total,
            'years_with_data': years_with_data,
            'year_data': year_data
        })
    
    site_totals.sort(key=lambda x: x['total'], reverse=True)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: TOP PORTUGUESE SITES IN ARQUIVO.PT (2010-2020)")
    print("="*80)
    print(f"{'Site':<20} {'Total Results':>15} {'Years w/ Data':>15} {'Avg/Year':>15}")
    print("-"*80)
    for r in site_totals:
        avg = r['total'] / 11 if r['total'] > 0 else 0
        print(f"{r['site']:<20} {r['total']:>15,} {r['years_with_data']:>15} {avg:>15,.0f}")
    
    # Print detailed year-by-year for top 5 sites
    print("\n" + "="*80)
    print("YEAR-BY-YEAR BREAKDOWN (Top 5 Sites)")
    print("="*80)
    
    for site_data in site_totals[:5]:
        site = site_data['site']
        year_data = site_data['year_data']
        
        print(f"\n{site}:")
        print(f"  {'Year':<8} {'Results':>12}")
        print(f"  {'-'*20}")
        for year in years:
            count = year_data.get(year, 0)
            if count > 0:
                print(f"  {year:<8} {count:>12,}")
            else:
                print(f"  {year:<8} {'(none)':>12}")
    
    print("\n" + "="*80)
    
    return site_year_data, site_totals

# Run the query
if __name__ == "__main__":
    year_data, totals = query_arquivo_statistics()