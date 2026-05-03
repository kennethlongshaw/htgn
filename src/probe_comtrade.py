import requests

url = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
HEADERS = {"User-Agent": "Mozilla/5.0 (probe script)"}

PROBES = [
    {"reporterCode": 842, "partnerCode": 156, "period": 2023, "cmdCode": "TOTAL", "flowCode": "X"},  # US exports to China
    {"reporterCode": 842, "partnerCode": 156, "period": 2023, "cmdCode": "TOTAL", "flowCode": "M"},  # US imports from China
    {"reporterCode": 156, "partnerCode": 842, "period": 2023, "cmdCode": "TOTAL", "flowCode": "X"},  # China exports to US
]

for params in PROBES:
    resp = requests.get(url, params=params, headers=HEADERS)
    if not resp.ok:
        print(f"HTTP {resp.status_code} for {params}: {resp.text[:200]}")
        continue
    data = resp.json()

    rows = data.get("data", [])
    if not rows:
        print(f"No data for params: {params}")
        continue

    for row in rows[:3]:
        reporter = row.get("reporterDesc", "?")
        partner = row.get("partnerDesc", "?")
        flow = row.get("flowDesc", "?")
        period = row.get("period", "?")
        value = row.get("primaryValue")
        value_str = f"{value:,.0f}" if value is not None else "N/A"
        print(f"{reporter} -> {partner}  [{flow}]  {period}  USD {value_str}")
    print()
