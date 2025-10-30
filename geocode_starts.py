import sys, time, json, pandas as pd, aiohttp, asyncio

async def geocode_one(session, q, city, ua):
    txt = q if q else ""
    query = f"{txt}, {city}" if city and city not in txt else txt
    params = {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 0}
    async with session.get("https://nominatim.openstreetmap.org/search",
                           params=params, headers={"User-Agent": ua}, timeout=aiohttp.ClientTimeout(total=15)) as r:
        if r.status != 200: return None
        data = await r.json()
        if not data: return None
        return float(data[0]["lat"]), float(data[0]["lon"])

async def main(inp, outp, city, ua):
    df = pd.read_csv(inp)
    lat = []; lon = []
    async with aiohttp.ClientSession() as s:
        for _, row in df.iterrows():
            if pd.notna(row.get("start_lat")) and pd.notna(row.get("start_lon")):
                lat.append(row["start_lat"]); lon.append(row["start_lon"]); continue
            res = await geocode_one(s, row.get("start_raw",""), city, ua)
            if res is None: lat.append(None); lon.append(None)
            else: lat.append(res[0]); lon.append(res[1])
            await asyncio.sleep(1.0)  # rate limit
    df["start_lat"] = lat; df["start_lon"] = lon
    df.to_csv(outp, index=False)

if __name__ == "__main__":
    # usage: python geocode_starts.py queries_parsed.csv queries_geo.csv "Нижний Новгород, Россия" "you@example.com"
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
