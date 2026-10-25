"""Yield-leg scan: Pendle's active PT markets joined to the perp that would hedge them.

For every active Pendle market (all chains, public API) whose underlying has a perp on Binance,
Bybit or Hyperliquid (including Hyperliquid's HIP-3 `xyz` dex for stocks and gold): PT implied APY,
pool liquidity, and for each venue the trailing 30-day funding of its best-paying candidate perp
(annualised; a short receives it) and that perp's current open interest. Combined carry =
PT implied APY + funding on the venue with the highest funding; costs and the margin reserve are
not deducted. Capacity = min(2% of PT pool liquidity, 1% of that venue's perp OI), the carry note's
participation rates. USD stablecoin and credit PTs need no hedge: they are fixed income, and their
liquidity-weighted median implied APY is the benchmark a hedged trade has to beat.

`--lst` scans liquid-staking and restaking tokens instead (DefiLlama yields API): each token's
issuer pool (apyBase = staking yield, tvlUsd), joined to the same 30-day funding and current perp OI
for its underlying. Combined carry = staking APY + funding on the venue paying the short most,
before costs. With a 25% idle margin reserve, 75% of capital holds the token and is hedged with an
equal perp short, so carry on capital = 0.75 x combined.

A snapshot, dated in the output, not a backtest.

    python scripts/yield_leg_scan.py          # writes reports/yield_leg/scan.csv
    python scripts/yield_leg_scan.py --lst    # writes reports/yield_leg/lst_scan.csv
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path("reports/yield_leg"); OUT.mkdir(parents=True, exist_ok=True)
PENDLE = "https://api-v2.pendle.finance/core"
# Pendle market name -> hedge underlying. Liquid staking / restaking / wrapped tokens hedge with the
# underlying's perp, tokenised stocks with the stock perp, gold tokens with any gold perp.
UNDERLYING = {
    "wstETH": "ETH", "pufETH": "ETH", "superWETH": "ETH", "mixWETH": "ETH",
    "slisBNB": "BNB", "kHYPE": "HYPE", "stHYPE": "HYPE", "vkHYPE": "HYPE", "haHYPE": "HYPE",
    "hBTC": "BTC", "sKAITO": "KAITO", "shMON": "MON", "sENA": "ENA",
    "NVDAc": "NVDA", "superNVDA": "NVDA", "GOOGLc": "GOOGL", "superTSLA": "TSLA", "STRCx": "STRC", "PFE": "PFE",
    "PAXGy": "GOLD", "XGLD": "GOLD",
}
STOCKS = {"NVDA", "GOOGL", "TSLA", "STRC", "PFE"}
GOLD = {"BN": ["XAUUSDT", "PAXGUSDT", "XAUTUSDT"], "BY": ["XAUUSDT", "PAXGUSDT", "XAUTUSDT"], "HL": ["xyz:GOLD", "PAXG"]}


def candidates(u: str) -> dict[str, list[str]]:
    """Perp symbols per venue that could hedge underlying u (a symbol a venue does not list returns nothing)."""
    if u == "GOLD":
        return GOLD
    hl = f"xyz:{u}" if u in STOCKS else u
    return {"BN": [f"{u}USDT"], "BY": [f"{u}USDT"], "HL": [hl]}
PART_POOL, PART_OI = 0.02, 0.01
DAYS = 30
_S = requests.Session()


def _get(url, **params):
    for k in range(4):
        r = _S.get(url, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** k)
    r.raise_for_status()


def _post(url, body):
    r = _S.post(url, json=body, timeout=30); r.raise_for_status(); return r.json()


def pendle_markets() -> pd.DataFrame:
    """GET /v2/markets/all?isActive=true (the documented cross-chain listing, 100 per page)."""
    rows, skip = [], 0
    while True:
        page = _get(f"{PENDLE}/v2/markets/all", isActive="true", skip=skip, limit=100)
        for m in page["results"]:
            d = m["details"]
            rows.append({"chain": m["chainId"], "market": m["name"], "expiry": m["expiry"][:10],
                         "pt_implied_apy": d["impliedApy"], "underlying_apy": d.get("underlyingApy"),
                         "pool_liquidity_usd": d["liquidity"], "volume_24h_usd": d.get("tradingVolume"),
                         "categories": ",".join(m.get("categoryIds", []))})
        skip += len(page["results"])
        if not page["results"] or skip >= page["total"]:
            return pd.DataFrame(rows)


def funding_30d(venue: str, sym: str, start_ms: int) -> float | None:
    """Sum of settlements over the window, annualised (/ days * 365)."""
    try:
        if venue == "BN":
            x = _get("https://fapi.binance.com/fapi/v1/fundingRate", symbol=sym, startTime=start_ms, limit=1000)
            r = [float(e["fundingRate"]) for e in x]
        elif venue == "BY":
            x = _get("https://api.bybit.com/v5/market/funding/history", category="linear", symbol=sym,
                     startTime=start_ms, endTime=int(time.time() * 1000), limit=200)["result"]["list"]
            r = [float(e["fundingRate"]) for e in x]
        else:
            r, t = [], start_ms                                       # HL returns at most 500 rows per call
            while True:
                x = _post("https://api.hyperliquid.xyz/info", {"type": "fundingHistory", "coin": sym, "startTime": t})
                r += [float(e["fundingRate"]) for e in x]
                if len(x) < 500:
                    break
                t = x[-1]["time"] + 1
    except Exception:
        return None
    return sum(r) / DAYS * 365 if r else None


def open_interest(bn_syms: set[str]) -> dict[str, dict[str, float]]:
    """Current OI in USD by venue and symbol."""
    oi = {"BN": {}, "BY": {}, "HL": {}}
    for sym in sorted(bn_syms):
        try:
            q = _get("https://fapi.binance.com/fapi/v1/openInterest", symbol=sym)
            px = float(_get("https://fapi.binance.com/fapi/v1/premiumIndex", symbol=sym)["markPrice"])
            oi["BN"][sym] = float(q["openInterest"]) * px
        except Exception:
            pass
    for t in _get("https://api.bybit.com/v5/market/tickers", category="linear")["result"]["list"]:
        oi["BY"][t["symbol"]] = float(t["openInterestValue"])
    for dex in (None, "xyz"):                                       # main dex and the HIP-3 stock/commodity dex
        body = {"type": "metaAndAssetCtxs"} | ({"dex": dex} if dex else {})
        meta, ctx = _post("https://api.hyperliquid.xyz/info", body)
        for a, c in zip(meta["universe"], ctx):
            oi["HL"][a["name"]] = float(c["openInterest"]) * float(c["markPx"])
    return oi


def kind(row) -> str:
    if isinstance(row.underlying, str):
        return "stock" if row.underlying in STOCKS else "gold" if row.underlying == "GOLD" else "crypto"
    cats = row.categories.split(",")
    return "fixed income (USD)" if ("stables" in cats or "rwa" in cats) and "stocks" not in cats else "unmapped"


def main():
    now = datetime.now(timezone.utc)
    start_ms = int((now.timestamp() - DAYS * 86400) * 1000)
    mk = pendle_markets()
    mk["underlying"] = mk.market.map(UNDERLYING)
    mk["kind"] = mk.apply(kind, axis=1)
    us = mk.underlying.dropna().unique()
    oi = open_interest({s_ for u in us for s_ in candidates(u)["BN"]})
    best = {}                                                       # (venue, underlying) -> (symbol, funding)
    for u in us:
        for v, syms in candidates(u).items():
            got = [(s_, f) for s_ in syms if (f := funding_30d(v, s_, start_ms)) is not None]
            if got:
                best[(v, u)] = max(got, key=lambda x: x[1])
    for v in ("BN", "BY", "HL"):
        b = [best.get((v, u)) if isinstance(u, str) else None for u in mk.underlying]
        mk[f"perp_{v}"] = [x[0] if x else None for x in b]
        mk[f"funding_30d_{v}"] = [x[1] if x else None for x in b]
        mk[f"perp_oi_usd_{v}"] = [oi[v].get(x[0]) if x else None for x in b]
    fcols = [f"funding_30d_{v}" for v in ("BN", "BY", "HL")]
    f = mk[fcols].astype(float)
    hv = pd.Series([r.idxmax()[-2:] if r.notna().any() else None for _, r in f.iterrows()], index=mk.index)
    mk["hedge_venue"] = hv
    mk["hedge_perp"] = [mk.at[i, f"perp_{v}"] if isinstance(v, str) else None for i, v in hv.items()]
    mk["hedge_funding"] = [mk.at[i, f"funding_30d_{v}"] if isinstance(v, str) else None for i, v in hv.items()]
    mk["hedge_oi_usd"] = [mk.at[i, f"perp_oi_usd_{v}"] if isinstance(v, str) else None for i, v in hv.items()]
    mk["combined_carry"] = mk.pt_implied_apy + mk.hedge_funding.astype(float)
    fi = mk[mk.kind == "fixed income (USD)"]
    w = fi.sort_values("pt_implied_apy"); c = w.pool_liquidity_usd.cumsum()
    bench = float(w.pt_implied_apy[c >= c.iloc[-1] / 2].iloc[0])     # liquidity-weighted median
    mk["fixed_income_benchmark"] = bench
    mk["carry_over_benchmark"] = mk.combined_carry - bench
    cap_pool = PART_POOL * mk.pool_liquidity_usd
    cap_oi = PART_OI * mk.hedge_oi_usd.astype(float)
    mk["capacity_usd"] = cap_pool.where(cap_oi.isna() | (cap_pool <= cap_oi), cap_oi)
    mk["binding"] = (cap_pool <= cap_oi).map({True: "PT pool", False: "perp OI"}).where(cap_oi.notna())
    mk["snapshot_utc"] = now.strftime("%Y-%m-%d %H:%M")
    mk = mk.sort_values(["kind", "underlying", "combined_carry"], ascending=[True, True, False], na_position="last")
    mk.to_csv(OUT / "scan.csv", index=False)
    pd.set_option("display.width", 300)
    h = mk[mk.underlying.notna()]
    print(h[["chain", "market", "expiry", "kind", "pt_implied_apy", "pool_liquidity_usd", *fcols, "hedge_perp",
             "combined_carry", "carry_over_benchmark", "hedge_oi_usd", "capacity_usd", "binding"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    print(f"\nfixed-income benchmark (liquidity-weighted median implied APY of {len(fi)} USD PTs): {bench:.2%}")
    print(mk.groupby("kind").agg(markets=("market", "size"), pool_liquidity_m=("pool_liquidity_usd", lambda x: x.sum() / 1e6)).round(1).to_string())


LLAMA = "https://yields.llama.fi/pools"
# token -> (DefiLlama issuer project, the pool's symbol there, perp underlying). The issuer's own pool
# carries the staking yield; lending pools with the same symbol report their lending rate instead.
# wstETH is wrapped stETH: same yield, same Lido pool. stHYPE has no issuer pool on DefiLlama.
LST = {
    "stETH/wstETH": ("lido", "STETH", "ETH"), "rETH": ("rocket-pool", "RETH", "ETH"),
    "cbETH": ("coinbase-wrapped-staked-eth", "CBETH", "ETH"), "weETH": ("ether.fi-stake", "WEETH", "ETH"),
    "ezETH": ("renzo", "EZETH", "ETH"), "jitoSOL": ("jito-liquid-staking", "JITOSOL", "SOL"),
    "mSOL": ("marinade-liquid-staking", "MSOL", "SOL"), "bnSOL": ("binance-staked-sol", "BNSOL", "SOL"),
    "slisBNB": ("lista-liquid-staking", "SLISBNB", "BNB"), "stHYPE": (None, "STHYPE", "HYPE"),
    "kHYPE": ("kinetiq-khype", "KHYPE", "HYPE"),
}
RESERVE = 0.25


def hl_quarterly(coins: list[str]) -> pd.DataFrame:
    """Hyperliquid funding by calendar quarter, annualised, from the stored hourly history
    (python -m qr.data.hl_history funding). Partial quarters are marked by their hour count."""
    import glob
    out = {}
    for c in coins:
        fs = sorted(glob.glob(f"data/raw/hyperliquid/funding/{c}-PERP/**/*.parquet", recursive=True))
        if not fs:
            continue
        r = pd.concat(pd.read_parquet(f) for f in fs)["rate"].astype(float)
        r = r[~r.index.duplicated()].sort_index()
        q = r.resample("QE").agg(["sum", "count"])
        out[f"{c}_apr"] = q["sum"] / (q["count"] / 24) * 365
        out[f"{c}_hours"] = q["count"]
    t = pd.DataFrame(out); t.index = t.index.tz_localize(None).to_period("Q").astype(str)
    return t


def lst_main():
    now = datetime.now(timezone.utc)
    start_ms = int((now.timestamp() - DAYS * 86400) * 1000)
    pools = pd.DataFrame(_get(LLAMA)["data"])
    pools["sym"] = pools.symbol.str.upper()
    rows = []
    for tok, (proj, sym, u) in LST.items():
        m = pools[(pools.project == proj) & (pools.sym == sym)] if proj else pools.iloc[:0]
        p = m.sort_values("tvlUsd", ascending=False).iloc[0] if len(m) else None
        rows.append({"token": tok, "underlying": u, "project": proj, "chain": p.chain if p is not None else None,
                     "staking_apy": p.apyBase / 100 if p is not None and pd.notna(p.apyBase) else None,
                     "tvl_usd": p.tvlUsd if p is not None else None})
    df = pd.DataFrame(rows)
    us = df.underlying.unique()
    oi = open_interest({s_ for u in us for s_ in candidates(u)["BN"]})
    fund = {(v, u): funding_30d(v, sym_, start_ms) for u in us for v, syms in candidates(u).items() for sym_ in syms[:1]}
    for v in ("BN", "BY", "HL"):
        df[f"funding_30d_{v}"] = [fund.get((v, u)) for u in df.underlying]
        df[f"perp_oi_usd_{v}"] = [oi[v].get(candidates(u)[v][0]) for u in df.underlying]
    f = df[[f"funding_30d_{v}" for v in ("BN", "BY", "HL")]].astype(float)
    hv = [r.idxmax()[-2:] if r.notna().any() else None for _, r in f.iterrows()]
    df["short_venue"] = hv
    df["short_funding_apr"] = [df.at[i, f"funding_30d_{v}"] if v else None for i, v in zip(df.index, hv)]
    df["short_perp_oi_usd"] = [df.at[i, f"perp_oi_usd_{v}"] if v else None for i, v in zip(df.index, hv)]
    df["combined_carry"] = df.staking_apy + df.short_funding_apr.astype(float)
    df["carry_with_25pct_reserve"] = (1 - RESERVE) * df.combined_carry
    df["capacity_1pct_oi_usd"] = PART_OI * df.short_perp_oi_usd.astype(float)      # per underlying, shared by its tokens
    df["snapshot_utc"] = now.strftime("%Y-%m-%d %H:%M")
    df = df.sort_values("combined_carry", ascending=False, na_position="last")
    df.to_csv(OUT / "lst_scan.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[["token", "project", "staking_apy", "tvl_usd", "funding_30d_BN", "funding_30d_BY", "funding_30d_HL",
              "short_venue", "short_funding_apr", "combined_carry", "carry_with_25pct_reserve", "short_perp_oi_usd", "capacity_1pct_oi_usd"]]
          .to_string(index=False, float_format=lambda v: f"{v:,.4g}"))
    q = hl_quarterly(list(df.underlying.unique()))
    q.to_csv(OUT / "lst_hl_funding_quarterly.csv")
    print("\nHyperliquid funding by quarter (APR; hours in quarter):")
    print(q.to_string(float_format=lambda v: f"{v:,.3g}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lst", action="store_true", help="liquid-staking tokens from DefiLlama instead of Pendle PTs")
    lst_main() if ap.parse_args().lst else main()
