"""Live trade + order-book collector for Binance USDT-M perps and Hyperliquid perps.

Run:  python -m qr.data.collector --symbols BTC ETH SOL --exchanges binance hyperliquid
Each exchange runs in its own task, reconnects on drop, and flushes buffered rows to the
store every FLUSH_SECONDS. Book snapshots are stored flat: ts, bid_px_0..N, bid_sz_0..N,
ask_px_0..N, ask_sz_0..N (N = DEPTH). Trades: ts, px, qty, side (+1 buy / -1 sell), id.

This is deliberately simple. The point is to have three weeks of your own book data by
day 20. Leave it running under nohup / tmux / a systemd unit.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import time
from collections import defaultdict

import pandas as pd

from qr.data.store import append_rows

log = logging.getLogger("collector")
DEPTH = 10
FLUSH_SECONDS = 30
BINANCE_WS = "wss://fstream.binance.com"
HL_WS = "wss://api.hyperliquid.xyz/ws"


class Buffer:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)  # (exchange, dataset, symbol)

    def add(self, exchange: str, dataset: str, symbol: str, row: dict) -> None:
        self.rows[(exchange, dataset, symbol)].append(row)

    def flush(self) -> int:
        n = 0
        for (ex, ds, sym), rows in list(self.rows.items()):
            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts").sort_index()
            append_rows(df, "raw", ex, ds, sym)
            n += len(rows)
            self.rows[(ex, ds, sym)] = []
        return n


def flatten_book(ts_ms: int, bids: list, asks: list) -> dict:
    row = {"ts": ts_ms}
    for i in range(DEPTH):
        bp, bs = (bids[i] if i < len(bids) else (None, None))
        ap, as_ = (asks[i] if i < len(asks) else (None, None))
        row[f"bid_px_{i}"] = float(bp) if bp is not None else None
        row[f"bid_sz_{i}"] = float(bs) if bs is not None else None
        row[f"ask_px_{i}"] = float(ap) if ap is not None else None
        row[f"ask_sz_{i}"] = float(as_) if as_ is not None else None
    return row


async def _ws_loop(name: str, url: str, subscribe: list[dict] | None, handler, buf: Buffer) -> None:
    import websockets

    backoff = 1
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, max_size=2**23) as ws:
                log.info("%s connected", name)
                backoff = 1
                for msg in subscribe or []:
                    await ws.send(json.dumps(msg))
                async for raw in ws:
                    try:
                        handler(json.loads(raw), buf)
                    except Exception as e:  # noqa: BLE001
                        log.warning("%s handler error: %s", name, e)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("%s dropped (%s); reconnecting in %ss", name, e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


# ---------------- Binance USDT-M perps ----------------

def binance_handler(msg: dict, buf: Buffer) -> None:
    stream = msg.get("stream", "")
    d = msg.get("data", {})
    sym = d.get("s", "").replace("USDT", "") + "-PERP"
    if stream.endswith("@aggTrade"):
        buf.add("binance", "trades", sym, {
            "ts": d["T"], "px": float(d["p"]), "qty": float(d["q"]),
            "side": -1 if d["m"] else 1, "id": d["a"],
        })
    elif "@depth" in stream:
        buf.add("binance", "book", sym, flatten_book(d["T"], d["b"], d["a"]))


async def binance_task(symbols: list[str], buf: Buffer):
    # Binance serves trades on /market and depth on /public; the bare /stream URL drops aggTrade.
    syms = [s.lower() + "usdt" for s in symbols]
    trades = "/".join(f"{s}@aggTrade" for s in syms)
    depth = "/".join(f"{s}@depth{20}@100ms" for s in syms)
    await asyncio.gather(
        _ws_loop("binance-trades", f"{BINANCE_WS}/market/stream?streams={trades}", None, binance_handler, buf),
        _ws_loop("binance-book", f"{BINANCE_WS}/public/stream?streams={depth}", None, binance_handler, buf),
    )


# ---------------- Hyperliquid perps ----------------

def hl_handler(msg: dict, buf: Buffer) -> None:
    ch = msg.get("channel")
    d = msg.get("data")
    if ch == "trades" and isinstance(d, list):
        for t in d:
            buf.add("hyperliquid", "trades", t["coin"] + "-PERP", {
                "ts": t["time"], "px": float(t["px"]), "qty": float(t["sz"]),
                "side": 1 if t["side"] == "B" else -1, "id": t.get("tid"),
            })
    elif ch == "l2Book" and isinstance(d, dict):
        bids = [(l["px"], l["sz"]) for l in d["levels"][0]]
        asks = [(l["px"], l["sz"]) for l in d["levels"][1]]
        buf.add("hyperliquid", "book", d["coin"] + "-PERP", flatten_book(d["time"], bids, asks))


def hl_task(symbols: list[str], buf: Buffer):
    subs = []
    for s in symbols:
        subs.append({"method": "subscribe", "subscription": {"type": "trades", "coin": s}})
        subs.append({"method": "subscribe", "subscription": {"type": "l2Book", "coin": s}})
    return _ws_loop("hyperliquid", HL_WS, subs, hl_handler, buf)


# ---------------- main ----------------

async def _flusher(buf: Buffer, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=FLUSH_SECONDS)
        except asyncio.TimeoutError:
            pass
        n = buf.flush()
        log.info("flushed %d rows", n)


async def main(symbols: list[str], exchanges: list[str]) -> None:
    buf = Buffer()
    stop = asyncio.Event()
    tasks = [asyncio.create_task(_flusher(buf, stop))]
    if "binance" in exchanges:
        tasks.append(asyncio.create_task(binance_task(symbols, buf)))
    if "hyperliquid" in exchanges:
        tasks.append(asyncio.create_task(hl_task(symbols, buf)))

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass
    await stop.wait()
    for t in tasks[1:]:
        t.cancel()
    buf.flush()
    log.info("stopped at %s", time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"])
    ap.add_argument("--exchanges", nargs="+", default=["binance", "hyperliquid"])
    ap.add_argument("--log", default="INFO")
    a = ap.parse_args()
    logging.basicConfig(level=a.log, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    asyncio.run(main(a.symbols, a.exchanges))
