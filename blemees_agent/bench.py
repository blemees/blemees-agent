"""Latency benchmark for blemeesd (blemees/3, #26 — spec §10–§12).

Run with::

    python -m blemees_agent.bench [--socket PATH] [--profile NAME] [--agent NAME]
                                  [--model MODEL] [--iters 3]

Measures, per the agent the daemon is configured with (or ``--profile``):
    * cold_first_chunk   — fresh open + first agent_message_chunk
    * warm_first_chunk   — second turn on the same session → first chunk
    * resume_first_chunk — close + re-open with resume:true → first chunk

The headline number is **warm prompt → first chunk**: the typing-latency a
user feels once the agent process is warm. Cold includes process spawn +
ACP ``initialize``; resume depends on the agent's ``session/load`` support
(#23). Numbers are reported, not gated — they vary by agent and account.

The daemon must already be running at ``--socket`` and its agent(s) must be
authenticated, otherwise the runs surface auth errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid

from .client import BlemeesClient, default_socket_path


def _options(model: str | None) -> dict[str, object]:
    return {"model": model} if model else {}


def _is_message_chunk(evt: dict) -> bool:
    """True for the first sign of model *text* output — an ACP
    ``agent_message_chunk`` carried on a session.update (the criterion's
    'first chunk'). Bookkeeping updates (tool_call, plan) don't count."""
    if evt.get("type") != "session.update":
        return False
    return evt.get("update", {}).get("sessionUpdate") == "agent_message_chunk"


async def _first_chunk_latency(sess, prompt: str) -> float:
    t0 = time.monotonic()
    await sess.send_user(prompt)
    async for evt in sess.events():
        t = evt.get("type")
        if t in ("error", "session.error"):
            raise RuntimeError(evt)
        if _is_message_chunk(evt):
            return time.monotonic() - t0
        # A turn that ends without any text chunk still stops the clock.
        if t == "session.result":
            return time.monotonic() - t0
    raise RuntimeError("stream ended without any event")


async def _drain_to_result(sess) -> int:
    """Drain events until a turn-end ``session.result`` arrives. Returns
    the highest seq seen so the caller can resume cleanly."""
    last_seq = 0
    async for evt in sess.events():
        seq = evt.get("seq")
        if isinstance(seq, int) and seq > last_seq:
            last_seq = seq
        if evt.get("type") == "session.result":
            return last_seq
    return last_seq


async def run_one(args: argparse.Namespace) -> dict[str, float]:
    results: dict[str, float] = {}
    session_id = str(uuid.uuid4())
    options = _options(args.model)
    prompt = args.prompt
    open_kw = {"options": options, "profile": args.profile, "agent": args.agent}

    async with await BlemeesClient.connect(args.socket) as c:
        # Cold open
        t_open = time.monotonic()
        async with c.open_session(session_id=session_id, **open_kw) as sess:
            results["cold_first_chunk"] = await _first_chunk_latency(sess, prompt)
            results["cold_open_plus_first"] = time.monotonic() - t_open
            last_seq = await _drain_to_result(sess)

            # Warm — the headline number.
            results["warm_first_chunk"] = await _first_chunk_latency(sess, prompt)
            last_seq = await _drain_to_result(sess)

    # Resume: reconnect and re-open with resume:true (depends on the agent's
    # session/load support, #23).
    async with await BlemeesClient.connect(args.socket) as c:
        t_resume = time.monotonic()
        async with c.open_session(
            session_id=session_id, resume=True, last_seen_seq=last_seq, **open_kw
        ) as sess:
            results["resume_first_chunk"] = await _first_chunk_latency(sess, prompt)
            results["resume_open_plus_first"] = time.monotonic() - t_resume
            await _drain_to_result(sess)

    return results


async def main_async(args: argparse.Namespace) -> int:
    rows: list[dict[str, float]] = []
    for i in range(args.iters):
        row = await run_one(args)
        rows.append(row)
        print(f"iter {i + 1}: {json.dumps(row, indent=2)}")
    if len(rows) > 1:
        keys = rows[0].keys()
        avg = {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
        print("average:", json.dumps(avg, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m blemees_agent.bench")
    ap.add_argument("--socket", default=default_socket_path())
    ap.add_argument("--profile", default=None, help="Profile to open under (blemees/3, #17)")
    ap.add_argument("--agent", default=None, help="Agent within the profile")
    ap.add_argument("--model", default=None, help="Model name to request (agent-specific)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--prompt", default="Reply with just the word OK.")
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
