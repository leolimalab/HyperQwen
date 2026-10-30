"""Single-stream benchmark for mac/serve.sh (llama-server) and mac/serve_mlx.sh
(mlx_vlm.server), stdlib only. Both servers return llama.cpp-style `timings`.

Two workloads, the two that decide the single-user setup letter in the main README:

  chat    bench/prompts_real.jsonl (8 mixed en/da/code prompts), the B row
  quote   reproduce a 2k-token document verbatim, the C row (lookup drafting)

Speeds come from llama-server's own per-request `timings` (decode = predicted
tokens / predicted ms, prefill = prompt tokens / prompt ms), so client overhead
and the HTTP round trip are not in them. Draft acceptance is `draft_n_accepted /
draft_n` summed over the requests. The first request after a start is a warmup
and is thrown away, as in bench/run_benchmarks.sh.

  python3 mac/bench.py mylabel                       # chat + quote, model-default sampling
  python3 mac/bench.py mylabel --sampling default,greedy --max-tokens 512

Results land in mac/results/<label>.json; the ROW lines are the report.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "bench"))
from verbatim import coverage  # noqa: E402


def api_key():
    for var in ("OPENAI_API_KEY", "VLLM_API_KEY"):
        if os.environ.get(var):
            return os.environ[var]
    path = os.path.join(REPO, "api_key.txt")
    return open(path).read().strip() if os.path.exists(path) else "EMPTY"


def get(url, path):
    req = urllib.request.Request(url + path, headers={"Authorization": "Bearer " + api_key()})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


MODEL = None   # the served model id: mlx_vlm loads (or downloads) whatever a request names


STREAM = False  # oMLX reports its timings only on the last chunk of a stream


def chat(url, messages, max_tokens, sampling, think=True):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": STREAM,
            "chat_template_kwargs": {"enable_thinking": think},   # llama.cpp, oMLX
            "enable_thinking": think}                             # mlx_vlm
    if STREAM:
        body["stream_options"] = {"include_usage": True}
    if sampling == "greedy":
        body.update(temperature=0.0, top_k=1)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json",
                                  "Authorization": "Bearer " + api_key()})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not STREAM:
            out = json.load(r)
            out["_wall"] = time.time() - t0
            return out
        text, reasoning, usage, t_first, t_last = [], [], {}, None, None
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                thought = d.get("reasoning_content") or d.get("reasoning") or ""
                piece = (d.get("content") or "") + thought
                if piece:
                    t_last = time.time()
                    t_first = t_first or t_last
                    text.append(d.get("content") or "")
                    reasoning.append(thought)
    # The server's own rates when it sends them; otherwise the client's clock,
    # first streamed token to last (speculative chunks carry several tokens, so
    # the count comes from usage, not from the chunks).
    n = usage.get("completion_tokens", 0)
    p = usage.get("prompt_tokens", 0) - ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    gen_s = usage.get("generation_duration") or ((t_last - t_first) if t_first and t_last else 0)
    pre_s = usage.get("prompt_eval_duration") or ((t_first - t0) if t_first else 0)
    return {"choices": [{"message": {"content": "".join(text), "reasoning_content": "".join(reasoning)}}],
            "usage": usage, "_wall": time.time() - t0,
            "timings": {"predicted_n": n, "predicted_ms": gen_s * 1000, "prompt_n": p, "prompt_ms": pre_s * 1000,
                        "predicted_per_second": usage.get("generation_tokens_per_second") or (n / gen_s if gen_s else 0)}}


def timings(r):
    """llama.cpp and mlx_vlm return `timings`; oMLX puts durations in `usage`."""
    if r.get("timings"):
        return r["timings"]
    u = r.get("usage") or {}
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    n, gen_s = u.get("completion_tokens", 0), u.get("generation_duration") or 0
    p, pre_s = u.get("prompt_tokens", 0) - cached, u.get("prompt_eval_duration") or 0
    return {"predicted_n": n, "predicted_ms": gen_s * 1000, "prompt_n": p, "prompt_ms": pre_s * 1000,
            "predicted_per_second": u.get("generation_tokens_per_second") or (n / gen_s if gen_s else 0)}


def swap_mb():
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.split()
    return float(s[s.index("used") + 2].rstrip("M")) if "used" in s else 0.0


def summarize(rows):
    t = lambda k: sum(timings(r).get(k, 0) or 0 for r in rows)
    pred_n, pred_ms, prompt_n, prompt_ms = t("predicted_n"), t("predicted_ms"), t("prompt_n"), t("prompt_ms")
    draft_n, draft_acc = t("draft_n"), t("draft_n_accepted")
    return {
        "decode_tok_s": round(pred_n / pred_ms * 1000, 2) if pred_ms else None,
        "prefill_tok_s": round(prompt_n / prompt_ms * 1000, 1) if prompt_ms else None,
        "acceptance": round(draft_acc / draft_n, 3) if draft_n else None,
        "predicted_tokens": pred_n,
        "requests": len(rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--url", default=os.environ.get("URL", "http://127.0.0.1:18020"))
    ap.add_argument("--workloads", default="chat,quote")
    ap.add_argument("--sampling", default="default", help="default, greedy, or both comma-separated")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--quote-max-tokens", type=int, default=2400,
                    help="room for the copy; a model that always thinks first (Flint) needs ~3x the document")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--no-think", action="store_true", help="chat prompts without the thinking block")
    ap.add_argument("--model", help="served model id (default: the first one /v1/models lists)")
    ap.add_argument("--stream", action="store_true", help="stream and time each answer (needed for oMLX)")
    a = ap.parse_args()

    global MODEL, STREAM
    STREAM = a.stream
    MODEL = a.model or get(a.url, "/v1/models")["data"][0]["id"]
    try:
        props = get(a.url, "/props")          # llama.cpp only
    except Exception:
        props = {"model_path": MODEL, "build_info": "mlx_vlm"}
    prompts = [json.loads(l)["prompt"] for l in open(os.path.join(REPO, "bench", "prompts_real.jsonl"))][:a.prompts]
    doc = open(os.path.join(REPO, "docs", "optimizations.md")).read()[:8000]
    quote_msg = [{"role": "user", "content": "Reproduce the document below exactly, character for character, "
                                             "with no preamble and no commentary.\n\n" + doc}]

    chat(a.url, [{"role": "user", "content": "Say hello."}], 32, "default")   # warmup, discarded
    result = {"label": a.label, "when": time.strftime("%Y-%m-%dT%H:%M"), "args": vars(a),
              "server": {k: props.get(k) for k in ("model_path", "total_slots", "build_info")},
              "swap_mb_before": swap_mb(), "rows": {}}

    for sampling in a.sampling.split(","):
        if "chat" in a.workloads:
            rows = []
            for i, p in enumerate(prompts):
                r = chat(a.url, [{"role": "user", "content": p}], a.max_tokens, sampling, think=not a.no_think)
                rows.append(r)
                tm = timings(r)
                print(f"  chat/{sampling} #{i}: {tm.get('predicted_per_second', 0):6.2f} tok/s decode, "
                      f"{tm.get('draft_n_accepted', 0)}/{tm.get('draft_n', 0)} drafts accepted", flush=True)
            s = summarize(rows)
            result["rows"][f"chat/{sampling}"] = s
            print(f"ROW chat/{sampling} | decode={s['decode_tok_s']} tok/s | prefill={s['prefill_tok_s']} tok/s "
                  f"| accept={s['acceptance']} | {s['predicted_tokens']} tokens", flush=True)
        if "quote" in a.workloads:
            r = chat(a.url, quote_msg, a.quote_max_tokens, sampling, think=False)
            msg = r["choices"][0]["message"]
            ans = msg.get("content") or ""
            s = summarize([r])
            s["coverage"] = round(coverage(ans, doc), 3)
            # A copy that never reaches the answer reads as coverage 0; these say why.
            s["answer_chars"], s["reasoning_chars"] = len(ans), len(msg.get("reasoning_content") or "")
            s["finish_reason"] = r["choices"][0].get("finish_reason")
            result["rows"][f"quote/{sampling}"] = s
            print(f"ROW quote/{sampling} | decode={s['decode_tok_s']} tok/s | prefill={s['prefill_tok_s']} tok/s "
                  f"| accept={s['acceptance']} | coverage={s['coverage']} | {s['predicted_tokens']} tokens "
                  f"| answer {s['answer_chars']} chars, reasoning {s['reasoning_chars']} chars, {s['finish_reason']}", flush=True)

    result["swap_mb_after"] = swap_mb()
    if result["swap_mb_after"] - result["swap_mb_before"] > 256:
        print(f"!! swap grew {result['swap_mb_after'] - result['swap_mb_before']:.0f} MiB during the run: "
              "the numbers above are memory-bound, not representative", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{a.label}.json")
    json.dump(result, open(path, "w"), indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
