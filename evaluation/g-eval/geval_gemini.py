"""
GEval for Vietnamese Summarization using Gemini (OpenAI-compatible endpoint)

Output: 1 row per model (input file) with average scores per dimension.

Usage (activate conda env llmrouter first):
    conda activate llmrouter
    python geval_gemini.py --input results/bartpho_cpo_full.csv
    python geval_gemini.py --input results/*.csv --output geval_results.csv
"""

import asyncio
import argparse
import os
import re
import glob
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

N_SAMPLES       = int(os.getenv("GEVAL_N_SAMPLES", "1"))
MAX_CONCURRENCY = int(os.getenv("GEVAL_CONCURRENCY", "50"))
TEMPERATURE     = float(os.getenv("GEVAL_TEMPERATURE", "0.0"))

PROMPT_DIR = Path(__file__).parent / "prompts"

DIMENSIONS = {
    "coherence":   {"file": "coherence.txt",   "max_score": 5},
    "consistency": {"file": "consistency.txt",  "max_score": 5},
    "fluency":     {"file": "fluency.txt",      "max_score": 3},
    "relevance":   {"file": "relevance.txt",    "max_score": 5},
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_prompts() -> dict[str, str]:
    return {
        dim: (PROMPT_DIR / cfg["file"]).read_text(encoding="utf-8")
        for dim, cfg in DIMENSIONS.items()
    }


def parse_score(text: str) -> float | None:
    """Extract score: take text after last colon, grab the last number."""
    t = text.strip()
    if ":" in t:
        t = t.rsplit(":", 1)[-1]
    matches = re.findall(r"\b(\d+(?:\.\d+)?)\b", t)
    return float(matches[-1]) if matches else None

# ── Core async evaluator ─────────────────────────────────────────────────────

async def evaluate_one(
    client: AsyncOpenAI,
    prompt: str,
    dim: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> float | None:
    scores: list[float | None] = []

    for _ in range(N_SAMPLES):
        async with semaphore:
            for attempt in range(retries):
                try:
                    resp = await client.chat.completions.create(
                        model=GEMINI_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=TEMPERATURE,
                        extra_body={
                            'extra_body': {
                                "google": {
                                "thinking_config": {
                                    "thinking_budget": 0,
                                    "include_thoughts": False 
                                }
                                }
                            }
                        }
                    )
                    raw = resp.choices[0].message.content or ""
                    score = parse_score(raw)
                    if score is None:
                        tqdm.write(f"  [warn] unparseable ({dim}): {repr(raw)}")
                    scores.append(score)
                    break
                except Exception as e:
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        scores.append(None)
                        tqdm.write(f"  [error] {dim}: {e}")

    valid = [s for s in scores if s is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


async def evaluate_file(
    df: pd.DataFrame,
    prompts: dict[str, str],
    model_name: str,
) -> dict:
    """Evaluate all rows in df, return 1 dict with mean scores per dimension."""
    client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL)
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # Build flat task list: (row_idx, dim, filled_prompt)
    tasks = []
    for idx, row in df.iterrows():
        doc     = str(row.get("input", ""))
        summary = str(row.get("prediction", ""))
        for dim, template in prompts.items():
            filled = template.replace("{{Document}}", doc).replace("{{Summary}}", summary)
            tasks.append((idx, dim, filled))

    n_calls = len(tasks) * N_SAMPLES
    tqdm.write(f"  {len(df)} rows × {len(DIMENSIONS)} dims × {N_SAMPLES} sample(s) = {n_calls} API calls | concurrency={MAX_CONCURRENCY}")
    t0 = time.time()

    raw_results = await atqdm.gather(
        *[evaluate_one(client, p, dim, semaphore) for _, dim, p in tasks],
        desc=f"  [{model_name}]", unit="call", dynamic_ncols=True,
    )
    tqdm.write(f"  Done in {time.time() - t0:.1f}s")
    await client.close()

    dim_all: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}
    for (_, dim, _), score in zip(tasks, raw_results):
        if score is not None:
            dim_all[dim].append(score)

    result = {"model": model_name}
    for dim, vals in dim_all.items():
        result[f"geval_{dim}"] = round(sum(vals) / len(vals), 4) if vals else None
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GEval summarization evaluation with Gemini")
    p.add_argument("--input", nargs="+", required=True,
                   help="Input CSV file(s). Supports glob, e.g. results/*.csv")
    p.add_argument("--output", default="geval_results.csv",
                   help="Output CSV path (default: geval_results.csv)")
    p.add_argument("--model", default=None,
                   help=f"Gemini model name (default: {GEMINI_MODEL})")
    p.add_argument("--n-samples", type=int, default=None,
                   help=f"Samples per instance per dim (default: {N_SAMPLES})")
    p.add_argument("--concurrency", type=int, default=None,
                   help=f"Max parallel API calls (default: {MAX_CONCURRENCY})")
    return p.parse_args()


async def main():
    args = parse_args()

    global GEMINI_MODEL, N_SAMPLES, MAX_CONCURRENCY
    if args.model:       GEMINI_MODEL    = args.model
    if args.n_samples:   N_SAMPLES       = args.n_samples
    if args.concurrency: MAX_CONCURRENCY = args.concurrency

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set — add it to .env")

    # Resolve input files (support shell glob passed as string)
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        input_files.extend(expanded if expanded else [pattern])
    input_files = sorted(set(input_files))

    if not input_files:
        raise FileNotFoundError(f"No files found: {args.input}")

    prompts = load_prompts()
    output_path = Path(args.output)

    # Load existing output to allow append / skip already-done models
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        done_models = set(existing_df["model"].unique())
        print(f"Existing output: {len(done_models)} model(s) already evaluated")
    else:
        existing_df = pd.DataFrame()
        done_models = set()

    new_rows: list[dict] = []

    for fp in tqdm(input_files, desc="Files", unit="file"):
        model_name = Path(fp).stem
        if model_name in done_models:
            tqdm.write(f"Skip [{model_name}] (already in output)")
            continue

        tqdm.write(f"\n[{model_name}] {fp}")
        df = pd.read_csv(fp)

        missing = {"input", "prediction"} - set(df.columns)
        if missing:
            tqdm.write(f"  [skip] Missing columns: {missing}")
            continue

        row = await evaluate_file(df, prompts, model_name)
        new_rows.append(row)
        tqdm.write("  Result: " + ", ".join(
            f"{k}={v}" for k, v in row.items() if k != "model"
        ))

        # Append new row to existing output and save
        combined = pd.concat(
            [existing_df, pd.DataFrame(new_rows)],
            ignore_index=True,
        ).drop_duplicates(subset=["model"], keep="last")
        combined.to_csv(output_path, index=False)
        tqdm.write(f"  Saved → {output_path}")

    print(f"\nAll done. Results: {output_path}")
    if new_rows:
        print(pd.DataFrame(new_rows).to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
