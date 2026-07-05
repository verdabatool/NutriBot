"""
NutriChat evaluation harness.

Measures, per objective:
  1. Grounding accuracy  — every (ID: N) in the answer is a real recipe whose
     name matches the DB (uses ungrounded_ids). Catches hallucination.
  2. Task success (TSR)  — the answer has the right shape for the task.
  3. Tool-usage          — the agent called an expected tool for the task.
  4. LLM-as-judge        — a model grades quality on a rubric (0-10 + verdict).

Run:  python -m src.eval.evaluate            (qwen3-32b, one case per objective)
      python -m src.eval.evaluate --model llama-3.1-8b-instant
"""
from __future__ import annotations

import argparse
import json
import re
import time
from os import getenv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Set

from dotenv import load_dotenv
load_dotenv()

from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from src.agent.react_agent import build_agent, ungrounded_ids
from src.db.recipes import get_recipes_by_ids


# --------------------------------------------------
# Tool-call recorder
# --------------------------------------------------
class ToolRecorder(BaseCallbackHandler):
    def __init__(self):
        self.tools: List[str] = []

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized or {}).get("name")
        if name:
            self.tools.append(name)


# --------------------------------------------------
# Task-shape checkers (Task Success Rate)
# --------------------------------------------------
def _ids(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\(ID:\s*(\d+)\)", text)]


def chk_qa(t: str) -> bool:
    # Pass if EITHER:
    # (a) a direct grounded answer — a time/temp value + a real recipe id, OR
    # (b) a clarification — lists >=2 real recipe options and asks which one
    #     (correct behavior when the question applies to many recipes).
    ids = _ids(t)
    has_val = bool(re.search(r"\b\d+\s*(?:-\s*\d+\s*)?(?:min|minute|hour|degree|°|f\b)", t, re.I))
    direct = has_val and bool(ids)
    clarify = len(ids) >= 2 and bool(re.search(r"which|specify|clarif|\?", t, re.I))
    return direct or clarify


def chk_suggestions(t: str) -> bool:
    return len(_ids(t)) >= 3


def chk_nutrition(t: str) -> bool:
    low = t.lower()
    has_cal = "cal" in low
    macros = sum(m in low for m in ("protein", "carb", "fat"))
    return has_cal and macros >= 2


def chk_plan(t: str) -> bool:
    days = len(re.findall(r"(?i)\bday\s*\d", t))
    return days >= 2 and bool(_ids(t))


def chk_modify(t: str) -> bool:
    low = t.lower()
    scaled = any(w in low for w in ("scale", "multiply", "0.33", "1/3", "serving"))
    return scaled and bool(re.search(r"\bingredient", low))


def chk_shopping(t: str) -> bool:
    items = re.findall(r"(?m)^\s*[-*•]\s+\S+", t)
    return len(items) >= 3 or "shopping list" in t.lower()


def chk_allergy(t: str) -> bool:
    # acknowledges the allergy and gives at least one grounded recipe
    return "peanut" in t.lower() and bool(_ids(t))


# --------------------------------------------------
# Extra correctness: allergen leakage check
# --------------------------------------------------
def has_allergen(text: str, allergen: str) -> bool:
    """True if any recipe id in the answer actually contains the allergen."""
    ids = _ids(text)
    if not ids:
        return False
    try:
        df = get_recipes_by_ids(ids)
    except Exception:
        return False
    stem = allergen.rstrip("s")
    for j in df.get("ingredients_json", []):
        if j and stem in str(j).lower():
            return True
    return False


# --------------------------------------------------
# Test cases: one per objective
# --------------------------------------------------
@dataclass
class Case:
    objective: str
    prompt: str
    checker: Callable[[str], bool]
    expected_tools: Set[str]
    preference: str = "No preference"
    extra: Optional[Callable[[str], bool]] = None  # extra correctness gate


CASES: List[Case] = [
    Case("1. Recipe Q&A", "How long do I bake lasagna?", chk_qa,
         {"recipe_lookup", "recipe_instructions"}),
    Case("2. Ingredient suggestions", "I have chicken, rice, and bell peppers. What can I make?",
         chk_suggestions, {"ingredient_suggester"}),
    Case("3. Nutrition analysis", "What are the calories and macros for recipe 15253?",
         chk_nutrition, {"nutrition_analyzer"}),
    Case("4. Meal planning", "Make me a 3-day vegetarian meal plan under 1500 calories per day.",
         chk_plan, {"meal_planner"}),
    Case("5. Modify & scale", "Scale recipe 15253 from 6 servings to 2.",
         chk_modify, {"modify_recipe"}),
    Case("6. Shopping list", "Give me a consolidated shopping list for recipes 15253 and 52.",
         chk_shopping, {"shopping_list"}),
    Case("7. Allergy honoring", "I'm allergic to peanuts. Suggest a snack recipe with its ID.",
         chk_allergy, {"recipe_lookup", "ingredient_suggester"},
         preference="ALLERGIES (never suggest these): peanuts",
         extra=lambda t: not has_allergen(t, "peanut")),
]


# --------------------------------------------------
# LLM-as-judge
# --------------------------------------------------
JUDGE_RUBRIC = (
    "Grade the assistant answer for a recipe chatbot on: grounded (uses real "
    "recipes, no invented facts), on-task, appropriately complete, and concise. "
    "IMPORTANT: when a question could apply to MANY different recipes (e.g. 'how "
    "long do I bake lasagna?'), asking the user to clarify which specific recipe "
    "they mean — while listing real options with IDs — is the CORRECT, complete "
    "response; treat it as ACCEPT, not incomplete. "
    "Reply ONLY with compact JSON: "
    '{"score": <0-10 int>, "verdict": "ACCEPT"|"REJECT", "reason": "<short>"}'
)


def judge(judge_llm, prompt: str, answer: str) -> dict:
    msg = f"{JUDGE_RUBRIC}\n\nUSER ASKED:\n{prompt}\n\nASSISTANT ANSWER:\n{answer}\n\nJSON:"
    try:
        resp = judge_llm.invoke(msg)
        content = getattr(resp, "content", "") or ""
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        return {"score": int(data.get("score", 0)), "verdict": data.get("verdict", "REJECT"),
                "reason": data.get("reason", "")}
    except Exception as e:
        return {"score": 0, "verdict": "REJECT", "reason": f"judge error: {e}"}


# --------------------------------------------------
# Runner
# --------------------------------------------------
def _mark(b) -> str:
    return "✅" if b else "❌"


def _run_once(agent, judge_llm) -> List[dict]:
    """Run all cases once; return a row per case."""
    rows = []
    for i, c in enumerate(CASES, 1):
        rec = ToolRecorder()
        t0 = time.time()
        try:
            res = agent.invoke(
                {"input": c.prompt, "user_preference": c.preference},
                config={"configurable": {"thread_id": f"eval-{i}-{int(t0*1000)}"}, "callbacks": [rec]},
            )
            answer = res.get("output") if isinstance(res, dict) else str(res)
            err = None
        except Exception as e:
            answer, err = "", str(e)[:120]
        dt = time.time() - t0

        grounded = (not ungrounded_ids(answer)) if answer else False
        tsr = (c.checker(answer) if answer else False) and (c.extra(answer) if (c.extra and answer) else True)
        tool_ok = bool(c.expected_tools & set(rec.tools))
        j = judge(judge_llm, c.prompt, answer) if answer else {"score": 0, "verdict": "REJECT", "reason": err or "no answer"}

        rows.append({
            "objective": c.objective, "prompt": c.prompt,
            "grounded": grounded, "tsr": tsr, "tool_ok": tool_ok,
            "tools": rec.tools, "judge": j["score"], "verdict": j["verdict"],
            "judge_reason": j.get("reason", ""), "seconds": round(dt, 1), "error": err,
            "answer": answer or "",
        })
    return rows


def run(model: Optional[str], judge_model: str, runs: int = 1) -> None:
    runs = max(1, int(runs))
    print(f"\nBuilding agent (model={model or 'default from .env'})…  runs={runs}")
    agent = build_agent(model=model)
    judge_llm = ChatGroq(model=judge_model, temperature=0, max_tokens=200)

    all_rows: List[dict] = []
    for run_idx in range(1, runs + 1):
        if runs > 1:
            print(f"\n--- Run {run_idx}/{runs} ---")
        rows = _run_once(agent, judge_llm)
        for r in rows:
            r["run"] = run_idx
            flag = _mark(r["grounded"] and r["tsr"] and r["tool_ok"])
            print(f"{flag} {r['objective']:26} grounded={r['grounded']!s:5} tsr={r['tsr']!s:5} "
                  f"tool={r['tool_ok']!s:5} judge={r['judge']}/10  ({r['seconds']}s)")
            if r["error"]:
                print(f"     error: {r['error']}")
        all_rows.extend(rows)

    # ---- Overall aggregate across every (run x case) result ----
    n = len(all_rows)
    def pct(key): return round(100 * sum(bool(r[key]) for r in all_rows) / n, 1)
    agg = {
        "runs": runs, "cases": len(CASES), "results": n,
        "grounding_pct": pct("grounded"), "tsr_pct": pct("tsr"), "tool_pct": pct("tool_ok"),
        "avg_judge": round(sum(r["judge"] for r in all_rows) / n, 1),
        "accept_pct": round(100 * sum(r["verdict"] == "ACCEPT" for r in all_rows) / n, 1),
    }

    print("\n" + "=" * 60)
    print(f"SCORECARD  ({runs} run(s) x {len(CASES)} cases = {n} results)")
    print("=" * 60)
    print(f"Grounding accuracy : {agg['grounding_pct']}%")
    print(f"Task success (TSR) : {agg['tsr_pct']}%")
    print(f"Tool-usage correct : {agg['tool_pct']}%")
    print(f"Avg judge score    : {agg['avg_judge']}/10")
    print(f"Judge ACCEPT rate  : {agg['accept_pct']}%")

    # ---- Per-objective aggregate (X out of runs) ----
    per_obj = {}
    for c in CASES:
        rs = [r for r in all_rows if r["objective"] == c.objective]
        k = len(rs) or 1
        per_obj[c.objective] = {
            "runs": k,
            "grounded": sum(bool(r["grounded"]) for r in rs),
            "tsr": sum(bool(r["tsr"]) for r in rs),
            "tool": sum(bool(r["tool_ok"]) for r in rs),
            "accept": sum(r["verdict"] == "ACCEPT" for r in rs),
            "avg_judge": round(sum(r["judge"] for r in rs) / k, 1),
        }

    print(f"\nPer-objective (passes out of {runs}):")
    print(f"{'Objective':26} {'Grnd':>6} {'TSR':>6} {'Tool':>6} {'Judge':>6} {'ACCEPT':>7}")
    print("-" * 62)
    for obj, a in per_obj.items():
        rn = a["runs"]
        g, t, to, ac = f"{a['grounded']}/{rn}", f"{a['tsr']}/{rn}", f"{a['tool']}/{rn}", f"{a['accept']}/{rn}"
        print(f"{obj:26} {g:>6} {t:>6} {to:>6} {str(a['avg_judge']):>6} {ac:>7}")

    # ---- Save ----
    out_dir = Path(__file__).resolve().parents[2] / "eval_runs"
    out_dir.mkdir(exist_ok=True)
    from src.agent.react_agent import DEFAULT_MODEL
    actual_model = model or DEFAULT_MODEL
    model_slug = re.sub(r"[^a-z0-9.]+", "-", actual_model.split("/")[-1].lower()).strip("-")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runtag = f"_x{runs}" if runs > 1 else ""
    stem = out_dir / f"eval_{ts}_{model_slug}{runtag}"

    # 1) JSON (full detail: every run's rows + aggregates)
    stem.with_suffix(".json").write_text(json.dumps(
        {"model": model, "judge_model": judge_model, "runs": runs,
         "aggregate": agg, "per_objective": per_obj, "rows": all_rows}, indent=2))

    # 2) CSV (one row per run x objective)
    import csv
    with open(stem.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "objective", "grounded", "tsr", "tool_ok", "judge", "verdict",
                    "seconds", "tools", "error", "judge_reason", "answer"])
        for r in all_rows:
            w.writerow([r.get("run", 1), r["objective"], r["grounded"], r["tsr"], r["tool_ok"],
                        r["judge"], r["verdict"], r["seconds"], "|".join(r["tools"]),
                        r["error"] or "", r.get("judge_reason", ""), r.get("answer", "")])

    # 3) Markdown report
    md = [
        f"# NutriChat evaluation — {ts}", "",
        f"- Model: `{model or 'default (.env)'}` | Judge: `{judge_model}` | Runs: {runs}", "",
        "## Scorecard (averaged over all runs)", "",
        "| Metric | Score |", "|---|---|",
        f"| Grounding accuracy | {agg['grounding_pct']}% |",
        f"| Task success (TSR) | {agg['tsr_pct']}% |",
        f"| Tool-usage correct | {agg['tool_pct']}% |",
        f"| Avg judge score | {agg['avg_judge']}/10 |",
        f"| Judge ACCEPT rate | {agg['accept_pct']}% |",
        "", f"## Per-objective (passes out of {runs})", "",
        "| Objective | Grounded | TSR | Tool | Avg judge | ACCEPT |",
        "|---|:--:|:--:|:--:|:--:|:--:|",
    ]
    for obj, a in per_obj.items():
        rn = a["runs"]
        md.append(f"| {obj} | {a['grounded']}/{rn} | {a['tsr']}/{rn} | {a['tool']}/{rn} | "
                  f"{a['avg_judge']}/10 | {a['accept']}/{rn} |")

    # Full responses + judge reasoning (grouped by objective, each run)
    md += ["", "## Full responses & judge reasoning", ""]
    for c in CASES:
        md += [f"### {c.objective}", f"**Prompt:** {c.prompt}", ""]
        for r in [x for x in all_rows if x["objective"] == c.objective]:
            head = f"**Run {r['run']}** — " if runs > 1 else ""
            md += [
                f"{head}Tools: {', '.join(r['tools']) or '—'} | "
                f"Grounded {_mark(r['grounded'])} TSR {_mark(r['tsr'])} Tool {_mark(r['tool_ok'])} | "
                f"Judge {r['judge']}/10 ({r['verdict']}) — {r.get('judge_reason','')}",
                "", "```", (r.get("answer") or "(no answer)"), "```", "",
            ]
    stem.with_suffix(".md").write_text("\n".join(md))

    print(f"\nSaved: {stem}.json  •  {stem}.csv  •  {stem}.md")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="agent model (default: .env GROQ_MODEL)")
    p.add_argument("--judge-model",
                   default=getenv("GROQ_JUDGE_MODEL", "llama-3.1-8b-instant"),
                   help="LLM-as-judge model (default: .env GROQ_JUDGE_MODEL, else "
                        "llama-3.1-8b-instant). For a trustworthy grade use a model "
                        "at least as strong as the one under test, e.g. "
                        "llama-3.3-70b-versatile. CLI flag overrides .env.")
    p.add_argument("--runs", type=int, default=1,
                   help="Run the whole suite N times and average, to smooth out "
                        "tool-calling variance (default 1).")
    args = p.parse_args()
    run(args.model, args.judge_model, runs=args.runs)


if __name__ == "__main__":
    main()