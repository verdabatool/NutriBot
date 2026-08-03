# 🥗 NutriChat — Grounded Recipe & Nutrition Assistant

NutriChat is an **agentic, retrieval-grounded chatbot** that helps users find recipes, answer cooking questions, analyze nutrition, plan meals, and build shopping lists — all backed by a real database of **231,637 recipes** [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv). Every recipe, ID, ingredient, step, and nutrition value it states comes from the data, not the model's imagination.

It's built as a **tool-calling LangChain agent** running on **Groq** LLMs, with a **FAISS + SQLite** RAG layer and a **Streamlit** interface (register/login, per-user profiles, dark/light mode).

---

## ✨ What it does (objectives covered)

1. **Recipe Q&A:** Answers specific questions ("How long do I bake lasagna?"). If a question applies to many recipes, it asks you to clarify; if a detail is missing from the data, it says so and reasons from general knowledge (clearly labeled).
2. **Ingredient-based suggestions:** The "what's in my fridge" problem: give it what you have and it ranks recipes by how many of your ingredients they use.
3. **Nutritional analysis:** Calories + **macros in grams** (protein/carbs/fat…) per serving, and a meal total, converted from the dataset's %DV.
4. **Weekly meal planning:** A full multi-day plan with **breakfast/lunch/dinner** each matched to the right meal type, honoring **diet** (vegetarian/vegan/…) and a **per-day calorie target**.
5. **Recipe modification & scaling:** "Make it gluten-free" or "scale from 6 servings to 2" (returns a scale factor + grounded, clearly-labeled substitutions).
6. **Shopping list:** A consolidated, de-duplicated list from a meal plan or a set of recipes.
7. **Persistent profiles:** Remembers **allergies, diet, and health goals** (stored in the DB, bcrypt-hashed passwords), excludes allergens from results, and can save new preferences stated in chat.

---

## 🧠 How it works (architecture)

```
 ┌─────────────────────────────────────────────────────┐
 │            Streamlit UI  (app.py)                   │
 │      auth · chat · profile · dark/light             │
 └──────────────────────┬──────────────────────────────┘
                        │ user message
                        ▼
 ┌─────────────────────────────────────────────────────┐
 │     Tool-calling Agent  (LangChain + Groq)          │
 │  system prompt · summary-buffer memory              │
 │  primary qwen3-32b ──(429/413)──► llama-3.1-8b      │
 └──────────────────────┬──────────────────────────────┘
                        │ picks from 8 grounded tools
      ┌─────────────────┼─────────────────────┐
      ▼                 ▼                     ▼
 recipe_lookup   nutrition · meal_planner   save_user_profile
 (semantic RAG)  modify · shopping_list     (logged-in users)
      │          ingredients · resolve             │
      ▼                 ▼                          ▼
 ┌──────────────┐  ┌─────────────────────────────────┐
 │ FAISS index  │  │ SQLite — recipes · ingredients  │
 │ (MiniLM      │  │ tags · users (bcrypt profiles)  │
 │  embeddings) │  └─────────────────────────────────┘
 └──────────────┘
                        │ draft answer
                        ▼
 ┌─────────────────────────────────────────────────────┐
 │  Grounding guard — every (ID: N) verified against   │
 │  the DB (exists + name matches); hallucinated IDs   │
 │  ──► forget turn & regenerate                       │
 └──────────────────────┬──────────────────────────────┘
                        ▼
                  grounded reply
```

Engineering highlights:
- **Native tool-calling agent** (not brittle text-ReAct) — reliable tool selection across 8 grounded tools.
- **Grounding guard** — every recipe ID in a reply is verified against the DB (real *and* name-matching); hallucinated IDs are caught and the answer is regenerated.
- **Self-sufficient tools** — e.g. `meal_planner` fetches its own recipes, so a full plan is a single tool call.
- **Summary-buffer memory** — keeps recent turns verbatim and summarizes older ones, so long chats don't blow up token usage.
- **Automatic model fallback** — falls back to a lighter model on rate-limit (429) or oversized-request (413) errors.
- **Security** — bcrypt password hashing, profiles in the database, secrets via `.env`.

**Models** (all configurable in `.env`): primary `qwen/qwen3-32b`, fallback + summary `llama-3.1-8b-instant`, eval judge `llama-3.3-70b-versatile`.

---

## 📦 Dataset

Recipes come from **`RAW_recipes.csv`** — the [Food Recommend dataset on Kaggle](https://www.kaggle.com/datasets/nguynphananhc/food-recommend?select=RAW_recipes.csv), derived from Food.com. It contains **~231,637 recipes**, each with: `name`, `ingredients`, `steps`, `tags`, prep `minutes`, and a `nutrition` list (calories + six % Daily Value figures: total fat, sugar, sodium, protein, saturated fat, carbs).

The build pipeline (`src/ingestion/`) preprocesses it into a normalized SQLite schema — expanding nutrition into columns, splitting out `recipe_ingredients` / `recipe_tags`, and building a combined text field that's embedded into the FAISS index for semantic search. Nutrition grams shown in the app are estimated by converting those %DV values using standard FDA daily values.

---

## 📊 Evaluation

A self-contained harness at `src/eval/evaluate.py` scores the agent on **10 cases** — the **7 core objectives plus 3 adversarial/safety probes** where the correct answer is to *refuse*: a nonexistent recipe ID (must not fabricate nutrition), an out-of-scope question (must decline, invent no recipe), and an allergen-under-pressure request ("suggest a peanut butter cookie" from a peanut-allergic user — must refuse). Four dimensions:

| Metric | How it's measured |
|---|---|
| **Grounding accuracy** | deterministic — every `(ID: N)` is checked against the DB (exists + name matches) |
| **Task success (TSR)** | deterministic — the answer must satisfy the **actual task constraint, verified against the DB**: stated calories match the recipe, a meal plan is under its cap *and* all-vegetarian, a scale factor is numerically correct, a shopping list draws from *both* source recipes and is de-duplicated, an allergen is genuinely excluded, a missing recipe is refused |
| **Tool-usage correctness** | deterministic — did the agent call the expected tool (or, for the out-of-scope prompt, correctly call *none*), recorded via a callback |
| **LLM-as-judge** | a **strict** rubric grades 0–10 + ACCEPT/REJECT + an `issues` list; the judge is fed the ground-truth names/calories for the cited IDs so it *verifies* rather than guesses. If the judge returns an identical score for every case, the report flags it as unreliable rather than trusting the average |

Each run writes **`.json` (full detail + responses), `.csv` (spreadsheet), and `.md` (report)** to `eval_runs/`, named by timestamp + model (e.g. `eval_2026-07-06_00-40-52_qwen3-32b.md`). The scorecard reports the judge-score **range** alongside the average, so a discriminating judge is visible at a glance.

```bash
# run the suite once (uses .env models)
python -m src.eval.evaluate

# a single run gives per-objective scores of 0/1 or 1/1 with no statistical
# weight — average 3–5 runs for a number you can trust
python -m src.eval.evaluate --runs 5

# override models (a strong, independent judge gives the most trustworthy grade)
python -m src.eval.evaluate --model llama-3.3-70b-versatile --judge-model llama-3.3-70b-versatile
```

> Note: the eval uses its own fixed benchmark prompts (the `CASES` list) in a separate process — it never touches your live chat sessions. It measures the **raw agent** (the grounding-guard retry that the app applies on top is disabled here, so the grounding metric stays meaningful). For a trustworthy grade, use a judge at least as strong as the model under test.

---

## 🚀 Running the project

### 1. Set up the environment (Python 3.11)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure `.env`
Create a `.env` in the project root (get a free key at [console.groq.com](https://console.groq.com)):
```bash
GROQ_API_KEY=your_groq_api_key_here

# Models (agent / fallback / summary / eval judge)
GROQ_MODEL=qwen/qwen3-32b
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant
GROQ_SUMMARY_MODEL=llama-3.1-8b-instant
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile

# Generation / memory (optional; sensible defaults)
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048
AGENT_MAX_ITERATIONS=6
MEMORY_KEEP_LAST=8
MEMORY_TRIGGER_AT=14

TOKENIZERS_PARALLELISM=false
```

### 3. Build the recipe database & search index — **required**
The SQLite DB and FAISS index are **not committed** (they're ~3 GB), so you build them once from the source data. The app won't run without them (the retriever loads the FAISS index on startup).

1. **Get the source data** — download **`RAW_recipes.csv`** from the [Food Recommend dataset on Kaggle](https://www.kaggle.com/datasets/nguynphananhc/food-recommend?select=RAW_recipes.csv) and place it in `Data/raw/`.
2. **Preprocess** it into `Data/processed/PROCESSED_recipes.csv` by running the notebook **`Data/data_pre-processing.ipynb`**.
3. **Build the DB and the vector index:**
   ```bash
   python -m src.ingestion.build_database      # PROCESSED_recipes.csv → Data/db/recipes.db
   python -m src.ingestion.build_vectorstore   # recipes.db → Data/vectorstore/ (FAISS)
   ```
   > One-time step; embedding 230k+ recipes takes a while. The `users` table is created automatically on first app run.

### 4. Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501 → **Welcome screen** → Register (new users) or Log in. First run also downloads the embedding model, so give it a moment.

---

## 🗂️ Project structure

```
app.py                     Streamlit app (auth, chat, profile, dark/light)
main.py                    CLI chat loop
src/
  agent/react_agent.py     Tool-calling agent, system prompt, memory, grounding guard
  tools/                   recipe_lookup, nutrition, meal_planner, modify_recipe,
                           shopping_list, ingredient_suggester, resolve, registry
  retrieval/               FAISS semantic retriever
  db/ , services/          SQLite engine, recipe + user services, bcrypt security
  ingestion/               build_database.py, build_vectorstore.py, schema.sql
  eval/evaluate.py         Evaluation harness (grounding / TSR / tools / LLM-judge)
Data/                      recipes.db, FAISS index, raw/processed CSVs
eval_runs/                 saved evaluation reports (json/csv/md)
```

---

## 🔧 Tech stack
**Python** · **LangChain** · **Groq** (Llama / Qwen) · **FAISS** · **SQLite** · **sentence-transformers** (`all-MiniLM-L6-v2`) · **Streamlit** · **bcrypt**

## ⚠️ Notes
- The dataset stores ingredient *names* only (no quantities), so scaling is expressed as a factor and shopping lists omit amounts.
- Nutrition grams are **estimated** from the dataset's % Daily Value.
- Chat memory is in-process (resets if the app restarts); user profiles persist in the DB.
