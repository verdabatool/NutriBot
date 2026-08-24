# 🥗 NutriChat — Grounded Recipe & Nutrition Assistant

NutriChat is an **retrieval-grounded chatbot** that helps users find recipes, answer cooking questions, analyze nutrition, plan meals, and build shopping lists — all backed by a real database of **231,637 recipes** [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv). Every recipe, ID, ingredient, step, and nutrition value it states comes from the data, not the model's imagination.

It's built as a **tool-calling LangChain agent** running on **Groq** LLMs, with a **FAISS + SQLite** RAG layer and a **Streamlit** interface (register/login, per-user profiles, dark/light mode).

---

## ✨ What it does (Objectives covered)

1. **Recipe Q&A:** Answers specific questions ("How long do I bake lasagna?"). If a question applies to many recipes, it asks you to clarify; if a detail is missing from the data, it says so and reasons from general knowledge (clearly labeled).
2. **Ingredient-based suggestions:** The "what's in my fridge" problem: give it what you have and it ranks recipes by how many of your ingredients they use.
3. **Nutritional analysis:** Calories + **macros in grams** (protein/carbs/fat…) per serving, and a meal total, converted from the dataset's %DV.
4. **Weekly meal planning:** A full multi-day plan with **breakfast/lunch/dinner** each matched to the right meal type, honoring **diet** (vegetarian/vegan/…) and a **per-day calorie target**.
5. **Recipe modification & scaling:** "Make it gluten-free" or "scale from 6 servings to 2" (returns a scale factor + grounded, clearly-labeled substitutions).
6. **Shopping list:** A consolidated, de-duplicated list from a meal plan or a set of recipes.
7. **Persistent profiles:** Remembers **allergies, diet, and health goals** (stored in the DB, bcrypt-hashed passwords), excludes allergens from results, and can save new preferences stated in chat.

---

## 🧠 How it works (Architecture)

Here is the journey of one question, from the moment you type it to the moment you get an answer:

1. **You ask a question.** You type something like *"suggest a quick vegetarian pasta"* into the chat screen (the **Streamlit app**). If you're logged in, the app also quietly attaches your saved profile including your allergies, diet, and health goal — so the assistant keeps them in mind.

2. **The assistant figures out what you need.** The "brain" (an **AI language model**) reads your question and decides *how* to answer it. It doesn't answer from its own head; instead it picks the right **tool** for the job.

3. **The tools look things up in the real data.** Each tool fetches actual facts from two places:
   - a **search index** that understands semantics (*meaning*) (so "quick pasta" finds relevant recipes even if they don't use those exact words), and
   - a **database** — the structured cookbook (of 231,637 real recipes) holding every recipe's ingredients, steps, tags, and nutrition, plus user accounts.
   The assistant can only talk about recipes these tools actually returned.

4. **Two safety checks run before you see anything.** Once the assistant drafts a reply, the system double-checks it against the database — automatically, without trusting the AI's word:
   - **Did it make up a recipe?** Every recipe number in the answer is verified to be real *and* to match the name shown. If the AI invented one, that turn is thrown away and redone.
   - **Is it safe for you?** If you listed an allergy, every suggested recipe's real ingredients are re-checked; anything containing your allergen is blocked, and the assistant offers a safe alternative instead.

5. **You get a grounded, safe reply** — real recipes, with real IDs you can trust, honoring your preferences.

Two more touches keep it smooth: it **remembers your conversation** (so "make *that one* gluten-free" works), and if the main AI model is ever busy or unavailable, it **automatically switches to a backup model** so the app keeps working.

```
 ┌───────────────────────────────────────────────────────┐
 │  1. App  (Streamlit)                                  │
 │     shows the chat, sign-in, and your profile         │
 └──────────────────────┬────────────────────────────────┘
                        │ your question (+ your saved profile)
                        ▼
 ┌───────────────────────────────────────────────────────┐
 │  2. The "brain"  (AI model via Groq)                  │
 │     reads your question, remembers the chat,          │
 │     and chooses from 8 tools                          │
 └──────────────────────┬────────────────────────────────┘
                        │ runs the chosen tool
                        ▼
 ┌───────────────────────────────────────────────────────┐
 │  3. Tools + data                                      │
 │     the tool fetches real facts from the search       │
 │     index (finds by meaning) + recipe database        │
 └──────────────────────┬────────────────────────────────┘
                        │ draft answer
                        ▼
 ┌───────────────────────────────────────────────────────┐
 │  4. Safety checks                                     │
 │     is every recipe real? free of your allergens?     │
 │     if not ──► redo the answer                        │
 └───────────────────────────────────────────────────────┘
                        │
                        ▼
          a real, safe, grounded reply
```

### The pieces, one by one

Each block in the diagram is a real part of the codebase. Here's what each one is — in plain terms, and what's actually running underneath.

| Layer | In plain terms | Under the hood |
|---|---|---|
| **App** | The screen you interact with — sign in, chat, edit your profile, switch light/dark. | Streamlit web app (`app.py`); a command-line version also exists (`main.py`). Passwords are stored safely (bcrypt-hashed). |
| **The "brain" (agent)** | The reasoning part that reads your question, remembers the chat so far, and decides *which tool* to use. | A tool-calling **LangChain** agent running on a **Groq** AI model. It calls tools natively (not by parsing text), and keeps a running summary of long chats so it doesn't forget earlier turns. |
| **Tools** | The assistant's toolbox — one specialized helper per job. | 8 grounded tools: *find recipes, cooking steps, nutrition, meal plan, adjust recipe, shopping list, ingredient-based suggestions, name→ID lookup* — plus a *save-profile* tool for logged-in users. Each returns only real data from the sources below. |
| **Search index** | Finds recipes by **meaning**, so "quick pasta" matches even when those exact words aren't used. | A **FAISS** vector index built from recipe text using sentence-embeddings (`all-MiniLM-L6-v2`). It returns candidate recipe IDs, which the database then fills in with full details. |
| **Database** | The structured cookbook — and the store for user accounts. | A **SQLite** database: recipes, their ingredients, tags, and nutrition, plus the users table. It's the single source of truth every tool reads from. |
| **Safety checks** | Two automatic gatekeepers that inspect the draft answer before you ever see it. | The **grounding guard** verifies every recipe ID is real and correctly named (redoes the turn if not); the **allergen guard** re-checks each cited recipe's real ingredients against your allergies (blocks it and offers an alternative if unsafe). Both are deterministic — they check the database directly, not the AI's word. |

### Engineering highlights
- **Grounded by design** — every recipe ID in a reply is verified against the database (must exist *and* its name must match); anything hallucinated is caught and the answer is regenerated, so the bot can't invent recipes.
- **Deterministic allergy safety** — allergens are filtered out *and* re-checked against each recipe's real ingredients, so a user's safety never depends on the AI remembering to be careful.
- **Resilient by default** — automatically falls back to a backup model when the main one is rate-limited (429), sent too large a request (413), or unavailable (404, e.g. decommissioned), so the app keeps answering.

**Models** (all configurable in `.env`): primary `qwen/qwen3.6-27b`, backup fallback `openai/gpt-oss-20b`, chat-memory summarizer `llama-3.1-8b-instant`, evaluation judge `llama-3.3-70b-versatile` (kept **stronger than and independent of** the models it grades, so it scores reliably rather than rubber-stamping).

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

Each run writes **`.json` (full detail + responses), `.csv` (spreadsheet), and `.md` (report)** to `eval_runs/`, named by timestamp + model (e.g. `eval_2026-08-12_18-12-01_qwen3-6-27b.md`). The scorecard reports the judge-score **range** alongside the average, so a discriminating judge is visible at a glance.

```bash
# run the suite once (uses .env models)
python -m src.eval.evaluate

# a single run gives per-objective scores of 0/1 or 1/1 with no statistical
# weight — average 3–5 runs for a number you can trust
python -m src.eval.evaluate --runs 5

# override models — keep the judge STRONGER THAN and a DIFFERENT FAMILY from the
# model under test, so it grades reliably instead of rubber-stamping its own kind
python -m src.eval.evaluate --model openai/gpt-oss-20b --judge-model llama-3.3-70b-versatile
```

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

# Models (agent / backup fallback / chat-memory summarizer / eval judge)
GROQ_MODEL=qwen/qwen3.6-27b
GROQ_FALLBACK_MODEL=openai/gpt-oss-20b
GROQ_SUMMARY_MODEL=llama-3.1-8b-instant
GROQ_JUDGE_MODEL=llama-3.3-70b-versatile

# Generation / memory (optional; sensible defaults)
LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048
AGENT_MAX_ITERATIONS=8
MEMORY_KEEP_LAST=8
MEMORY_TRIGGER_AT=14

TOKENIZERS_PARALLELISM=false
```

### 3. Build the recipe database & search index 
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
Open http://localhost:8501 → **Welcome screen** → Register (new users) or Log in. First run also downloads the embedding model, so it takes a moment.

---


