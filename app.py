
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # for short continuation when cut off

# Load environment variables
load_dotenv()

from src.agent.react_agent import (
    build_agent, FALLBACK_MODEL, ungrounded_ids, forget_last_turn,
)
from src.services.users import (
    authenticate_user, register_user, update_user_profile,
    update_user_preference, get_user_by_username,
)

# --- Page config ---
st.set_page_config(
    page_title="Nutribot",
    page_icon="🥗",
    layout="centered",
)

# --- Theming (light / dark) -------------------------------------------------
LIGHT_THEME = {
    "grad1": "#F4FBF6", "grad2": "#E3F4EA",
    "text": "#14281E", "muted": "#51655A",
    "card": "#FFFFFF", "card_border": "#D9E9DF",
    "user_bubble": "#2E7D5B", "user_text": "#FFFFFF",
    "bot_bubble": "#FFFFFF", "bot_text": "#14281E", "bot_border": "#DCEBE2",
    "accent": "#2E7D5B", "accent_text": "#FFFFFF",
    "input_bg": "#FFFFFF", "input_text": "#14281E", "input_border": "#CFE3D7",
    "placeholder": "#8AA093", "shadow": "rgba(20,40,30,0.08)",
}
DARK_THEME = {
    "grad1": "#0E1512", "grad2": "#13221B",
    "text": "#E9F2EC", "muted": "#9DB2A6",
    "card": "#16211B", "card_border": "#28382F",
    "user_bubble": "#2E9269", "user_text": "#F3FBF6",
    "bot_bubble": "#1A251F", "bot_text": "#E9F2EC", "bot_border": "#28382F",
    "accent": "#43C08A", "accent_text": "#07130D",
    "input_bg": "#16211B", "input_text": "#E9F2EC", "input_border": "#2C3D34",
    "placeholder": "#7E9488", "shadow": "rgba(0,0,0,0.35)",
}


def apply_theme(dark: bool) -> None:
    """Inject theme-aware CSS. Guarantees readable text in both modes."""
    p = DARK_THEME if dark else LIGHT_THEME
    st.markdown(f"""
<style>
/* ---- app shell ---- */
.stApp {{
  background: linear-gradient(160deg, {p['grad1']} 0%, {p['grad2']} 100%);
  color: {p['text']};
}}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; max-width: 840px; }}

/* ---- base text everywhere ---- */
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] * {{
  color: {p['text']};
}}
h1, h2, h3, h4, h5 {{ color: {p['text']} !important; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{
  color: {p['muted']} !important;
}}
hr {{ border-color: {p['card_border']}; }}

/* ---- forms / cards ---- */
[data-testid="stForm"] {{
  background: {p['card']};
  border: 1px solid {p['card_border']};
  border-radius: 18px;
  padding: 26px 24px;
  box-shadow: 0 8px 28px {p['shadow']};
}}

/* ---- text inputs ---- */
.stTextInput input, .stTextArea textarea,
[data-baseweb="input"] input, [data-baseweb="textarea"] textarea {{
  background: {p['input_bg']} !important;
  color: {p['input_text']} !important;
}}
[data-baseweb="input"], [data-baseweb="base-input"] {{
  background: {p['input_bg']} !important;
  border: 1px solid {p['input_border']} !important;
  border-radius: 10px !important;
}}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {{
  color: {p['placeholder']} !important;
}}
.stTextInput label, .stTextArea label, [data-testid="stWidgetLabel"] * {{
  color: {p['text']} !important; font-weight: 600;
}}

/* ---- buttons ---- */
.stButton > button, .stFormSubmitButton > button {{
  background: {p['accent']};
  color: {p['accent_text']} !important;
  border: none; border-radius: 10px;
  padding: 9px 16px; font-weight: 600;
  transition: transform .05s ease, filter .15s ease;
}}
.stButton > button *, .stFormSubmitButton > button * {{ color: {p['accent_text']} !important; }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ filter: brightness(1.08); }}
.stButton > button:active {{ transform: translateY(1px); }}

/* ---- chat messages ---- */
[data-testid="stChatMessage"] {{
  background: {p['bot_bubble']};
  border: 1px solid {p['bot_border']};
  border-radius: 16px;
  padding: 4px 16px;
  margin-bottom: 10px;
  box-shadow: 0 2px 12px {p['shadow']};
}}
[data-testid="stChatMessage"] * {{ color: {p['bot_text']} !important; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
  background: {p['user_bubble']}; border-color: transparent;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) *,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) * {{
  color: {p['user_text']} !important;
}}

/* ---- chat input ---- */
[data-testid="stChatInput"] {{
  background: {p['input_bg']};
  border: 1px solid {p['input_border']};
  border-radius: 14px;
}}
[data-testid="stChatInput"] textarea {{
  color: {p['input_text']} !important; background: transparent !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{ color: {p['placeholder']} !important; }}

/* ---- toggle ---- */
[data-testid="stToggle"] label p, .stToggle label p {{ color: {p['text']} !important; }}
</style>
""", unsafe_allow_html=True)

def clean_response(text: str) -> str:
    """Remove parser noise and troubleshooting footer; dedupe lines."""
    import re
    if not isinstance(text, str):
        text = str(text)
    # Remove common parser noise
    text = re.sub(r"(?i)could not parse LLM output:\s*", "", text)
    text = re.sub(r"(?i)output_parsing_failure.*", "", text)
    # Strip troubleshooting footer
    text = re.split(r"For troubleshooting", text, flags=re.IGNORECASE)[0]
    # Dedupe lines
    lines = [ln.rstrip() for ln in text.split("\n")]
    out, seen = [], set()
    for ln in lines:
        key = ln.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(ln)
    return "\n".join(out).strip()

# --- Hide agent scaffolding and tool errors from users ---
def _sanitize_agent_text(text: str) -> str:
    """Keep only the user-facing content; strip Thought/Action and tool/parser errors."""
    import re
    s = str(text or "")
    # Prefer the Final Answer block if present
    m = re.search(r"(?:\*\*Final Answer:\*\*|Final Answer:)\s*(.+)", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        s = m.group(1).strip()
    else:
        # Remove internal ReAct scaffolding lines
        s = re.sub(r"(?m)^\s*Thought:\s.*$", "", s)
        s = re.sub(r"(?m)^\s*Action:\s.*$", "", s)
        s = re.sub(r"(?m)^\s*Action Input:\s.*$", "", s)
        # Remove tool error leakage and parser noise
        s = re.sub(r"(?i)resolve_recipe_by_name is not a valid tool, try one of \[.*?\]\.?", "", s)
        s = re.sub(r"(?i)could not parse LLM output:\s*", "", s)
        s = re.sub(r"(?i)sorry, I couldn't parse the response\. please try again\.", "", s)
        # Strip fenced code blocks that sometimes leak
        s = re.sub(r"```+.*?```+", "", s, flags=re.DOTALL)

    # Strip leaked pseudo tool-calls a model may emit as plain text, e.g.
    #   <nutrition_analyzer>{"recipe_ids": [15253]}</nutrition_analyzer>
    #   <function=...>...</function> / <tool_call>...</tool_call>
    s = re.sub(r"<([a-zA-Z_][\w\-]*)>\s*\{.*?\}\s*</\1>", "", s, flags=re.DOTALL)
    s = re.sub(r"</?(function|tool|tool_call|tool_calls|invoke|parameter)\b[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<\|[^>]*\|>", "", s)

    result = clean_response(s)
    # Never show a blank or junk-only reply.
    if not result.strip():
        return ("I couldn't complete that one just now — please try again, or "
                "rephrase your question (e.g. name the dish or its recipe ID).")
    return result

# --- Truncation helpers (lightweight, non-disruptive) ---
def _is_truncated(text: str) -> bool:
    """Heuristic to detect mid-sentence/list cutoffs."""
    import re
    if not text or len(text) < 60:
        return False
    tail = text.strip()[-120:]
    if tail.endswith((":", "-", "–", "—", "•")):
        return True
    if not text.strip().endswith((".", "!", "?")) and len(text) > 200:
        return True
    # common half-step range like "Blend 30‑45"
    if re.search(r"\b\d+\s*[–-]\s*\d+$", tail):
        return True
    return False

def _finish_answer(partial: str) -> str:
    """Complete the current sentence/list item without adding new sections."""
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_tokens=256)
    except Exception:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, max_tokens=256)
    prompt = (
        "Continue the assistant's answer exactly from where it stopped. "
        "Finish the current sentence or list item, keep the same tone, and do not start new sections. "
        "Do not introduce new facts; only complete what's already begun. "
        "Output only the continuation.\n\n"
        f"Answer so far:\n{partial.strip()}\n\nContinuation:"
    )
    resp = llm.invoke(prompt)
    return getattr(resp, "content", None) or getattr(resp, "text", None) or ""

# --- Soft fallback: produce a helpful best-effort answer when tools/parsing fail ---
def _soft_fallback_answer(user_text: str, preference: str = "") -> str:
    """Generate a friendly answer (options or an approximate recipe) without tool markup."""
    try:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, max_tokens=512)
    except Exception:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, max_tokens=512)
    prompt = (
        "You are NutriChat, a friendly recipe assistant. Tools may be unavailable.\n"
        "Give a helpful answer anyway:\n"
        "- If the user asks for ideas, offer 3–5 engaging options with one-line blurbs.\n"
        "- If the user asks for a specific recipe, provide a concise, approximate home-style recipe "
        "(title, ingredients, numbered steps). Keep it plausible and complete.\n"
        "- Do NOT include Thought/Action or any tool markup. No code blocks. End with a friendly next step.\n"
        f"User preference: {preference or 'None'}\n"
        f"User: {user_text}\n"
        "Assistant:"
    )
    resp = llm.invoke(prompt)
    text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
    return clean_response(text)

def extract_final_answer_from_error(error: Exception, user_text: str = "", preference: str = "") -> str:
    """Best-effort extract of Final Answer from errors; otherwise return a soft fallback."""
    import re
    msg = str(error)
    m = re.findall(
        r"(?:\*\*Final Answer:\*\*|Final Answer:)\s*(.+?)(?=(?:\n\*\*Thought:|\nThought:|\nAction:|\n\*\*|$))",
        msg, flags=re.IGNORECASE | re.DOTALL
    )
    if m:
        return clean_response(m[-1].strip())
    # If we can't extract a Final Answer, produce a helpful fallback
    return _soft_fallback_answer(user_text, preference)

# --- Initialize session state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "user_preference" not in st.session_state:
    st.session_state.user_preference = ""
if "user_allergies" not in st.session_state:
    st.session_state.user_allergies = ""
if "user_diet" not in st.session_state:
    st.session_state.user_diet = ""
if "user_goal" not in st.session_state:
    st.session_state.user_goal = ""
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "auth_page" not in st.session_state:
    st.session_state.auth_page = "welcome"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    base_user = st.session_state.username or "guest"
    st.session_state.thread_id = f"nutrichat-ui-{base_user}"
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

def _build_user_context() -> str:
    """Compose the profile the agent should honor (preference, allergies, diet, goal)."""
    parts = []
    if st.session_state.user_preference:
        parts.append(f"Preference: {st.session_state.user_preference}")
    if st.session_state.user_allergies:
        parts.append(f"ALLERGIES (never suggest these): {st.session_state.user_allergies}")
    if st.session_state.user_diet:
        parts.append(f"Diet: {st.session_state.user_diet}")
    if st.session_state.user_goal:
        parts.append(f"Health goal: {st.session_state.user_goal}")
    return " | ".join(parts) if parts else "No preference"


# --- Auth Pages ---
def show_welcome_page():
    """Landing screen: route first-time users to Register, returning users to Login."""
    st.title("🥗 Welcome to Nutribot")
    st.subheader("Your grounded recipe & nutrition assistant.")
    st.write("")
    st.markdown("**Are you new here, or already a member?**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 I'm new — Register", use_container_width=True):
            st.session_state.auth_page = "register"
            st.rerun()
    with col2:
        if st.button("🔐 I'm a member — Log in", use_container_width=True):
            st.session_state.auth_page = "login"
            st.rerun()


def show_login_page():
    """Display the login page."""
    st.title("🥗 Nutribot")
    st.subheader("Welcome back! Please log in.")
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        col1, col2 = st.columns(2)
        with col1:
            login_btn = st.form_submit_button("🔐 Login", use_container_width=True)
        with col2:
            register_btn = st.form_submit_button("📝 Register", use_container_width=True)
    if login_btn:
        success, message, user_data = authenticate_user(username, password)
        if success:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.user_preference = user_data.get("preference") or ""
            st.session_state.user_allergies = user_data.get("allergies") or ""
            st.session_state.user_diet = user_data.get("diet_type") or ""
            st.session_state.user_goal = user_data.get("health_goal") or ""
            st.session_state.thread_id = f"nutrichat-ui-{username}"
            # Rebuild the agent bound to this user so it can persist profile info.
            st.session_state.agent = build_agent(username=username)
            st.session_state.pop("fallback_agent", None)  # rebuild bound to user on demand
            st.success(message)
            st.rerun()
        else:
            st.error(message)
    if register_btn:
        st.session_state.auth_page = "register"
        st.rerun()

def show_register_page():
    """Display the registration page."""
    st.title("🥗 Nutribot")
    st.subheader("Create your account")
    with st.form("register_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            username = st.text_input("Username", placeholder="Choose a username")
        with col2:
            password = st.text_input("Password", type="password", placeholder="Choose a password")
        with col3:
            preference = st.text_input("Food Preference", placeholder="e.g., I like vegan food")
        col4, col5, col6 = st.columns(3)
        with col4:
            allergies = st.text_input("Allergies", placeholder="e.g., peanuts, shellfish")
        with col5:
            diet_type = st.text_input("Diet", placeholder="e.g., vegetarian, keto")
        with col6:
            health_goal = st.text_input("Health Goal", placeholder="e.g., calorie deficit")
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            register_btn = st.form_submit_button("✅ Register", use_container_width=True)
        with col_btn2:
            back_btn = st.form_submit_button("⬅️ Back to Login", use_container_width=True)
    if register_btn:
        success, message = register_user(
            username, password, preference,
            allergies=allergies, diet_type=diet_type, health_goal=health_goal,
        )
        if success:
            st.success(message)
            st.session_state.auth_page = "login"
            st.rerun()
        else:
            st.error(message)
    if back_btn:
        st.session_state.auth_page = "login"
        st.rerun()

def show_chat_page():
    """Display the main chat page (frontend over main.py agent)."""
    col_title, col_user, col_logout = st.columns([3, 2, 1])
    with col_title:
        st.title("🥗 Nutribot")
    with col_user:
        st.markdown(f"👤 **{st.session_state.username or 'Guest'}**")
        if st.session_state.user_preference:
            st.caption(f"🍽️ {st.session_state.user_preference}")
        if st.session_state.user_allergies:
            st.caption(f"🚫 Allergies: {st.session_state.user_allergies}")
        if st.session_state.user_goal:
            st.caption(f"🎯 {st.session_state.user_goal}")
    with col_logout:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_preference = ""
            st.session_state.user_allergies = ""
            st.session_state.user_diet = ""
            st.session_state.user_goal = ""
            st.session_state.messages = []
            st.session_state.thread_id = "nutrichat-ui-guest"
            st.session_state.auth_page = "welcome"
            st.session_state.pop("agent", None)
            st.session_state.pop("fallback_agent", None)
            st.rerun()

    # --- Editable, persistent profile ---
    with st.expander("⚙️ My Profile (saved to your account)"):
        # Load the latest profile from the DB so the form never shows stale values
        # (e.g. after the chat persisted a new allergy) and can't overwrite them.
        if st.session_state.username:
            _u = get_user_by_username(st.session_state.username)
            if _u:
                st.session_state.user_preference = _u.get("preference") or ""
                st.session_state.user_allergies = _u.get("allergies") or ""
                st.session_state.user_diet = _u.get("diet_type") or ""
                st.session_state.user_goal = _u.get("health_goal") or ""
        with st.form("profile_form"):
            pref_in = st.text_input("Food preference", value=st.session_state.user_preference)
            alg_in = st.text_input("Allergies", value=st.session_state.user_allergies,
                                   placeholder="e.g., peanuts, shellfish")
            diet_in = st.text_input("Diet", value=st.session_state.user_diet,
                                    placeholder="e.g., vegetarian, keto")
            goal_in = st.text_input("Health goal", value=st.session_state.user_goal,
                                    placeholder="e.g., calorie deficit")
            if st.form_submit_button("💾 Save profile"):
                ok, msg = update_user_profile(
                    st.session_state.username,
                    allergies=alg_in, diet_type=diet_in, health_goal=goal_in,
                )
                update_user_preference(st.session_state.username, pref_in)
                st.session_state.user_preference = pref_in
                st.session_state.user_allergies = alg_in
                st.session_state.user_diet = diet_in
                st.session_state.user_goal = goal_in
                st.success("Profile saved." if ok else msg)

    st.markdown("---")

    # Ensure agent exists (bound to the user so it can persist profile info).
    if "agent" not in st.session_state:
        st.session_state.agent = build_agent(username=st.session_state.username or None)

    # Seed chat with a friendly message once
    if not st.session_state.messages:
        intro = "👋 Hey! I'm Nutribot 🥗 — I can find recipes, share instructions, nutrition, meal plans, and shopping lists."
        if st.session_state.user_preference:
            intro += f" I’ll keep your preference in mind: **{st.session_state.user_preference}**."
        st.session_state.messages.append({"role": "assistant", "content": intro})

    # Render history
    for msg in st.session_state.messages:
        avatar = "🥗" if msg["role"] == "assistant" else None
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask me anything about food & nutrition…")

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Trim very long input to reduce token usage (avoid Groq 413)
        context_input = (user_input or "")[:600]

        with st.chat_message("assistant", avatar="🥗"):
            stream_box = st.empty()
            stream_text = ""

            # Single-shot invoke inside a spinner. This is the grounded path
            # (memory + tools) and avoids the old streaming loop that leaked raw
            # intermediate step-dicts into the reply. On failure we show a safe
            # message and NEVER fabricate recipes.
            try:
                with st.spinner("Cooking up an answer…"):
                    result = st.session_state.agent.invoke(
                        {
                            "input": context_input,
                            "user_preference": _build_user_context(),
                        },
                        config={"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                if isinstance(result, dict) and "output" in result:
                    stream_text = result["output"]
                elif isinstance(result, dict) and "messages" in result and result["messages"]:
                    last = result["messages"][-1]
                    stream_text = getattr(last, "content", None) or getattr(last, "text", None) or str(last)
                else:
                    stream_text = str(result)
            except Exception as e:
                # Log the real error (so the cause is visible in the app logs)
                # and show a cause-specific, non-hallucinating message.
                import traceback
                traceback.print_exc()
                err = str(e).lower()
                is_rate = any(t in err for t in ("rate limit", "rate_limit", "429",
                                                 "tokens per day", "tpd", "quota"))
                is_overflow = any(t in err for t in ("413", "request too large", "too large"))
                if is_rate or is_overflow:
                    # Auto-retry on the lighter fallback model (handles both the
                    # rate limit and requests too large for the primary model).
                    try:
                        if "fallback_agent" not in st.session_state:
                            st.session_state.fallback_agent = build_agent(
                                username=st.session_state.username or None,
                                model=FALLBACK_MODEL,
                            )
                        with st.spinner(f"Main model busy — retrying on {FALLBACK_MODEL}…"):
                            result = st.session_state.fallback_agent.invoke(
                                {"input": context_input, "user_preference": _build_user_context()},
                                config={"configurable": {"thread_id": st.session_state.thread_id}},
                            )
                        stream_text = result.get("output") if isinstance(result, dict) else str(result)
                    except Exception as e2:
                        traceback.print_exc()
                        stream_text = (
                            "⏳ **Usage limit reached** and the fallback model is also "
                            "unavailable right now. Please wait a few minutes and try again."
                        )
                elif "locked" in err:
                    stream_text = (
                        "🔒 The database is busy — it may be open in another tool "
                        "(like DB Browser for SQLite). Please close it and try again."
                    )
                else:
                    stream_text = (
                        "Sorry, I hit a snag with that one. Could you rephrase it — "
                        "or, if you meant one of the options above, tell me the recipe "
                        "name or its number?"
                    )

            # --- Grounding guard ---------------------------------------------
            # If the model fabricated recipe IDs (answered from memory instead of
            # the DB), purge that turn from memory and retry once, forcing a
            # database lookup. Guarantees no fake recipe IDs reach the user.
            if ungrounded_ids(stream_text):
                forget_last_turn(st.session_state.thread_id)
                try:
                    with st.spinner("Verifying against the database…"):
                        retry = st.session_state.agent.invoke(
                            {
                                "input": context_input + (
                                    "\n\n[IMPORTANT: Do NOT answer from memory. Call "
                                    "recipe_lookup and use ONLY the real recipe names "
                                    "and IDs it returns. Never invent IDs.]"
                                ),
                                "user_preference": _build_user_context(),
                            },
                            config={"configurable": {"thread_id": st.session_state.thread_id}},
                        )
                    retry_text = retry.get("output") if isinstance(retry, dict) else str(retry)
                    if not ungrounded_ids(retry_text):
                        stream_text = retry_text
                    else:
                        forget_last_turn(st.session_state.thread_id)
                        stream_text = (
                            "I caught myself about to answer from memory instead of the "
                            "database. Please ask again (e.g. “quick pasta recipes”) and "
                            "I'll pull real, grounded recipes."
                        )
                except Exception:
                    import traceback; traceback.print_exc()
                    stream_text = (
                        "I caught myself about to answer from memory instead of the "
                        "database. Please ask again and I'll pull grounded recipes."
                    )

            stream_box.markdown(_sanitize_agent_text(stream_text))

        final_text = _sanitize_agent_text(stream_text)
        # Gracefully finish cut-off responses (keeps the same flow)
        if _is_truncated(final_text):
            cont = _finish_answer(final_text)
            if cont.strip():
                final_text = _sanitize_agent_text(final_text + "\n" + cont.strip())
        st.session_state.messages.append({"role": "assistant", "content": final_text})

# --- Main App Logic ---
apply_theme(st.session_state.dark_mode)

# Top bar: theme toggle (right-aligned, shown on every page)
_spacer, _toggle_col = st.columns([6, 1.4])
with _toggle_col:
    st.toggle("🌙 Dark", key="dark_mode", help="Switch between light and dark mode")

if not st.session_state.logged_in:
    if st.session_state.auth_page == "welcome":
        show_welcome_page()
    elif st.session_state.auth_page == "login":
        show_login_page()
    else:
        show_register_page()
else:
    show_chat_page()