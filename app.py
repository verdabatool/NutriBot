
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.agent import ChatPipeline
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
if "pipeline" not in st.session_state:
    st.session_state.pipeline = ChatPipeline()

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
            # Rebuild the pipeline bound to this user so it can persist profile info.
            st.session_state.pipeline = ChatPipeline(username=username)
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
            st.session_state.pop("pipeline", None)
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

    # Ensure the pipeline exists (bound to the user so it can persist profile info).
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = ChatPipeline(username=st.session_state.username or None)

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

        with st.chat_message("assistant", avatar="🥗"):
            status_box = st.empty()
            # The pipeline owns the grounded path: invoke → auto-fallback on
            # rate-limit/overflow → grounding guard → sanitize → finish cut-offs.
            # It never fabricates recipes; on failure it returns a safe message.
            with st.spinner("Cooking up an answer…"):
                final_text = st.session_state.pipeline.answer(
                    user_input,
                    thread_id=st.session_state.thread_id,
                    user_preference=_build_user_context(),
                    complete_truncated=True,
                    on_status=lambda m: status_box.caption(m),
                )
            status_box.empty()
            st.markdown(final_text)

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