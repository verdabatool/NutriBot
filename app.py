import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.agent.react_agent import build_agent

# --- Page config ---
st.set_page_config(
    page_title="Nutribot",
    page_icon="🥗",
    layout="centered",
)

# --- Custom CSS ---
st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #F0FFEB;
}


/* User messages → right aligned */
.stChatMessage.user {
    flex-direction: row-reverse;
    text-align: right;
}

/* User bubble styling */
.stChatMessage.user .stMarkdown {
    background: rgba(255, 255, 255, 0.25);
    border-radius: 20px 20px 5px 20px;
    padding: 12px 16px;
    max-width: 70%;
    margin-left: auto;
}

/* Assistant bubble styling */
.stChatMessage.assistant .stMarkdown {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px 20px 20px 5px;
    padding: 12px 16px;
    max-width: 70%;
    margin-right: auto;
}


/* Hide Streamlit header gap */
header { visibility: hidden; }


</style>
""", unsafe_allow_html=True)


st.title("🥗 Nutribot")

# --- Initialize agent once ---
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

# Stable thread id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "nutribot-streamlit-1"

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hey there! I’m **Nutribot 🥗** — your smart food companion here to whip up recipes, build healthy habits, and dish out personalized nutrition tips 🍽️✨"
        }
    ]

for msg in st.session_state.messages:
    avatar = "🥗" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


suggestions = [
    "🛒 Make a shopping list",
    "📋 Share recipe steps",
    "🍎 Share nutritional value",
    "🍽️ Recommend healthy recipes",
]

# --- User input ---
user_input = st.chat_input("Ask me anything about food & nutrition…")

cols = st.columns(len(suggestions))
for col, prompt in zip(cols, suggestions):
    if col.button(prompt):
        st.session_state.suggested_prompt = prompt
        
if "suggested_prompt" in st.session_state:
    user_input = st.session_state.pop("suggested_prompt")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🥗"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown("_🥗 Nutribot is typing..._")

        with st.spinner("Cooking up something smart… 🧠🍳"):
            result = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )

        typing_placeholder.empty()

        last = result["messages"][-1]
        content = getattr(last, "content", None) or getattr(last, "text", None) or str(last)
        st.markdown(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
