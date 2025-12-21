# from __future__ import annotations
#
# from dotenv import load_dotenv
# load_dotenv()
#
# from src.agent.react_agent import build_agent
#
#
# def run():
#     print("NutriChat (agentic + grounded) — type 'exit' to quit\n")
#
#     agent = build_agent()
#
#     # Stable thread id = persistent memory for this run.
#     # Change this if you want a "fresh chat".
#     config = {"configurable": {"thread_id": "nutrichat-local-1"}}
#
#     while True:
#         user = input("You: ").strip()
#         if user.lower() == "exit":
#             break
#         if not user:
#             continue
#
#         # create_agent expects messages
#         result = agent.invoke(
#             {"messages": [{"role": "user", "content": user}]},
#             config=config,
#         )
#
#         # result["messages"] is a list; last one is the assistant message
#         last = result["messages"][-1]
#         # Some message objects expose .content, some .text; handle both safely
#         content = getattr(last, "content", None) or getattr(last, "text", None) or str(last)
#
#         print(f"\nBot: {content}\n")
#
#
# if __name__ == "__main__":
#     run()
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from src.agent.react_agent import build_agent


def get_agent():
    return build_agent()

def run():
    print("NutriChat (agentic + grounded) — type 'exit' to quit\n")

    agent = get_agent()

    # Stable thread id = persistent memory for this run.
    # Change this if you want a "fresh chat".
    config = {"configurable": {"thread_id": "nutrichat-local-1"}}

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            break
        if not user:
            continue

        # create_agent expects messages
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user}]},
            config=config,
        )

        # result["messages"] is a list; last one is the assistant message
        last = result["messages"][-1]
        # Some message objects expose .content, some .text; handle both safely
        content = getattr(last, "content", None) or getattr(last, "text", None) or str(last)

        print(f"\nBot: {content}\n")


if __name__ == "__main__":
    run()