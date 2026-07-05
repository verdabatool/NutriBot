from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from src.agent.react_agent import build_agent


def run():
    print("NutriChat (agentic + grounded) — type 'exit' to quit\n")

    agent = build_agent()

    # Stable thread id = persistent memory for this run.
    # Change this if you want a "fresh chat".
    config = {"configurable": {"thread_id": "nutrichat-local-1"}}

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            break
        if not user:
            continue

        result = agent.invoke(
            {"input": user},
            config=config,
        )

        if isinstance(result, dict) and "output" in result:
            content = result["output"]
        elif isinstance(result, dict) and "messages" in result:
            last = result["messages"][-1]
            content = getattr(last, "content", None) or getattr(last, "text", None) or str(last)
        else:
            content = str(result)

        print(f"\nBot: {content}\n")


if __name__ == "__main__":
    run()
