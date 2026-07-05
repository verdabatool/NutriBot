from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from src.agent import ChatPipeline


def run():
    print("NutriChat (agentic + grounded) — type 'exit' to quit\n")

    pipeline = ChatPipeline()

    # Stable thread id = persistent memory for this run.
    # Change this if you want a "fresh chat".
    thread_id = "nutrichat-local-1"

    while True:
        user = input("You: ").strip()
        if user.lower() == "exit":
            break
        if not user:
            continue

        content = pipeline.answer(user, thread_id=thread_id)
        print(f"\nBot: {content}\n")


if __name__ == "__main__":
    run()