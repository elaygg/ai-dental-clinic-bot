"""Ask the assistant a question from the terminal, without starting the API.

python -m app.cli "How much is a kids' dental cleaning?"
"""

import argparse

from app.assistant import reply_to


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the dental clinic assistant a question.")
    parser.add_argument("question", help="The patient's question, in quotes.")
    args = parser.parse_args()

    reply = reply_to(args.question)
    print(reply.answer)
    print(f"\nSources: {reply.sources}  |  handoff: {reply.handoff_reason or 'no'}")


if __name__ == "__main__":
    main()
