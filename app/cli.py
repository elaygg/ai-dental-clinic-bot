"""Ask the assistant a question from the terminal, without starting the API.

python -m app.cli "How much is a kids' dental cleaning?"
"""

import argparse

from app.rag import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the dental clinic assistant a question.")
    parser.add_argument("question", help="The patient's question, in quotes.")
    args = parser.parse_args()

    result = answer_question(args.question)
    print(result.answer)
    print(f"\nSources: {result.sources}  |  top relevance: {result.top_score}")


if __name__ == "__main__":
    main()
