from termcolor import colored
from pprint import pprint
import vui.infrastructure.locator as locator


def main():
    status("Working...")
    evaluator = locator.get_evaluator()
    result = evaluator.evaluate().get_summary()

    status("Printing result...")
    pprint(vars(result))


def status(text: str):
    print(colored("[Evaluate]: " + text, 'green'))


if __name__ == "__main__":
    main()
