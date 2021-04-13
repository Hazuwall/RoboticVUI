from termcolor import colored
import vui.infrastructure.locator as locator


def main():
    status("Working...")
    evaluator = locator.get_evaluator()
    result = evaluator.evaluate()

    status("Result: {}".format(result))


def status(text: str):
    print(colored("[Evaluate]: " + text, 'green'))


if __name__ == "__main__":
    main()
