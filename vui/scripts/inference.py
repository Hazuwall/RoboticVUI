from typing import Optional
import vui


def main(duration: Optional[float] = None):
    def word_handler(word, weight):
        print(word + ", " + str(weight))

    vui.run(word_handler, duration)


if __name__ == "__main__":
    main()
