from typing import Optional
import locator


def main(duration: Optional[float] = None):
    def word_handler(word, weight):
        print(word + ", " + str(weight))

    vui = locator.get_voice_user_interface(word_handler)
    vui.run(duration)


if __name__ == "__main__":
    main()
