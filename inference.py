import locator


def word_handler(word, weight):
    print(word + ", " + str(weight))


vui = locator.get_voice_user_interface(word_handler)
vui.run()
