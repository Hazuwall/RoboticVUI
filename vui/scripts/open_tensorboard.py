import subprocess
import webbrowser
import vui.infrastructure.locator as locator


def main():
    logs_path = locator.get_filesystem_provider().get_logs_dir()

    try:
        proc = subprocess.Popen(["tensorboard", "--logdir", logs_path])
        webbrowser.open("http://localhost:6006/", new=0, autoraise=True)
        proc.communicate()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
