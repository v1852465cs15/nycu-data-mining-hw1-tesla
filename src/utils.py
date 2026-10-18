from yaml import load, SafeLoader

SETTINGS_PATH = "settings.yml"


def read_settings(path=SETTINGS_PATH):
    with open(path, encoding="utf-8") as f:
        return load(f, Loader=SafeLoader)
