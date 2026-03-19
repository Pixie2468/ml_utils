import os
from dotenv import load_dotenv


def _load_dotenv():
    load_dotenv()


def get_env_variable(variable: str):
    return os.getenv(variable)
