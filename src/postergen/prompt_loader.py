from pathlib import Path


_PROMPT_DIR = Path(__file__).with_name("prompts")


def load_prompt(name: str, **values: object) -> str:
    prompt = (_PROMPT_DIR / name).read_text(encoding="utf-8")
    for key, value in values.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt
