from pathlib import Path

IGNORE_DIRS = {".git", "__pycache__", "venv", ".venv", "node_modules"}

TEXT_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
}

with open("project_prompt.txt", "w", encoding="utf-8") as out:
    for p in Path(".").rglob("*"):
        if not p.is_file():
            continue

        if any(part in IGNORE_DIRS for part in p.parts):
            continue

        if p.suffix.lower() not in TEXT_EXTENSIONS:
            continue

        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            continue

        out.write(f"\n=== FILE: {p} ===\n")
        out.write(content)
        out.write("\n")

print("Generated project_prompt.txt")
