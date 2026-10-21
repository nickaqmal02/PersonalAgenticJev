import subprocess

def volume(level: int) -> str:
    """Set system output volume 0-100."""
    level = max(0, min(100, int(level)))          # clamp — LLMs pass weird values
    subprocess.run(
        ["osascript", "-e", f"set volume output volume {level}"],
        check=True,                                # raise if AppleScript fails
    )
    return f"volume set to {level}"

def open_app(name: str) -> str:
    """Open a macOS application by name"""
    subprocess.run(
        ["open", "-a", name], check=True
    )
    return f"{name} opened"

def quit_app(name: str) -> str:
    """Quit a running macOS application by name."""
    subprocess.run(
        ["osascript", "-e", f'tell application "{name}" to quit'],
        check=True,
    )
    return f"{name} quit"
