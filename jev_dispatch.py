"""
Jev Decision Layer — Phase 1

Two-model architecture:
  - Jev (System One) decides WHICH tool to call + confidence
  - Simple regex extracts free-text args (app names, levels) from the user's sentence

Jev's strength is closed-set decisions (pick from a list), not free-text generation.
So we use Jev for tool selection (Choice over 10 tools) and actionability (Noul),
then parse args ourselves. Later, for closed-set args (Literal types), Jev can fill
those too — see the function_calling cookbook pattern.

Flow:
    user text → Jev: which tool? (Choice) + is this actionable? (Noul)
             → confidence gate (>= 0.7 to auto-run)
             → extract args from text (regex)
             → execute tool
             → return result string
"""

import os
import re
import sys

# --- .env loader (no dependency needed) ---
def load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

load_env(os.path.join(os.path.dirname(__file__), ".env"))

from typesafe_sdk import TypeSafeClient, Choice, Noul
import jev_tools


def build_tool_question() -> Choice:
    """Build the Choice question: which tool should Jev pick?

    The criteria are the tool names — Jev picks one.
    The instructions describe each tool so Jev knows what each does.
    """
    criteria = {}
    descriptions = []
    for name, info in jev_tools.TOOL_REGISTRY.items():
        criteria[name] = None
        descriptions.append(f"{name}: {info['description']}")

    return Choice(
        instructions="Which tool matches the user's request? " + " | ".join(descriptions),
        criteria=criteria,
    )


def extract_app_name(text: str) -> str | None:
    """Extract an app name from user text.

    Handles: 'open Spotify', 'quit Safari', 'close Notes', 'launch Terminal'
    """
    # Common app names on Nik's Mac (from list_running_apps)
    known_apps = [
        "Spotify", "Safari", "Terminal", "Notes", "Finder", "Mail",
        "Messages", "Calendar", "Maps", "Music", "Photos", "Preview",
        "Pages", "Numbers", "Keynote", "Xcode", "TextEdit", "Calculator",
        "Telegram", "WhatsApp", "Brave Browser", "Google Chrome",
        "Microsoft Outlook", "Microsoft Word", "Microsoft Excel",
        "Microsoft Teams", "MSTeams", "Zoom", "Slack", "Discord",
        "VS Code", "CotEditor", "Alacritty", "kitty", "Hermes",
        "wpsoffice", "Notion", "Obsidian", "Figma",
    ]

    # Sort by length (longest first) so "Brave Browser" matches before "Brave"
    known_apps.sort(key=len, reverse=True)

    # Look for known app names in the text
    text_lower = text.lower()
    for app in known_apps:
        if app.lower() in text_lower:
            return app

    # Fallback: word after open/quit/launch/close
    match = re.search(r"(?:open|quit|launch|close|start)\s+(\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    return None


def extract_level(text: str) -> int | None:
    """Extract a volume/brightness level from text.

    Handles: 'set volume to 50', 'turn down to 30', 'volume 70',
             '50%', 'make it louder/quieter' (relative)
    """
    # Direct number
    match = re.search(r"(\d+)\s*%?", text)
    if match:
        return max(0, min(100, int(match.group(1))))

    # Relative terms
    text_lower = text.lower()
    if "mute" in text_lower or "silence" in text_lower:
        return 0
    if "max" in text_lower or "full" in text_lower or "all the way" in text_lower:
        return 100
    if "louder" in text_lower or "brighter" in text_lower or "up" in text_lower:
        return 80  # default "up" level
    if "quieter" in text_lower or "dimmer" in text_lower or "down" in text_lower:
        return 30  # default "down" level

    return None


def dispatch(user_text: str, confirm_threshold: float = 0.7) -> str:
    """
    Send user text to Jev, get a typed decision, execute the tool.

    Returns a human-readable result string.
    """
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key or api_key == "your_key_here":
        return "❌ ERROR: TYPESAFE_API_KEY not set in .env"

    print(f"\n  You: {user_text}")
    print("  → Sending to Jev for decision...")

    with TypeSafeClient() as client:
        result = client.system_one(
            state=user_text,
            questions={
                "tool": build_tool_question(),
                "actionable": Noul(instructions="Is this a real command to act on, or just conversation?"),
            },
        )

        tool_name = result.choices["tool"].choice
        tool_confidence = result.choices["tool"].confidence
        actionable_prob = result.nouls["actionable"].noul

        print(f"  ← Jev: tool={tool_name} (confidence={tool_confidence:.2f}), actionable={actionable_prob:.2f}")

        # Gate 1: is this a command at all?
        if actionable_prob < 0.5:
            return f"🤔 That seems like conversation, not a command. (Jev: P(actionable)={actionable_prob:.2f})"

        # Gate 2: confidence check
        if tool_confidence < confirm_threshold:
            return f"⚠️ Jev is unsure (confidence {tool_confidence:.2f}). Best guess: {tool_name}. Confirm?"

        # Execute
        tool_info = jev_tools.TOOL_REGISTRY.get(tool_name)
        if not tool_info:
            return f"❌ Jev picked '{tool_name}' but that tool doesn't exist."

        fn = tool_info["fn"]
        args = tool_info.get("args", {})

        if args:
            # Extract args from the user's text
            kwargs = {}
            for arg_name in args:
                if arg_name == "level":
                    level = extract_level(user_text)
                    kwargs[arg_name] = level if level is not None else 50  # default
                elif arg_name == "name":
                    app = extract_app_name(user_text)
                    if app is None:
                        return f"❌ Couldn't figure out which app you mean. Try: 'open Spotify'"
                    kwargs[arg_name] = app
                else:
                    return f"❌ Don't know how to extract arg '{arg_name}'"

            print(f"  → Executing: {tool_name}({kwargs})")
            result_str = fn(**kwargs)
        else:
            print(f"  → Executing: {tool_name}()")
            result_str = fn()

        return f"✅ {result_str}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        print("Jev Phase 1 — type a command (or 'quit' to exit)")
        while True:
            user_input = input("\njev> ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            print(dispatch(user_input))
        sys.exit(0)

    print(dispatch(user_input))
