"""
Jev Brain — DeepSeek LLM Layer

The conversation/planning layer. Sits between the user and Jev (System One).

Flow:
    user text → DeepSeek extracts structured intent (tool + args)
              → Jev validates tool choice (confidence gate)
              → jev_tools executes
              → DeepSeek writes a natural language reply

DeepSeek is the "understanding" — Jev is the "validation."
DeepSeek can hallucinate; Jev can't (closed output space).
"""

import os
import json
import sys

# --- .env loader ---
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

from openai import OpenAI
import jev_tools


# DeepSeek uses the OpenAI-compatible API
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def build_tool_catalog() -> str:
    """Build a description of all available tools for the system prompt."""
    lines = []
    for name, info in jev_tools.TOOL_REGISTRY.items():
        args_desc = ""
        if info["args"]:
            args_desc = " | args: " + ", ".join(
                f"{k} ({v})" for k, v in info["args"].items()
            )
        lines.append(f"- {name}: {info['description']}{args_desc}")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are Jev, a personal AI assistant running on Nik's Mac.
You understand what the user wants and convert it into a tool call.

Available tools:
{catalog}

When the user gives a command, respond with ONLY a JSON object:
{{"tool": "<tool_name>", "args": {{<arg_name>: <value>}}, "reply": "<short natural reply>"}}

Rules:
- Pick exactly ONE tool per response.
- If the user is just chatting (no command), respond with:
  {{"tool": null, "args": {{}}, "reply": "<your conversational reply>"}}
- For app names, use the exact macOS app name (e.g. "Spotify", "Safari", "Terminal").
- For volume/brightness, pick a reasonable level based on context (0-100).
- The "reply" field is what you'd say back to Nik — keep it casual, short, with personality.
- You are Jev. Nik is a final-year CS student. Keep it bro-like but not cringe.
"""


def extract_intent(user_text: str) -> dict:
    """
    Send user text to DeepSeek, get back structured intent.

    Returns: {"tool": str|None, "args": dict, "reply": str}
    """
    prompt = SYSTEM_PROMPT.format(catalog=build_tool_catalog())

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.1,  # low temp = more deterministic tool selection
        max_tokens=200,
    )

    raw = response.choices[0].message.content
    if raw is None:
        raw = ""
    raw = raw.strip()

    # DeepSeek sometimes wraps JSON in ```json blocks
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        # If JSON fails, treat the whole thing as a conversational reply
        intent = {"tool": None, "args": {}, "reply": raw}

    return intent


def think(user_text: str, confirm_threshold: float = 0.7) -> str:
    """
    Full two-model pipeline:
    1. DeepSeek extracts intent (tool + args + reply)
    2. Jev validates: is this the right tool? (confidence gate)
    3. Execute if Jev agrees, or ask user to confirm
    4. Return DeepSeek's natural reply
    """
    print(f"\n  You: {user_text}")

    # Step 1: DeepSeek understands
    print("  → DeepSeek understanding...")
    intent = extract_intent(user_text)

    tool_name = intent.get("tool")
    args = intent.get("args", {})
    reply = intent.get("reply", "")

    print(f"  ← DeepSeek: tool={tool_name}, args={args}")

    # If DeepSeek says no tool (conversation), just return the reply
    if not tool_name:
        print("  → (conversation, no tool needed)")
        return f"🧠 {reply}"

    # Step 2: Jev validates the tool choice
    print("  → Jev validating...")
    from jev_dispatch import validate_with_jev
    jev_result = validate_with_jev(user_text, tool_name)

    jev_confidence = jev_result.get("confidence", 0)
    jev_tool = jev_result.get("tool", "")
    actionable = jev_result.get("actionable", 0)

    print(f"  ← Jev: tool={jev_tool} (confidence={jev_confidence:.2f}), actionable={actionable:.2f}")

    # Gate 1: Jev thinks it's not actionable
    if actionable < 0.5:
        return f"🤔 {reply} (but Jev wasn't sure this was a command)"

    # Gate 2: Jev disagrees with the tool (low confidence)
    if jev_confidence < confirm_threshold:
        return f"⚠️ Not sure about this — DeepSeek said {tool_name}, Jev agreed at {jev_confidence:.2f}. Confirm?"

    # Gate 3: Jev picked a different tool than DeepSeek
    if jev_tool != tool_name and jev_confidence > 0.8:
        print(f"  ! Jev disagreed: DeepSeek said {tool_name}, Jev said {jev_tool}. Using Jev's pick.")
        tool_name = jev_tool
        tool_info = jev_tools.TOOL_REGISTRY.get(tool_name, {})
        # If Jev's tool has no args but DeepSeek provided some, clear them
        if not tool_info.get("args"):
            args = {}

    # Step 3: Execute
    tool_info = jev_tools.TOOL_REGISTRY.get(tool_name)
    if not tool_info:
        return f"❌ Tool '{tool_name}' doesn't exist"

    fn = tool_info["fn"]
    expected_args = tool_info.get("args", {})

    # Clean args — only pass what the function expects
    clean_args = {}
    for arg_name in expected_args:
        if arg_name in args:
            val = args[arg_name]
            if arg_name == "level":
                val = max(0, min(100, int(val)))
            clean_args[arg_name] = val

    print(f"  → Executing: {tool_name}({clean_args})")
    try:
        if clean_args:
            result = fn(**clean_args)
        else:
            result = fn()
        print(f"  ← Result: {result}")
        return f"✅ {reply}"
    except Exception as e:
        return f"❌ {reply} — but execution failed: {e}"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        print(think(user_input))
    else:
        print("Jev v0.2 — two-model (DeepSeek + Jev). Type 'quit' to exit.\n")
        while True:
            user_input = input("jev> ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue
            print(think(user_input))
        sys.exit(0)
