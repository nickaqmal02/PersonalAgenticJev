import subprocess
import re


def volume(level: int) -> str:
    """Set system output volume 0-100."""
    level = max(0, min(100, int(level)))          # clamp — LLMs pass weird values
    subprocess.run(
        ["osascript", "-e", f"set volume output volume {level}"],
        check=True,                                # raise if AppleScript fails
    )
    return f"volume set to {level}"


def open_app(name: str) -> str:
    """Open a macOS application by name."""
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


def battery() -> str:
    """Get battery level and charging status."""
    out = subprocess.check_output(
        ["pmset", "-g", "batt"], text=True
    )
    # parse: "Now drawing from 'AC Power'" or 'Battery Power'
    # parse: "-InternalBattery-0 (id=...) 42%; charging"
    pct_match = re.search(r"(\d+)%", out)
    charging = "AC Power" in out
    if not pct_match:
        return "could not read battery"
    pct = pct_match.group(1)
    status = "charging" if charging else "on battery"
    return f"battery at {pct}%, {status}"


def brightness(level: int) -> str:
    """Set screen brightness 0-100."""
    level = max(0, min(100, int(level)))
    subprocess.run(
        ["osascript", "-e", f"tell application \"System Events\" to set brightness of display 1 to {level / 100}"],
        check=True,
    )
    return f"brightness set to {level}"


def screenshot() -> str:
    """Capture a screenshot to the Desktop."""
    path = "/Users/apple/Desktop/jev_screenshot.png"
    subprocess.run(
        ["screencapture", path], check=True
    )
    return f"screenshot saved to {path}"


def wifi_status() -> str:
    """Get current WiFi network name."""
    out = subprocess.check_output(
        ["networksetup", "-getairportnetwork", "en0"],
        text=True, stderr=subprocess.DEVNULL,
    )
    # output: "You are not associated with an AirPort network."
    #   or: "Current Wi-Fi Network: MyNetworkName"
    if "not associated" in out:
        return "not connected to wifi"
    match = re.search(r"Current Wi-Fi Network:\s+(.+)", out)
    if match:
        return f"connected to {match.group(1).strip()}"
    return f"wifi status: {out.strip()}"


def frontmost_app() -> str:
    """Get the name of the currently active application."""
    out = subprocess.check_output(
        ["osascript", "-e", 'tell application "System Events" to get name of first process whose frontmost is true'],
        text=True,
    )
    return f"frontmost app: {out.strip()}"


def lock_screen() -> str:
    """Lock the screen immediately."""
    subprocess.run(
        ["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"],
        check=True,
    )
    return "screen locked"


def list_running_apps() -> str:
    """List all currently running applications."""
    out = subprocess.check_output(
        ["osascript", "-e", 'tell application "System Events" to get name of every process whose background only is false'],
        text=True,
    )
    apps = [a.strip() for a in out.split(",") if a.strip()]
    return f"running apps: {', '.join(apps)}"


# ─── Tool registry for Jev dispatch ───
# This dict maps function names to their function + descriptions.
# Jev reads these descriptions to decide which tool to call.
TOOL_REGISTRY = {
    "volume": {
        "fn": volume,
        "description": "Set the system audio output volume (0-100). Use when user says 'turn up/down volume', 'mute', etc.",
        "args": {"level": "integer 0-100, the desired volume level"},
    },
    "open_app": {
        "fn": open_app,
        "description": "Open a macOS application by name. Use when user says 'open Spotify', 'launch Safari', etc.",
        "args": {"name": "exact macOS app name, e.g. 'Spotify', 'Safari', 'Terminal'"},
    },
    "quit_app": {
        "fn": quit_app,
        "description": "Quit a running macOS application. Use when user says 'close Spotify', 'quit Safari', etc.",
        "args": {"name": "exact macOS app name to quit"},
    },
    "battery": {
        "fn": battery,
        "description": "Check battery percentage and charging status. Use when user asks 'what's my battery', 'how much battery left'.",
        "args": {},
    },
    "brightness": {
        "fn": brightness,
        "description": "Set screen brightness (0-100). Use when user says 'make screen brighter/dimmer'.",
        "args": {"level": "integer 0-100, the desired brightness"},
    },
    "screenshot": {
        "fn": screenshot,
        "description": "Take a screenshot and save to Desktop. Use when user says 'screenshot', 'capture my screen'.",
        "args": {},
    },
    "wifi_status": {
        "fn": wifi_status,
        "description": "Check which WiFi network is connected. Use when user asks 'what wifi am I on', 'network status'.",
        "args": {},
    },
    "frontmost_app": {
        "fn": frontmost_app,
        "description": "Get the name of the currently active/focused application. Use when user asks 'what am I looking at', 'what app is open'.",
        "args": {},
    },
    "lock_screen": {
        "fn": lock_screen,
        "description": "Lock the screen immediately. Use when user says 'lock my mac', 'lock screen'.",
        "args": {},
    },
    "list_running_apps": {
        "fn": list_running_apps,
        "description": "List all currently running applications. Use when user asks 'what apps are open', 'what's running'.",
        "args": {},
    },
}
