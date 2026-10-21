# JEV — Personal Jarvis Automation Project

> Building a personal AI assistant that automates apps and tasks on my Mac.
> Decision engine: **TypeSafe's Jev** (System One model — typed decisions, not text generation).
> Chat layer: GLM (for conversation around the decisions).
> Platform: Hermes Agent (tool calling, skills, memory, cron, gateway).
> Path: `~/degree/PersonalAgentic/jev/`
> Started: Oct 2026.

---

## The Architecture (how the pieces fit)

```
You speak/type: "hey jev, turn down the volume and open Spotify"
       ↓
GLM (chat model) ── understands the sentence, plans what to do
       ↓
Jev (System One) ── outputs typed decision: volume(30), open_app("Spotify")
       │              with confidence scores per argument
       │              CAN'T hallucinate — output space is closed by construction
       ↓
jev_tools.py ── executes the functions on macOS
       ↓
Jev (memory gate) ── Noul: "was this request worth remembering?"
       ↓
jev replies: "done — volume at 30, Spotify's on 🎵"
```

**Two models, two roles:**
- **GLM** = conversation (understands you, replies in natural language)
- **Jev** = decisions (picks which tool, which args, with confidence — no prose, no hallucination)

---

## Phase Breakdown

### Phase 0 — Foundation ✅ DONE
**Goal:** Build the tools layer. Pure Python, stdlib-only, no API key needed.

| Feature | Status | Notes |
|---|---|---|
| `volume(level)` | ✅ DONE | Tested live, sets macOS volume via osascript |
| `open_app(name)` | ✅ DONE | Opens apps via `open -a` |
| `quit_app(name)` | 🔧 FIX NEEDED | Function is cut off mid-body — missing subprocess call |
| `battery()` | ❌ TODO | Parse `pmset -g batt` output with regex |
| `brightness(level)` | ❌ TODO | via osascript or MonitorControl CLI |
| `screenshot()` | ❌ TODO | `screencapture` command |
| `wifi_status()` | ❌ TODO | `networksetup -getairportnetwork` |
| `frontmost_app()` | ❌ TODO | osascript to read active window |
| `lock_screen()` | ❌ TODO | `pmset displaysleepnow` |
| `list_running_apps()` | ❌ TODO | osascript System Events |

**Exit criteria:** all functions tested live, each returns a clean string, handles bad input with try/except.

---

### Phase 1 — Two-Model Decision Layer ✅ DONE
**Goal:** Wire your tool functions to Jev so the model decides which to call.
**Requires:** `TYPESAFE_API_KEY` from console.typesafe.ai/keys + `pip install typesafe-sdk`

**What happens here:**
```
you type: "turn it down"
  ↓
Jev receives: state="turn it down"
  questions: [which function? which volume level?]
  ↓
Jev returns: {function: "volume", level: "30", confidence: 0.91}
  ↓
jev_tools.volume(30) runs
  ↓
mac volume drops to 30
```

**Steps:**
1. Get API key from console.typesafe.ai/keys
2. `pip install typesafe-sdk` + set `TYPESAFE_API_KEY` env var
3. Build `jev_dispatch.py` — reads your function signatures, creates TypeSafe questions:
   - **Choice** question: which function? (options = your function names + descriptions)
   - **Choice/Noul** per argument: which value? (options = Literal sets or yes/no)
4. Write the function spec — plain-language descriptions of what each tool does
5. Test: type "open spotify" → Jev picks `open_app("Spotify")` with confidence → runs
6. Add confidence gating: if confidence < 0.7, ask user to confirm before running

**The "aha" moment:** you type "hey jev, open my coding apps" and watch the model pick the right functions with the right args, no hardcoding.

**Why Jev and not just GLM:**
| | GLM (LLM) | Jev (System One) |
|---|---|---|
| Output | free text (can hallucinate) | typed decision (can't hallucinate) |
| Speed | slow (token by token) | 100x faster |
| Confidence | none | per-argument confidence scores |
| Use for | conversation, reasoning | tool selection, argument filling |

---

### Phase 2 — System Senses
**Goal:** Jev can *see* your Mac state before making decisions.

| Feature | What it teaches |
|---|---|
| `screenshot()` | capture screen, pass as state to Jev for visual context |
| `frontmost_app()` | Jev knows what you're working on |
| `battery()` + `wifi_status()` | Jev can reason: "battery at 12%, don't start heavy ML training" |
| `list_running_apps()` | Jev knows what's already open — won't re-launch |

**The pattern:** sense functions feed their output into Jev's `state` field, so the model has context when deciding.

---

### Phase 3 — Memory
**Goal:** Jev remembers your routines across sessions.
**Pattern:** TypeSafe's "Simple memory system" cookbook — after each Jev call, a **Noul** question asks "was this request worth remembering?" If P(yes) ≥ 0.5, save it.

| What Jev learns | How |
|---|---|
| "nik codes every monday at 8pm" | memory gate saves recurring patterns |
| "nik's coding apps are CotEditor + Alacritty" | remembers open_app combos |
| "nik's FYP demo is on Nov 5" | deadline tracking |
| "battery was 12% last time, charged now" | state diffs |

**Two memory layers:**
- **Jev memory** (TypeSafe cookbook pattern) — decides what's worth saving
- **Hermes memory** (built-in) — persists across sessions, injected into every turn

---

### Phase 4 — Proactive Automation (Cron)
**Goal:** Jev acts on his own without you asking.
**Requires:** Hermes `cronjob` tool.

| Cron job | What Jev decides |
|---|---|
| 8AM daily briefing | which apps to open, what to remind you about, what's on your calendar |
| Battery low alert | if battery < 20%, notify + suggest quitting heavy apps |
| FYP deadline countdown | how many days left, what to prep |
| "coding time" trigger | if it's monday 8pm and you usually code, open your coding apps |

**The magic:** the cron job fires, Jev reads system state, makes a typed decision, your tools execute — all without you saying anything.

---

### Phase 5 — Voice
**Goal:** Speak to Jev, he speaks back.
**Requires:** Hermes STT/TTS config (whisper + edge-tts or similar).

| Config | What it does |
|---|---|
| `stt.enabled: true` | you talk → transcribed to text → Jev decides |
| `tts.enabled: true` | Jev's response → spoken back to you |
| Wake word (stretch) | "hey jev" → starts listening |

**Flow:**
```
you speak → STT → "turn down volume"
  ↓
Jev decides → volume(30)
  ↓
TTS → "done, volume at 30"
```

---

### Phase 6 — Remote Control (Telegram Gateway)
**Goal:** Text Jev from your phone → he acts on your Mac at home.
**Requires:** Hermes Telegram channel setup.

| You text | Jev does |
|---|---|
| "start the FYP server" | opens terminal, runs Django dev server |
| "what's my battery?" | calls battery(), replies "62% charging" |
| "open spotify and play focus playlist" | open_app + spotify CLI |
| "screenshot my screen" | captures screen, sends image back to telegram |

**This is the Jarvis moment:** you're at campus, your Mac is at home, you text Jev, your Mac obeys.

---

## Progress Tracker

| Phase | Features Done | Status |
|---|---|---|
|| 0 — Foundation | 10/10 | ✅ DONE |
|| 1 — Two-Model Layer | 4/6 | ✅ DONE (DeepSeek + Jev working) |
|| 2 — System Senses | 0/4 | ⏳ PENDING |
|| 3 — Memory | 0/4 | ⏳ PENDING |
|| 4 — Proactive Cron | 0/4 | ⏳ PENDING |
|| 5 — Voice | 0/3 | ⏳ PENDING |
|| 6 — Telegram Gateway | 0/4 | ⏳ PENDING ||

---

## ⭐ Jev Integration Points (highlighted)

1. **Phase 1** — Jev becomes the decision brain (picks which tool to call + with what args)
2. **Phase 3** — Jev's Noul gate decides what's worth remembering (memory system)
3. **Phase 4** — Jev inside cron jobs decides what to do when the schedule fires
4. **Phase 6** — Jev reads your Telegram text and decides what to run on your Mac

---

## Tech Stack Summary

| Component | Technology |
|---|---|
| Tools (hands) | Python 3 stdlib + subprocess + osascript |
| Decision brain | TypeSafe Jev (System One model) |
| Chat layer | GLM via Hermes |
| Memory | TypeSafe memory gate + Hermes persistent memory |
| Scheduling | Hermes cronjob |
| Voice | Hermes STT/TTS config |
| Remote | Hermes Telegram gateway |
| Platform | Hermes Agent (desktop app) |
| Machine | M1 Pro MacBook Pro, 32GB, macOS 26.3 |

---

## Interview Talking Points (built from this project)

1. **"I built a personal agent using a System One model as a typed decision layer"** — Jev returns structured decisions, not free text, so tool selection can't hallucinate
2. **"Two-model architecture: chat model for conversation, System One for decisions"** — clean separation of concerns
3. **"Confidence-gated routing"** — Jev returns per-argument confidence; below threshold, ask user to confirm
4. **"Memory gate via Noul questions"** — the model itself decides what's worth remembering
5. **"Same agent core, multiple surfaces"** — CLI, desktop, Telegram, voice — the gateway pattern

---

*Last updated: Oct 21, 2026*
*Phase 0 + Phase 1 complete. Next: Phase 2 (System Senses)*
