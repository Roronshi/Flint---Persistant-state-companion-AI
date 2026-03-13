# interface/terminal.py — Terminal chat UI

import sys
import os
import logging
from datetime import datetime

try:
    import readline  # Arrow keys and input history (Unix only)
except ImportError:
    pass  # Windows — degrades gracefully

import config
from core.model import CompanionModel
from core.session import ConversationDB, Session
from lora.scheduler import LoRAScheduler

log = logging.getLogger(__name__)


class C:
    """ANSI escape codes."""
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    USER   = "\033[36m"   # Cyan
    BOT    = "\033[32m"   # Green
    SYS    = "\033[33m"   # Yellow
    ERR    = "\033[31m"   # Red
    TPS    = "\033[35m"   # Magenta — tokens/s readout


def print_banner(has_state: bool, lora_loaded: bool):
    name = config.BOT_NAME[:16]
    print(f"""
{C.BOLD}╭─────────────────────────────────────╮
│         RWKV Companion              │
│         {name:<20}        │
╰─────────────────────────────────────╯{C.RESET}

  State:  {"✓ Loaded" if has_state else "○ New session"}
  LoRA:   {"✓ Active" if lora_loaded else "○ Base weights"}
  Model:  {os.path.basename(config.MODEL_PATH)}

{C.DIM}  /help for commands  •  Ctrl+C to exit{C.RESET}
""")


def handle_command(
    cmd: str,
    model: CompanionModel,
    db: ConversationDB,
    session: Session,
    scheduler: LoRAScheduler,
) -> bool:
    """Process a slash command. Returns True if the program should exit."""
    parts   = cmd.strip().split()
    command = parts[0].lower()

    if command == "/help":
        print(f"""
{C.SYS}Commands:{C.RESET}
  /status           — session stats, LoRA status, performance
  /save             — save state manually
  /checkpoint <tag> — save a named backup checkpoint
  /reset            — reset state (start relationship from scratch)
  /search <q>       — full-text search conversation history
  /export [file]    — export conversation history to a text file
  /lora now         — run LoRA training immediately
  /lora status      — show LoRA scheduler status
  /quit             — exit and save
""")

    elif command == "/status":
        stats = db.stats()
        print(f"""
{C.SYS}Status:{C.RESET}
  Session ID:           {session.session_id}
  Turns this session:   {session.turn_count}
  Total sessions:       {stats['total_sessions']}
  Total messages:       {stats['total_messages']}
  LoRA runs:            {stats['lora_runs']}
  Unprocessed sessions: {stats['unprocessed_sessions']}
  Scheduler:            {scheduler.status()}
""")

    elif command == "/save":
        model.save_state()
        print(f"{C.SYS}State saved.{C.RESET}")

    elif command == "/checkpoint":
        label = (
            parts[1] if len(parts) > 1
            else datetime.now().strftime("%Y%m%d_%H%M")
        )
        path = model.checkpoint_state(label)
        print(f"{C.SYS}Checkpoint saved: {path}{C.RESET}")

    elif command == "/reset":
        confirm = input(
            f"{C.SYS}Reset state? This erases relationship memory. (yes/no): {C.RESET}"
        )
        if confirm.lower() in ("yes", "y"):
            model.reset_state()
            model.prime_system_prompt()
            print(f"{C.SYS}State reset. System prompt loaded.{C.RESET}")
        else:
            print(f"{C.SYS}Cancelled.{C.RESET}")

    elif command == "/search":
        if len(parts) < 2:
            print(f"{C.SYS}Usage: /search <term>{C.RESET}")
        else:
            query   = " ".join(parts[1:])
            results = db.search(query)
            if results:
                print(f"\n{C.SYS}Results for '{query}':{C.RESET}")
                for ts, role, content, sid in results:
                    prefix  = config.USER_NAME if role == "user" else config.BOT_NAME
                    preview = content[:120].replace("\n", " ")
                    print(
                        f"  {C.DIM}[{ts[:10]} | {sid}]{C.RESET} "
                        f"{prefix}: {preview}"
                    )
            else:
                print(f"{C.SYS}No results found.{C.RESET}")
        print()

    elif command == "/export":
        from datetime import datetime as _dt
        default_name = f"companion_export_{_dt.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filename = parts[1] if len(parts) > 1 else default_name
        try:
            rows  = db.get_recent_messages(limit=5000)
            lines = [f"RWKV Companion — Export {_dt.now().strftime('%Y-%m-%d %H:%M')}", "=" * 50, ""]
            prev_session = None
            for r in rows:
                if r[3] != prev_session:
                    if prev_session is not None:
                        lines.append("")
                    lines.append(f"── Session {r[3]} ─────────────────")
                    prev_session = r[3]
                prefix   = config.USER_NAME if r[1] == "user" else config.BOT_NAME
                ts_short = r[0][:16].replace("T", " ")
                lines.append(f"[{ts_short}] {prefix}: {r[2]}")
            with open(filename, "w", encoding="utf-8") as ef:
                ef.write("\n".join(lines) + "\n")
            print(f"{C.SYS}Exported {len(rows)} messages → {filename}{C.RESET}")
        except Exception as e:
            print(f"{C.ERR}Export failed: {e}{C.RESET}")

    elif command == "/lora":
        sub = parts[1].lower() if len(parts) > 1 else ""
        if sub == "now":
            print(f"{C.SYS}Starting LoRA training in background...{C.RESET}")
            scheduler.run_now()
        elif sub == "status":
            print(f"{C.SYS}{scheduler.status()}{C.RESET}")
        else:
            print(f"{C.SYS}Usage: /lora now | /lora status{C.RESET}")

    elif command in ("/quit", "/exit", "/q"):
        return True

    else:
        print(
            f"{C.ERR}Unknown command: {command}. "
            f"Type /help for available commands.{C.RESET}"
        )

    return False


def run_chat():
    """Main terminal chat loop."""
    db    = ConversationDB()
    model = CompanionModel()

    lora_loaded = model.load_lora()
    has_state   = model.load_state()

    scheduler = LoRAScheduler(db)
    scheduler.start()

    print_banner(has_state, lora_loaded)

    if not has_state:
        model.prime_system_prompt()

    session = Session(db)

    print(f"{C.SYS}Ready. Type your first message.{C.RESET}\n")

    try:
        while True:
            try:
                user_input = input(
                    f"{C.USER}{C.BOLD}{config.USER_NAME}:{C.RESET} "
                ).strip()
            except EOFError:
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if handle_command(user_input, model, db, session, scheduler):
                    break
                continue

            prompt = f"{config.USER_NAME}: {user_input}\n{config.BOT_NAME}:"

            print(
                f"\n{C.BOT}{C.BOLD}{config.BOT_NAME}:{C.RESET} ",
                end="", flush=True
            )

            result = model.generate(
                prompt=prompt,
                stream_callback=lambda token: print(
                    f"{C.BOT}{token}{C.RESET}", end="", flush=True
                )
            )
            print("\n")

            # Tokens/s readout after each response
            print(
                f"{C.TPS}{C.DIM}  {result.tokens} tokens · "
                f"{result.tokens_per_second:.1f} tok/s{C.RESET}\n"
            )

            session.add_turn(user_input, result.text)

            if session.turn_count % config.AUTOSAVE_TURNS == 0:
                model.save_state()
                print(f"{C.DIM}  [state autosaved]{C.RESET}")

    except KeyboardInterrupt:
        print(f"\n\n{C.SYS}Interrupted.{C.RESET}")

    finally:
        print(f"\n{C.SYS}Saving state...{C.RESET}")
        model.save_state()
        session.end()
        scheduler.stop()
        print(f"{C.SYS}Goodbye.{C.RESET}")
