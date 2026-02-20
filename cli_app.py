"""
Experimental Textual TUI - Chat interface for the AI Wellness Advisor.
Run this alongside main.py for a chat interface with the Ollama model.
It reads the latest report files and lets you have a conversation.
"""

import os
import glob
import time
import ollama
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, RichLog
from textual import work

OLLAMA_MODEL = "qwen3:4b"

SYSTEM_PROMPT = (
    "You are a caring and supportive academic wellness advisor embedded in an "
    "Ethical AI Surveillance System that monitors student burnout via facial expressions. "
    "You have access to the student's latest emotion report. Be warm, supportive, and concise. "
    "If they seem stressed or burnt out, gently suggest a break or self-care. "
    "If they seem fine, encourage them. Keep responses to 3-5 sentences unless asked for more. "
    "Do NOT use emojis in your responses."
)


def get_latest_report():
    """Find and read the most recent report file."""
    reports = glob.glob("report_*.txt")
    if not reports:
        return None
    latest = max(reports, key=os.path.getmtime)
    with open(latest, 'r', encoding='utf-8') as f:
        return f.read()


class ChatApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #status-bar {
        height: 3;
        background: $surface;
        color: $text;
        padding: 0 1;
    }
    #chat-log {
        height: 1fr;
        border: solid $primary;
        padding: 0 1;
    }
    #input-box {
        dock: bottom;
        height: 3;
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit_app", "Quit"),
        ("ctrl+r", "load_report", "Load Report"),
    ]

    def __init__(self):
        super().__init__()
        self.chat_history = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(
            "[bold cyan]AI Wellness Advisor[/] - Chat Interface  |  "
            "Ctrl+R: Load latest report  |  Ctrl+Q: Quit",
            id="status-bar"
        )
        yield RichLog(highlight=True, markup=True, wrap=True, id="chat-log")
        yield Input(placeholder="Chat with your AI wellness advisor... (Enter to send)", id="input-box")
        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        log.write("[bold cyan]Ethical AI Surveillance System[/]")
        log.write("[dim]Smart Academic Burnout Predictor - Team Project Protocol[/]")
        log.write("")
        log.write("[bold green]AI Advisor:[/] Hey! I'm your wellness advisor.")
        log.write("Run [bold]main.py[/] in another terminal for the camera.")
        log.write("Press [bold]Ctrl+R[/] to load the latest report, or just chat with me.")
        log.write("")

    def action_quit_app(self) -> None:
        self.exit()

    def action_load_report(self) -> None:
        log = self.query_one("#chat-log", RichLog)
        report = get_latest_report()
        if report:
            log.write("[bold yellow]Loaded latest report:[/]")
            log.write(f"[dim]{report}[/]")
            log.write("[bold yellow]Asking AI advisor for feedback...[/]")
            self.send_chat(f"[AUTO REPORT]\n{report}\n\nPlease give supportive feedback on this report.")
        else:
            log.write("[bold red]No report files found. Run main.py and press R or S to generate one.[/]")

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        msg = event.value.strip()
        if not msg:
            return
        event.input.value = ""
        log = self.query_one("#chat-log", RichLog)
        log.write(f"[bold white]You:[/] {msg}")
        self.send_chat(msg)

    @work(thread=True)
    def send_chat(self, msg: str) -> None:
        log = self.query_one("#chat-log", RichLog)

        # Attach latest report as context if available
        report = get_latest_report()
        if report and not msg.startswith("[AUTO REPORT]"):
            content = f"{msg}\n\n[Current session data for context]:\n{report}"
        else:
            content = msg

        self.chat_history.append({"role": "user", "content": content})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.chat_history[-20:]

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                think=True,
            )
            thinking = response.message.thinking
            answer = response.message.content
            self.chat_history.append({"role": "assistant", "content": answer})
            if thinking:
                log.write(f"[dim italic]Thinking: {thinking}[/]")
            log.write(f"[bold green]AI Advisor:[/] {answer}")
            log.write("")
        except Exception as e:
            log.write(f"[bold red]Ollama error: {e}[/]")
            log.write("")


if __name__ == "__main__":
    app = ChatApp()
    app.run()
