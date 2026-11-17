"""The console: log lines, the status line and the number formats they use."""
import re
import shutil
import sys
import threading
import time


# control characters in a log line (a request's path or fields reach the console): shown escaped, but for tab and newline
LOG_CONTROL = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f]")


class Console:
    """stderr: log lines, and on a terminal one status line under them that is rewritten in place while a request runs
    (a carriage return and spaces, no escape sequences: works in any console). A log line erases the status line and it
    is drawn again below it. Off a terminal (a log file, a pipe) only the log lines are written."""

    def __init__(self):
        self.lock = threading.Lock()
        self.text = ""  # the status line ("": none)
        self.shown = 0  # columns it occupies on the screen

    @staticmethod
    def tty():
        try:
            return sys.stderr.isatty()
        except (AttributeError, ValueError):  # (no stderr, a closed one)
            return False

    def _erase(self, s):
        if self.shown:
            s.write("\r" + " " * self.shown + "\r")
            self.shown = 0

    def _draw(self, s):
        if self.text:
            # one row less than the width: a line that fills the row would wrap, and \r only returns to the last row
            t = self.text[:max(shutil.get_terminal_size((120, 24)).columns - 1, 20)]
            s.write("\r" + t + " " * max(self.shown - len(t), 0))
            self.shown = max(self.shown, len(t))

    def line(self, text):
        with self.lock:
            s = sys.stderr
            if s is None:
                return
            live = self.tty()
            if live:
                self._erase(s)
            s.write(LOG_CONTROL.sub(lambda m: f"\\x{ord(m.group()):02x}", text) + "\n")
            if live:
                self._draw(s)
            s.flush()

    def status(self, text):
        """sets the status line ("": removes it); nothing off a terminal"""
        with self.lock:
            s = sys.stderr
            if s is None or text == self.text or not self.tty():
                return
            self.text = text
            if not text:
                self._erase(s)
            else:
                self._draw(s)
            s.flush()


CON = Console()


def log(*a):
    CON.line(" ".join(["[mk-server]", time.strftime("%H:%M:%S"), *map(str, a)]))


def secs(s):
    """a duration for the console: 0.42 s, 7.3 s, 2 min 05 s"""
    if s < 10:
        return f"{s:.2f} s"
    if s < 60:
        return f"{s:.1f} s"
    return f"{int(s // 60)} min {int(s % 60):02d} s"


def plural(n, word):
    return f"{n:,} {word}{'' if n == 1 else 's'}"


def mib(n):
    return f"{n / 2**20:,.0f} MiB"
