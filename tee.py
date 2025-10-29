# tee.py A simple package to recreate the tee command from unix
import os
import sys
import atexit
import traceback

class Tee:
    """Duplicates output to multiple streams (like Unix 'tee')."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# Internal globals to keep track of redirected streams
_originalStdout = None
_originalStderr = None
_logfile = None
_isActive = False


def startTee(outputDir='./test', baseName='output'):
    """Start duplicating stdout and stderr to a new numbered text file in the test folder."""
    global _originalStdout, _originalStderr, _logfile, _isActive

    if _isActive:
        print("[Tee already active]")
        return None

    # Ensure directory exists
    os.makedirs(outputDir, exist_ok=True)

    # Find next available filename like output001.txt
    i = 1
    while True:
        filename = f"{baseName}{i:03}.txt"
        filepath = os.path.join(outputDir, filename)
        if not os.path.exists(filepath):
            break
        i += 1

    _logfile = open(filepath, 'w', buffering=1)  # line-buffered
    _originalStdout = sys.stdout
    _originalStderr = sys.stderr

    # Redirect output
    sys.stdout = Tee(_originalStdout, _logfile)
    sys.stderr = Tee(_originalStderr, _logfile)
    _isActive = True

    # Automatically clean up on exit, even on crash
    atexit.register(endTee, silent=True)

    print(f"[Tee started] Output also logging to: {filename}")
    return filename


def endTee(silent=False):
    """Restore stdout/stderr and close the log file safely."""
    global _originalStdout, _originalStderr, _logfile, _isActive

    if not _isActive:
        return

    # Restore streams
    if _originalStdout:
        sys.stdout = _originalStdout
    if _originalStderr:
        sys.stderr = _originalStderr

    if _logfile:
        if not silent:
            print(f"[Tee ended] Log saved to: {_logfile.name}")
        _logfile.close()

    _originalStdout = None
    _originalStderr = None
    _logfile = None
    _isActive = False


# Optional: automatically log unhandled exceptions to file before exiting
def _log_uncaught_exceptions(exctype, value, tb):
    if _logfile:
        traceback.print_exception(exctype, value, tb, file=_logfile)
        _logfile.flush()
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = _log_uncaught_exceptions
