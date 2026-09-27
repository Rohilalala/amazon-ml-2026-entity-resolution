"""RSS watchdog: aborts the process above CAP_GB (checked every second)."""
import os, subprocess, threading, time


def start(cap_gb=5.0):
    def run():
        while True:
            g = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout) / 2**20
            if g > cap_gb:
                print(f"WATCHDOG: rss {g:.2f} GB > {cap_gb} GB, abort", flush=True)
                os._exit(99)
            time.sleep(1)
    threading.Thread(target=run, daemon=True).start()
