import argparse
import threading
import time
import queue
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
try:
    from analysis.qw import QRNGStream
except ImportError:
    from qw import QRNGStream

DRBG_PARAMS = {"window": 120, "margin": 0.15, "streak": 3}
QRNG_PARAMS = {"window": 120, "margin": 0.50, "streak": 10 ** 9}

class ChaCha20RNG:
    def __init__(self, key, nonce):
        self.key = key
        self.nonce = nonce
        self.counter = 0
        self.buffer = b""
    def _rotl(self, v, n):
        return ((v << n) & 0xffffffff) | (v >> (32 - n))
    def _round(self, s, a, b, c, d):
        s[a] = (s[a] + s[b]) & 0xffffffff
        s[d] ^= s[a]
        s[d] = self._rotl(s[d], 16)
        s[c] = (s[c] + s[d]) & 0xffffffff
        s[b] ^= s[c]
        s[b] = self._rotl(s[b], 12)
        s[a] = (s[a] + s[b]) & 0xffffffff
        s[d] ^= s[a]
        s[d] = self._rotl(s[d], 8)
        s[c] = (s[c] + s[d]) & 0xffffffff
        s[b] ^= s[c]
        s[b] = self._rotl(s[b], 7)
    def _block(self):
        const = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
        key_words = [int.from_bytes(self.key[i:i + 4], "little") for i in range(0, 32, 4)]
        nonce_words = [int.from_bytes(self.nonce[i:i + 4], "little") for i in range(0, 12, 4)]
        state = const + key_words + [self.counter & 0xffffffff] + nonce_words
        work = state.copy()
        for _ in range(10):
            self._round(work, 0, 4, 8, 12)
            self._round(work, 1, 5, 9, 13)
            self._round(work, 2, 6, 10, 14)
            self._round(work, 3, 7, 11, 15)
            self._round(work, 0, 5, 10, 15)
            self._round(work, 1, 6, 11, 12)
            self._round(work, 2, 7, 8, 13)
            self._round(work, 3, 4, 9, 14)
        out = bytearray()
        for i in range(16):
            out.extend(((work[i] + state[i]) & 0xffffffff).to_bytes(4, "little"))
        self.counter = (self.counter + 1) & 0xffffffff
        return bytes(out)
    def next_bytes(self, n):
        out = bytearray()
        while len(out) < n:
            if not self.buffer:
                self.buffer = self._block()
            take = min(n - len(out), len(self.buffer))
            out.extend(self.buffer[:take])
            self.buffer = self.buffer[take:]
        return bytes(out)
    def next_u64(self):
        return int.from_bytes(self.next_bytes(8), "little")

class QRNGAdapter:
    def __init__(self):
        self.stream = QRNGStream()
        self.buffer = bytearray()
    def next_bytes(self, n):
        while len(self.buffer) < n:
            try:
                chunk = self.stream.next_bytes(256)
            except Exception:
                chunk = os.urandom(256)
            if not chunk:
                chunk = os.urandom(256)
            self.buffer.extend(chunk)
        out = self.buffer[:n]
        del self.buffer[:n]
        return bytes(out)
    def next_u64(self):
        return int.from_bytes(self.next_bytes(8), "little")
    def source_name(self):
        return self.stream.source_name()

class OnlineCensor:
    def __init__(self, paths):
        self.paths = paths
        self.freq = {}
    def predict(self, ctx):
        table = self.freq.get(ctx)
        if not table:
            return None
        return max(table.items(), key=lambda item: item[1])[0]
    def update(self, ctx, path):
        table = self.freq.get(ctx)
        if not table:
            table = {}
            self.freq[ctx] = table
        table[path] = table.get(path, 0) + 1

def stream_loop(out_q, rng, paths, jitter_ns, stop_event, delay, compromised=False):
    if stop_event is None:
        return
    last_path = -1
    while not stop_event.is_set():
        try:
            r0 = rng.next_u64()
            r1 = rng.next_u64()
        except Exception:
            r0 = int.from_bytes(os.urandom(8), "little")
            r1 = int.from_bytes(os.urandom(8), "little")
        jitter = r0 % jitter_ns if jitter_ns > 0 else 0
        if compromised and last_path >= 0 and (r1 & 0xff) < 220:
            path = (last_path + 1) % paths
        else:
            path = r1 % paths
        out_q.put((last_path, path, jitter))
        last_path = path
        time.sleep(delay)

class ModeRunner:
    def __init__(self, label, params, rng_factory, axes):
        self.label = label
        self.params = params
        self.rng_factory = rng_factory
        self.path_ax, self.success_ax = axes
        self.paths = 3
        self.jitter_ns = 100000
        self.interval = 0.03
        self.display_len = 200
        self.true_series = [-1] * self.display_len
        self.pred_series = [-1] * self.display_len
        self.success_history = []
        self.window_scores = []
        self.queue = None
        self.stop_event = None
        self.thread = None
        self.censor = OnlineCensor(self.paths)
        self.running = False
        self.finished = False
        self._streak = 0
        self.rng = None
        self.status = self.path_ax.text(0.02, 0.95, "", transform=self.path_ax.transAxes, color="#ff7575", fontsize=12)
        self.title = self.path_ax.set_title(f"{self.label} run")
        self.line_true, = self.path_ax.plot(self.true_series, color="#4ce08f", linewidth=1.8, label="true path")
        self.line_pred, = self.path_ax.plot(self.pred_series, color="#ffac40", linewidth=1.4, label="prediction")
        self.path_ax.set_ylim(-0.5, self.paths - 0.5)
        self.path_ax.set_xlim(0, self.display_len)
        self.path_ax.set_ylabel("path")
        self.path_ax.legend(loc="upper right")
        self.line_success, = self.success_ax.plot([], [], color="#5fb3ff", linewidth=1.6)
        self.baseline_line = self.success_ax.axhline(1 / self.paths, color="#cccccc", linewidth=0.8, linestyle="--")
        self.success_ax.set_ylim(0, 1)
        self.success_ax.set_xlim(0, 100)
        self.success_ax.set_ylabel("success")
        self.success_ax.set_xlabel("window index")
    def start(self, event=None):
        if self.running:
            return
        if self.finished:
            self.finished = False
        self.rng = self.rng_factory()
        self.queue = queue.Queue()
        stop_evt = threading.Event()
        self.stop_event = stop_evt
        compromised = self.label == "DRBG"
        self.thread = threading.Thread(target=stream_loop, args=(self.queue, self.rng, self.paths, self.jitter_ns, stop_evt, self.interval, compromised), daemon=True)
        self.thread.start()
        self.running = True
        self.status.set_text("running")
        self.success_history = []
        self.window_scores = []
        self.censor = OnlineCensor(self.paths)
        self.true_series = [-1] * self.display_len
        self.pred_series = [-1] * self.display_len
        self.finished = False
        self._streak = 0
    def stop(self, event=None):
        if self.stop_event:
            self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)
            self.thread = None
        self.running = False
        self._streak = 0
        if not self.finished:
            self.status.set_text("")
    def update(self):
        updated = False
        if self.queue:
            while not self.queue.empty():
                prev_path, path, jitter = self.queue.get()
                ctx = (prev_path, jitter // 2000)
                pred = self.censor.predict(ctx)
                self.censor.update(ctx, path)
                self.true_series.pop(0)
                self.true_series.append(path)
                if pred is None:
                    self.pred_series.pop(0)
                    self.pred_series.append(-1)
                    self.success_history.append(0.0)
                else:
                    self.pred_series.pop(0)
                    self.pred_series.append(pred)
                    self.success_history.append(1.0 if pred == path else 0.0)
                if len(self.success_history) > self.params["window"] * 20:
                    self.success_history = self.success_history[-self.params["window"] * 20:]
                if len(self.success_history) >= self.params["window"]:
                    rate = float(np.mean(self.success_history[-self.params["window"]:]))
                    self.window_scores.append(rate)
                    if len(self.window_scores) > 400:
                        self.window_scores = self.window_scores[-400:]
                    if self.params["streak"] < 10 ** 8:
                        if rate > (1 / self.paths) + self.params["margin"]:
                            streak = getattr(self, "_streak", 0) + 1
                        else:
                            streak = 0
                        self._streak = streak
                        if streak >= self.params["streak"] and not self.finished:
                            self.finished = True
                            self.running = False
                            self.status.set_text("censor matched")
                            if self.stop_event:
                                self.stop_event.set()
                            if self.thread:
                                self.thread.join(timeout=1)
                                self.thread = None
                    else:
                        self._streak = 0
                updated = True
        if updated:
            x = np.arange(len(self.true_series))
            self.line_true.set_data(x, self.true_series)
            self.line_pred.set_data(x, self.pred_series)
            if self.window_scores:
                xs = np.arange(len(self.window_scores))
                self.line_success.set_data(xs, self.window_scores)
                self.success_ax.set_xlim(max(0, len(self.window_scores) - 120), len(self.window_scores))
            self.baseline_line.set_ydata([1 / self.paths, 1 / self.paths])
        return updated

class Dashboard:
    def __init__(self, args):
        if args.qrng_url:
            os.environ["QRNG_URL"] = args.qrng_url
        if args.qrng_key:
            os.environ["QRNG_KEY"] = args.qrng_key
        self.fig = plt.figure(figsize=(11, 6))
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1])
        ax_drbg_path = self.fig.add_subplot(gs[0, 0])
        ax_drbg_success = self.fig.add_subplot(gs[1, 0])
        ax_qrng_path = self.fig.add_subplot(gs[0, 1])
        ax_qrng_success = self.fig.add_subplot(gs[1, 1])
        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
        nonce = bytes.fromhex("000000000000004a00000000")
        drbg_rng = ChaCha20RNG(key, nonce)
        self.drbg_runner = ModeRunner("DRBG", DRBG_PARAMS, lambda: drbg_rng, (ax_drbg_path, ax_drbg_success))
        self.qrng_runner = ModeRunner("QRNG", QRNG_PARAMS, lambda: QRNGAdapter(), (ax_qrng_path, ax_qrng_success))
        ax_drbg_success.set_title("censor success (windowed)")
        ax_qrng_success.set_title("censor success (windowed)")
        self._build_buttons()
        self.anim = FuncAnimation(self.fig, self._update, interval=int(1000 / args.rate_hz))
        self.drbg_runner.start()
        self.qrng_runner.start()
    def _build_buttons(self):
        ax_start_drbg = self.fig.add_axes([0.18, 0.91, 0.08, 0.05])
        ax_stop_drbg = self.fig.add_axes([0.28, 0.91, 0.08, 0.05])
        ax_start_qrng = self.fig.add_axes([0.62, 0.91, 0.08, 0.05])
        ax_stop_qrng = self.fig.add_axes([0.72, 0.91, 0.08, 0.05])
        self.btn_start_drbg = Button(ax_start_drbg, "start DRBG")
        self.btn_stop_drbg = Button(ax_stop_drbg, "stop DRBG")
        self.btn_start_qrng = Button(ax_start_qrng, "start QRNG")
        self.btn_stop_qrng = Button(ax_stop_qrng, "stop QRNG")
        self.btn_start_drbg.on_clicked(self.drbg_runner.start)
        self.btn_stop_drbg.on_clicked(self.drbg_runner.stop)
        self.btn_start_qrng.on_clicked(self.qrng_runner.start)
        self.btn_stop_qrng.on_clicked(self.qrng_runner.stop)
    def _update(self, frame):
        updated1 = self.drbg_runner.update()
        updated2 = self.qrng_runner.update()
        drbg_rate = self.drbg_runner.window_scores[-1] if self.drbg_runner.window_scores else 0.0
        qrng_rate = self.qrng_runner.window_scores[-1] if self.qrng_runner.window_scores else 0.0
        self.fig.suptitle(f"DRBG window success {drbg_rate * 100:4.1f}%   QRNG window success {qrng_rate * 100:4.1f}%", color="#202020")
        return updated1 or updated2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrng_url", default="")
    parser.add_argument("--qrng_key", default="")
    parser.add_argument("--rate_hz", type=float, default=25.0)
    return parser.parse_args()

def main():
    args = parse_args()
    dash = Dashboard(args)
    plt.show()
    dash.drbg_runner.stop()
    dash.qrng_runner.stop()

if __name__ == "__main__":
    main()
