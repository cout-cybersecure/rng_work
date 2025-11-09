import sys
import time
import math
import random
from collections import deque
import numpy as np
try:
    from PySide6.QtCore import Qt, QTimer, QPointF, QLineF
    from PySide6.QtGui import QPainter, QColor, QPen, QFont, QPalette
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QButtonGroup, QFrame, QSizePolicy
    ALIGN_CENTER = Qt.AlignCenter
    ALIGN_LEFT = Qt.AlignLeft
    ALIGN_VCENTER = Qt.AlignVCenter
    ALIGN_RIGHT = Qt.AlignRight
    ALIGN_TOP = Qt.AlignTop
    def align(flags):
        return flags
except ImportError:
    from PyQt6.QtCore import Qt, QTimer, QPointF, QLineF
    from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPalette
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QButtonGroup, QFrame, QSizePolicy
    ALIGN_CENTER = Qt.AlignmentFlag.AlignCenter
    ALIGN_LEFT = Qt.AlignmentFlag.AlignLeft
    ALIGN_VCENTER = Qt.AlignmentFlag.AlignVCenter
    ALIGN_RIGHT = Qt.AlignmentFlag.AlignRight
    ALIGN_TOP = Qt.AlignmentFlag.AlignTop
    def align(flags):
        return flags
try:
    from analysis.qrng_wrapper import QRNGStream
except ImportError:
    from qrng_wrapper import QRNGStream

class QRNGAdapter:
    def __init__(self):
        self.stream = QRNGStream()
        self.error = None
    def next_bytes(self, n):
        data = self.stream.next_bytes(n)
        self.error = getattr(self.stream, "last_error", None)
        return data
    def next_u64(self):
        return int.from_bytes(self.next_bytes(8), "little")
    def source_name(self):
        return self.stream.source_name()

class ChaCha20RNG:
    def __init__(self, key, nonce):
        self.key = key
        self.nonce = nonce
        self.counter = 0
        self.buffer = b""
        self.bytes_since_reseed = 0
    def _rotl32(self, v, shift):
        return ((v << shift) & 0xffffffff) | (v >> (32 - shift))
    def _quarter_round(self, state, a, b, c, d):
        state[a] = (state[a] + state[b]) & 0xffffffff
        state[d] ^= state[a]
        state[d] = self._rotl32(state[d], 16)
        state[c] = (state[c] + state[d]) & 0xffffffff
        state[b] ^= state[c]
        state[b] = self._rotl32(state[b], 12)
        state[a] = (state[a] + state[b]) & 0xffffffff
        state[d] ^= state[a]
        state[d] = self._rotl32(state[d], 8)
        state[c] = (state[c] + state[d]) & 0xffffffff
        state[b] ^= state[c]
        state[b] = self._rotl32(state[b], 7)
    def _serialize(self, state):
        output = bytearray()
        for value in state:
            output.extend(value.to_bytes(4, "little"))
        return bytes(output)
    def _chacha_block(self):
        constants = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
        key_words = [int.from_bytes(self.key[i:i + 4], "little") for i in range(0, 32, 4)]
        counter_word = self.counter & 0xffffffff
        nonce_words = [int.from_bytes(self.nonce[i:i + 4], "little") for i in range(0, 12, 4)]
        state = constants + key_words + [counter_word] + nonce_words
        working = state.copy()
        for _ in range(10):
            self._quarter_round(working, 0, 4, 8, 12)
            self._quarter_round(working, 1, 5, 9, 13)
            self._quarter_round(working, 2, 6, 10, 14)
            self._quarter_round(working, 3, 7, 11, 15)
            self._quarter_round(working, 0, 5, 10, 15)
            self._quarter_round(working, 1, 6, 11, 12)
            self._quarter_round(working, 2, 7, 8, 13)
            self._quarter_round(working, 3, 4, 9, 14)
        for i in range(16):
            working[i] = (working[i] + state[i]) & 0xffffffff
        block = self._serialize(working)
        self.counter = (self.counter + 1) & 0xffffffff
        return block
    def next_bytes(self, n):
        output = bytearray()
        while len(output) < n:
            if not self.buffer:
                self.buffer = self._chacha_block()
            take = min(n - len(output), len(self.buffer))
            output.extend(self.buffer[:take])
            self.buffer = self.buffer[take:]
        self.bytes_since_reseed += len(output)
        if self.bytes_since_reseed >= 4096:
            self.counter = 0
            self.buffer = b""
            self.bytes_since_reseed = 0
        return bytes(output)

class StreamState:
    def __init__(self, name):
        self.name = name
        self.samples = deque()
        self.counts = np.zeros(256, dtype=np.int32)
        self.waveform = np.zeros(256, dtype=np.float32)
        self.lag_values = np.zeros(255, dtype=np.float32)
        self.metrics = {"min": 0, "max": 0, "mean": 0.0, "entropy": 0.0, "hex": ""}
        self.total_bytes = 0
        self.byte_rate = 0.0
        self.last_time = None
        self.source_note = ""
        self.structure_counter = 0
        self.structure_flag = False
    def add_bytes(self, data):
        now = time.monotonic()
        if self.last_time is None:
            self.last_time = now
        for value in data:
            if len(self.samples) >= 256:
                self.samples.popleft()
            self.samples.append(value)
        if len(self.samples) == 0:
            return
        arr = np.fromiter(self.samples, dtype=np.uint8)
        self.counts = np.bincount(arr, minlength=256)
        self.waveform = arr.astype(np.float32)
        if len(arr) > 1:
            self.lag_values = np.abs(np.diff(self.waveform))
        else:
            self.lag_values = np.zeros(1, dtype=np.float32)
        self.metrics["min"] = int(self.waveform.min())
        self.metrics["max"] = int(self.waveform.max())
        self.metrics["mean"] = float(self.waveform.mean())
        max_count = self.counts.max() if len(arr) > 0 else 1
        entropy = -math.log2(max_count / len(arr)) if max_count > 0 else 0.0
        self.metrics["entropy"] = entropy
        recent = bytes(arr[-16:]) if len(arr) >= 16 else bytes(arr)
        self.metrics["hex"] = recent.hex().upper()
        self.total_bytes += len(data)
        dt = now - self.last_time
        if dt > 0:
            instant = len(data) / dt
            self.byte_rate = 0.7 * self.byte_rate + 0.3 * instant
        self.last_time = now
        if entropy < 1.2:
            self.structure_counter += 1
            if self.structure_counter >= 3:
                self.structure_flag = True
        else:
            self.structure_counter = 0
            self.structure_flag = False
    def entropy(self):
        return self.metrics.get("entropy", 0.0)

class StreamPanel(QWidget):
    def __init__(self, state, panel_type):
        super().__init__()
        self.state = state
        self.panel_type = panel_type
        self.setMinimumHeight(360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor(16, 18, 26))
        if len(self.state.waveform) == 0:
            return
        width = rect.width()
        height = rect.height()
        margin = 16
        section1 = int(height * 0.4)
        section2 = int(height * 0.25)
        section3 = int(height * 0.2)
        top_y = margin
        waveform_rect = rect.adjusted(margin, top_y, -margin, -(height - (section1 + top_y)))
        painter.setPen(QPen(QColor(60, 70, 90), 1))
        for i in range(5):
            y = waveform_rect.top() + i * waveform_rect.height() / 4
            painter.drawLine(QLineF(waveform_rect.left(), y, waveform_rect.right(), y))
        for i in range(5):
            x = waveform_rect.left() + i * waveform_rect.width() / 4
            painter.drawLine(QLineF(x, waveform_rect.top(), x, waveform_rect.bottom()))
        values = self.state.waveform
        if len(values) > 1:
            step_x = waveform_rect.width() / (len(values) - 1)
            painter.setPen(QPen(QColor(0, 200, 180), 2))
            prev_point = QPointF(waveform_rect.left(), waveform_rect.bottom() - (values[0] / 255.0) * waveform_rect.height())
            for i in range(1, len(values)):
                x = waveform_rect.left() + i * step_x
                y = waveform_rect.bottom() - (values[i] / 255.0) * waveform_rect.height()
                painter.drawLine(QLineF(prev_point.x(), prev_point.y(), x, y))
                prev_point = QPointF(x, y)
        hist_top = waveform_rect.bottom() + margin
        hist_rect = rect.adjusted(margin, hist_top, -margin, -(height - (hist_top + section2)))
        counts = self.state.counts
        max_count = counts.max() if counts.max() > 0 else 1
        bar_width = max(hist_rect.width() / 256.0, 1.0)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 160, 80))
        for i in range(256):
            h = (counts[i] / max_count) * hist_rect.height()
            x = hist_rect.left() + i * bar_width
            y = hist_rect.bottom() - h
            painter.drawRect(int(x), int(y), int(bar_width) + 1, int(h))
        lag_top = hist_rect.bottom() + margin
        lag_rect = rect.adjusted(margin, lag_top, -margin, -(height - (lag_top + section3)))
        lag_values = self.state.lag_values if len(self.state.lag_values) > 0 else np.zeros(1)
        lag_max = lag_values.max() if lag_values.max() > 0 else 1
        painter.setPen(QPen(QColor(120, 200, 255), 2))
        if len(lag_values) > 1:
            step_x = lag_rect.width() / (len(lag_values) - 1)
            prev_point = QPointF(lag_rect.left(), lag_rect.bottom() - (lag_values[0] / lag_max) * lag_rect.height())
            for i in range(1, len(lag_values)):
                x = lag_rect.left() + i * step_x
                y = lag_rect.bottom() - (lag_values[i] / lag_max) * lag_rect.height()
                painter.drawLine(QLineF(prev_point.x(), prev_point.y(), x, y))
                prev_point = QPointF(x, y)
        text_top = lag_rect.bottom() + margin
        text_rect = rect.adjusted(margin, text_top, -margin, -margin)
        painter.setPen(QPen(QColor(200, 210, 230)))
        painter.setFont(QFont("Inter", 12))
        metrics = self.state.metrics
        summary = f"min {metrics['min']:3d}   max {metrics['max']:3d}   mean {metrics['mean']:.2f}   entropy {metrics['entropy']:.2f}   rate {self.state.byte_rate:.1f} B/s"
        painter.drawText(text_rect.left(), text_rect.top() + 18, summary)
        painter.drawText(text_rect.left(), text_rect.top() + 38, f"source {self.state.source_note}")
        painter.drawText(text_rect.left(), text_rect.top() + 58, f"last16 {metrics['hex']}")
        painter.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        if self.panel_type == "drbg" and self.state.structure_flag:
            painter.setPen(QPen(QColor(255, 90, 90)))
            painter.drawText(rect.adjusted(0, 10, -16, 0), align(ALIGN_RIGHT | ALIGN_TOP), "structure detected")
        if self.panel_type == "qrng":
            painter.setPen(QPen(QColor(110, 230, 150)))
            painter.drawText(rect.adjusted(0, 10, -16, 0), align(ALIGN_RIGHT | ALIGN_TOP), "live")

class AdversaryModel:
    def __init__(self, margin, window, enable_flag):
        self.margin = margin
        self.window = window
        self.enable_flag = enable_flag
        self.baseline = 1.0 / 256.0
        self.threshold = self.baseline + self.margin
        self.T1 = {}
        self.T2 = {}
        self.prev = None
        self.success_history = deque(maxlen=self.window)
        self.last_rate = 0.0
        self.consecutive = 0
        self.flagged = False
    def _predict(self):
        if self.prev is None:
            return random.getrandbits(8)
        key2 = (self.prev[0], self.prev[1])
        arr2 = self.T2.get(key2)
        if arr2 is not None:
            if arr2.sum() > 0:
                return int(arr2.argmax())
        arr1 = self.T1.get(self.prev[0])
        if arr1 is not None and arr1.sum() > 0:
            return int(arr1.argmax())
        return random.getrandbits(8)
    def update(self, value, bucket):
        guess = self._predict()
        success = 1 if guess == value else 0
        self.success_history.append(success)
        if self.prev is not None:
            key2 = (self.prev[0], self.prev[1])
                
        if self.prev is not None:
            arr2 = self.T2.setdefault((self.prev[0], self.prev[1]), np.zeros(256, dtype=np.int32))
            arr2[value] += 1
            arr1 = self.T1.setdefault(self.prev[0], np.zeros(256, dtype=np.int32))
            arr1[value] += 1
        self.prev = (value, bucket)
        if len(self.success_history) > 0:
            self.last_rate = sum(self.success_history) / len(self.success_history)
        else:
            self.last_rate = 0.0
        if len(self.success_history) == self.window and self.last_rate > self.threshold:
            self.consecutive += 1
        else:
            self.consecutive = 0
        if self.enable_flag and self.consecutive >= 3:
            self.flagged = True
        return guess

class AdversaryView(QWidget):
    def __init__(self, drbg_model, qrng_model):
        super().__init__()
        self.drbg_model = drbg_model
        self.qrng_model = qrng_model
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.fillRect(rect, QColor(20, 24, 32))
        half = rect.width() // 2
        drbg_rect = rect.adjusted(12, 12, -half - 6, -12)
        qrng_rect = rect.adjusted(half + 6, 12, -12, -12)
        self._draw_box(painter, drbg_rect, "DRBG attack", self.drbg_model, True)
        self._draw_box(painter, qrng_rect, "QRNG attack", self.qrng_model, False)
    def _draw_box(self, painter, rect, title, model, flaggable):
        painter.setPen(QPen(QColor(60, 70, 90)))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QPen(QColor(200, 210, 230)))
        painter.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(12, 8, -12, 0), align(ALIGN_LEFT | ALIGN_TOP), title)
        rate = model.last_rate if model.last_rate else 0.0
        painter.setFont(QFont("Inter", 32, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(12, 50, -12, -60), align(ALIGN_LEFT | ALIGN_TOP), f"{rate * 100:5.1f}%")
        bar_rect = rect.adjusted(12, rect.height() - 56, -12, -32)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(40, 46, 60))
        painter.drawRoundedRect(bar_rect, 6, 6)
        filled = max(0.0, min(rate / model.threshold, 1.0)) if model.threshold > 0 else 0.0
        painter.setBrush(QColor(255, 100, 100) if flaggable and model.flagged else QColor(100, 220, 160))
        painter.drawRoundedRect(bar_rect.adjusted(0, 0, int((filled - 1) * bar_rect.width()), 0), 6, 6)
        painter.setFont(QFont("Inter", 16, QFont.Weight.Medium))
        if flaggable:
            if model.flagged:
                painter.setPen(QPen(QColor(255, 80, 80)))
                status = "predictable"
            else:
                painter.setPen(QPen(QColor(160, 200, 255)))
                status = "monitoring"
        else:
            if rate <= model.threshold:
                painter.setPen(QPen(QColor(120, 220, 160)))
                status = "no leverage"
            else:
                painter.setPen(QPen(QColor(255, 120, 80)))
                status = "anomaly"
        painter.drawText(rect.adjusted(12, rect.height() - 30, -12, -12), align(ALIGN_LEFT | ALIGN_BOTTOM), status)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Randomness Studio")
        self.setMinimumSize(1200, 700)
        self.drbg_state = StreamState("DRBG")
        self.qrng_state = StreamState("QRNG")
        self.drbg_state.source_note = "ChaCha20 reseeded"
        self.qrng_adapter = QRNGAdapter()
        self.qrng_state.source_note = self.qrng_adapter.source_name()
        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
        nonce = bytes.fromhex("000000000000004a00000000")
        self.drbg_rng = ChaCha20RNG(key, nonce)
        self.adversary_drbg = AdversaryModel(0.02, 200, True)
        self.adversary_qrng = AdversaryModel(0.02, 200, False)
        self._build_ui()
        for _ in range(8):
            self._prefill()
        self._update_header()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)
    def _build_ui(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(12, 14, 20))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(230, 235, 245))
        palette.setColor(QPalette.ColorRole.Base, QColor(20, 24, 32))
        palette.setColor(QPalette.ColorRole.Button, QColor(32, 38, 48))
        self.setPalette(palette)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 12, 12, 12)
        header_layout.setSpacing(20)
        self.title_label = QLabel("Randomness Studio")
        self.title_label.setFont(QFont("Inter", 28, QFont.Weight.Bold))
        header_layout.addWidget(self.title_label, 1)
        self.info_label = QLabel()
        self.info_label.setFont(QFont("Inter", 14))
        self.info_label.setAlignment(align(ALIGN_LEFT | ALIGN_VCENTER))
        header_layout.addWidget(self.info_label, 3)
        button_frame = QFrame()
        button_layout = QHBoxLayout(button_frame)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        self.button_group = QButtonGroup(self)
        self.btn_drbg = QPushButton("DRBG only")
        self.btn_qrng = QPushButton("QRNG only")
        self.btn_both = QPushButton("Both")
        for idx, btn in enumerate([self.btn_drbg, self.btn_qrng, self.btn_both]):
            btn.setCheckable(True)
            btn.setStyleSheet("QPushButton { color: #e0e4ee; background: #1e2433; padding: 8px 16px; border-radius: 6px; } QPushButton:checked { background: #3f57ff; }")
            self.button_group.addButton(btn, idx)
            button_layout.addWidget(btn)
        self.btn_both.setChecked(True)
        try:
            self.button_group.idClicked.connect(self._change_mode)
        except AttributeError:
            self.button_group.buttonClicked[int].connect(self._change_mode)
        header_layout.addWidget(button_frame, 0, align(ALIGN_RIGHT | ALIGN_VCENTER))
        layout.addWidget(header)
        panels_frame = QFrame()
        panels_layout = QHBoxLayout(panels_frame)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(12)
        self.drbg_panel = StreamPanel(self.drbg_state, "drbg")
        self.qrng_panel = StreamPanel(self.qrng_state, "qrng")
        panels_layout.addWidget(self.drbg_panel)
        panels_layout.addWidget(self.qrng_panel)
        layout.addWidget(panels_frame, 1)
        self.adversary_view = AdversaryView(self.adversary_drbg, self.adversary_qrng)
        layout.addWidget(self.adversary_view)
        self.mode = "both"
    def _prefill(self):
        drbg_bytes = self.drbg_rng.next_bytes(32)
        self.drbg_state.add_bytes(drbg_bytes)
        for b in drbg_bytes:
            bucket = b // 16
            self.adversary_drbg.update(b, bucket)
        qrng_bytes = self.qrng_adapter.next_bytes(32)
        self.qrng_state.source_note = self.qrng_adapter.source_name()
        self.qrng_state.add_bytes(qrng_bytes)
        for b in qrng_bytes:
            bucket = b // 16
            self.adversary_qrng.update(b, bucket)
    def _tick(self):
        self._prefill()
        self.drbg_panel.update()
        self.qrng_panel.update()
        self.adversary_view.update()
        self._update_header()
    def _update_header(self):
        drbg_entropy = self.drbg_state.entropy()
        qrng_entropy = self.qrng_state.entropy()
        info = f"DRBG entropy {drbg_entropy:.2f} bits | rate {self.drbg_state.byte_rate:.1f} B/s     QRNG entropy {qrng_entropy:.2f} bits | rate {self.qrng_state.byte_rate:.1f} B/s | source {self.qrng_state.source_note}"
        self.info_label.setText(info)
    def _change_mode(self, idx):
        if idx == 0:
            self.mode = "drbg"
        elif idx == 1:
            self.mode = "qrng"
        else:
            self.mode = "both"
        self._apply_mode()
    def _apply_mode(self):
        if self.mode == "drbg":
            self.drbg_panel.show()
            self.qrng_panel.hide()
        elif self.mode == "qrng":
            self.drbg_panel.hide()
            self.qrng_panel.show()
        else:
            self.drbg_panel.show()
            self.qrng_panel.show()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
