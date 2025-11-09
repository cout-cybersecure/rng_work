import os
import sys
import random
from collections import deque
import numpy as np
try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QColor, QPainter, QPen, QFont, QPalette, QKeySequence, QShortcut
    from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
except ImportError:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QColor, QPainter, QPen, QFont, QPalette, QKeySequence, QShortcut
    from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
try:
    from analysis.qrng_wrapper import QRNGStream
except ImportError:
    class QRNGStream:
        def next_bytes(self, n):
            return os.urandom(n)
        def source_name(self):
            return "local stand-in"

WINDOW_MIN_W = 1200
WINDOW_MIN_H = 700
MARGIN_X = 80
HEADER_H = 60
FOOTER_H = 50
LANE_H = 36
LANE_SPACING = 60
CENSOR_W = 55
PACKET_R = 10
ANIM_FRAMES = 8
BG = QColor("#0f131a")
LANE_COLOR = QColor("#434a57")
PACKET_NORMAL = QColor("#f7c840")
PACKET_PASS = QColor("#38d47a")
PACKET_QRNG_PASS = QColor("#7ec8ff")
PACKET_BLOCKED = QColor("#f24f4f")
CENSOR_BAR = QColor("#7b818d")
CENSOR_ACTIVE = QColor("#f24f4f")
NODE_COLOR = QColor("#7e8ba0")
TEXT_COLOR = QColor("#ffffff")

class ChaCha20:
    def __init__(self, key, nonce):
        self.key = key
        self.nonce = nonce
        self.counter = 0
        self.buffer = b""
        self.generated = 0
    def _rotl32(self, v, s):
        return ((v << s) & 0xffffffff) | (v >> (32 - s))
    def _quarter(self, st, a, b, c, d):
        st[a] = (st[a] + st[b]) & 0xffffffff
        st[d] ^= st[a]
        st[d] = self._rotl32(st[d], 16)
        st[c] = (st[c] + st[d]) & 0xffffffff
        st[b] ^= st[c]
        st[b] = self._rotl32(st[b], 12)
        st[a] = (st[a] + st[b]) & 0xffffffff
        st[d] ^= st[a]
        st[d] = self._rotl32(st[d], 8)
        st[c] = (st[c] + st[d]) & 0xffffffff
        st[b] ^= st[c]
        st[b] = self._rotl32(st[b], 7)
    def _block(self):
        constants = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574]
        key_words = [int.from_bytes(self.key[i:i + 4], "little") for i in range(0, 32, 4)]
        counter_word = self.counter & 0xffffffff
        nonce_words = [int.from_bytes(self.nonce[i:i + 4], "little") for i in range(0, 12, 4)]
        state = constants + key_words + [counter_word] + nonce_words
        working = state.copy()
        for _ in range(10):
            self._quarter(working, 0, 4, 8, 12)
            self._quarter(working, 1, 5, 9, 13)
            self._quarter(working, 2, 6, 10, 14)
            self._quarter(working, 3, 7, 11, 15)
            self._quarter(working, 0, 5, 10, 15)
            self._quarter(working, 1, 6, 11, 12)
            self._quarter(working, 2, 7, 8, 13)
            self._quarter(working, 3, 4, 9, 14)
        for i in range(16):
            working[i] = (working[i] + state[i]) & 0xffffffff
        block = bytearray()
        for value in working:
            block.extend(value.to_bytes(4, "little"))
        self.counter = (self.counter + 1) & 0xffffffff
        return bytes(block)
    def next_bytes(self, n):
        out = bytearray()
        while len(out) < n:
            if not self.buffer:
                self.buffer = self._block()
            take = min(n - len(out), len(self.buffer))
            out.extend(self.buffer[:take])
            self.buffer = self.buffer[take:]
        self.generated += len(out)
        if self.generated >= 200 * 16:
            self.counter = 0
            self.buffer = b""
            self.generated = 0
        return bytes(out)

class DRBGSource:
    def __init__(self):
        key = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
        nonce = bytes.fromhex("000000000000004a00000000")
        self.key = key
        self.nonce = nonce
        self.chacha = ChaCha20(key, nonce)
        self.steps = 0
        self.prev_path = None
        self.transition_map = [[1, 0, 2], [2, 2, 1], [0, 1, 0]]
    def next_path(self):
        if self.prev_path is None:
            seed = int.from_bytes(self.chacha.next_bytes(8), "big") % 3
            self.prev_path = seed
            result = seed
        else:
            idx = int.from_bytes(self.chacha.next_bytes(8), "big") % 3
            result = self.transition_map[self.prev_path][idx]
            self.prev_path = result
        self.steps += 1
        if self.steps >= 200:
            self.chacha = ChaCha20(self.key, self.nonce)
            self.steps = 0
            self.prev_path = None
        return result

class QRNGSource:
    def __init__(self):
        try:
            self.stream = QRNGStream()
            self.label = self.stream.source_name() if hasattr(self.stream, "source_name") else "QRNG"
        except Exception:
            self.stream = None
            self.label = "local stand-in"
    def next_path(self):
        if self.stream is not None:
            try:
                raw = self.stream.next_bytes(8)
            except Exception:
                raw = os.urandom(8)
        else:
            raw = os.urandom(8)
        return int.from_bytes(raw, "big") % 3

class CensorModel:
    def __init__(self):
        self.table = {}
        self.prev = None
        self.window = deque(maxlen=200)
        self.hit_sum = 0
        self.last_prediction = 0
        self.freeze_counter = 0
    def observe(self, path):
        if self.prev is None:
            pred = 0
        else:
            counts = self.table.get(self.prev)
            if counts is None or sum(counts) == 0:
                pred = 0
            else:
                pred = max(range(3), key=lambda i: counts[i])
        hit = 1 if pred == path else 0
        if len(self.window) == self.window.maxlen:
            self.hit_sum -= self.window[0]
        self.window.append(hit)
        self.hit_sum += hit
        if self.prev is not None:
            counts = self.table.setdefault(self.prev, [0, 0, 0])
            counts[path] += 1
        self.prev = path
        self.last_prediction = pred
        if len(self.window) == self.window.maxlen and self.hit_rate() > 0.6:
            self.freeze_counter += 1
        else:
            self.freeze_counter = 0
        return pred, hit
    def hit_rate(self):
        if not self.window:
            return 0.0
        return self.hit_sum / len(self.window)

class TrafficPanel(QWidget):
    def __init__(self, title_drbg, title_qrng, source, world):
        super().__init__()
        self.source = source
        self.world = world
        self.title_drbg = title_drbg
        self.title_qrng = title_qrng
        self.censor = CensorModel()
        self.source_label = getattr(source, "label", "")
        self.packets = []
        self.highlight_lane = 0
        self.freeze = False
        self.flowing_text = False
    def logical_to_px(self, logical_x, left, width):
        return left + (logical_x / 1000.0) * width
    def step(self):
        if self.world == "drbg" and self.freeze:
            return
        self._move_packets()
        path = self.source.next_path()
        pred, hit = self.censor.observe(path)
        self.highlight_lane = pred
        if self.world == "drbg" and self.censor.freeze_counter >= 20:
            self.freeze = True
            return
        self._spawn_packet(path, hit)
    def _spawn_packet(self, path, hit):
        state = "blocked" if hit else "normal"
        packet = {"lane": path, "logical": 0.0, "state": state, "anim": ANIM_FRAMES}
        self.packets.append(packet)
    def _move_packets(self):
        speed = 42
        censor_logical = 510
        dest_logical = 1000
        for packet in list(self.packets):
            packet["logical"] += speed
            if packet["state"] == "blocked":
                if packet["logical"] > censor_logical:
                    packet["logical"] = censor_logical
                if packet["logical"] >= censor_logical:
                    if packet["anim"] > 0:
                        packet["anim"] -= 1
                    else:
                        self.packets.remove(packet)
                continue
            if packet["state"] == "normal" and packet["logical"] >= censor_logical:
                packet["state"] = "delivered"
                packet["anim"] = ANIM_FRAMES
            if packet["state"] == "delivered":
                if packet["anim"] > 0:
                    packet["anim"] -= 1
                else:
                    packet["state"] = "cooled"
            if packet["logical"] >= dest_logical:
                if packet in self.packets:
                    self.packets.remove(packet)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BG)
        width = self.width() - 2 * MARGIN_X
        left = MARGIN_X
        top = 0
        half_height = self.height()
        header_rect = (left, top, width, HEADER_H)
        footer_rect = (left, top + half_height - FOOTER_H, width, FOOTER_H)
        lanes_top = top + HEADER_H + 10
        lanes_height = max(half_height - HEADER_H - FOOTER_H - 20, LANE_H * 3 + LANE_SPACING * 2)
        total_lane_span = LANE_H * 3 + LANE_SPACING * 2
        start_y = lanes_top + (lanes_height - total_lane_span) / 2
        lane_positions = [start_y + i * (LANE_H + LANE_SPACING) + LANE_H / 2 for i in range(3)]
        painter.setPen(QPen(LANE_COLOR, 2))
        for y in lane_positions:
            painter.drawLine(int(left), int(y), int(left + width), int(y))
        src_px = self.logical_to_px(0, left, width)
        dst_px = self.logical_to_px(1000, left, width)
        painter.setBrush(NODE_COLOR)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(src_px - LANE_H / 2), int(lane_positions[1] - LANE_H / 2), int(LANE_H), int(LANE_H))
        painter.drawEllipse(int(dst_px - LANE_H / 2), int(lane_positions[1] - LANE_H / 2), int(LANE_H), int(LANE_H))
        censor_px = self.logical_to_px(510, left, width)
        bar_top = lane_positions[0] - (LANE_H + LANE_SPACING)
        bar_bottom = lane_positions[-1] + (LANE_H + LANE_SPACING)
        painter.setBrush(CENSOR_BAR)
        painter.drawRoundedRect(int(censor_px - CENSOR_W / 2), int(bar_top), int(CENSOR_W), int(bar_bottom - bar_top), 24, 24)
        lane_center = lane_positions[self.highlight_lane]
        painter.setBrush(CENSOR_ACTIVE)
        painter.drawRoundedRect(int(censor_px - (CENSOR_W - 10) / 2), int(lane_center - LANE_H / 2), int(CENSOR_W - 10), int(LANE_H), 18, 18)
        for packet in self.packets:
            x_px = self.logical_to_px(packet["logical"], left, width)
            y_px = lane_positions[packet["lane"]]
            if packet["state"] == "blocked":
                radius = PACKET_R
                alpha = max(0, int(255 * (packet["anim"] / max(1, ANIM_FRAMES))))
                color = QColor(PACKET_BLOCKED)
                color.setAlpha(alpha)
            elif packet["state"] == "delivered":
                radius = PACKET_R + int((ANIM_FRAMES - packet["anim"]) * 3 / ANIM_FRAMES)
                color = QColor(PACKET_PASS)
            elif packet["state"] == "cooled":
                radius = PACKET_R
                color = QColor(PACKET_PASS)
                color.setAlpha(180)
            else:
                radius = PACKET_R
                color = QColor(PACKET_PASS)
            painter.setBrush(color)
            painter.setPen(QPen(QColor("#0a0c10"), 2))
            painter.drawEllipse(int(x_px - radius), int(y_px - radius), int(radius * 2), int(radius * 2))
        painter.setFont(QFont("Inter", 18, QFont.Weight.Bold))
        painter.setPen(QPen(TEXT_COLOR))
        title = self.title_drbg if self.world == "drbg" else self.title_qrng
        painter.drawText(int(header_rect[0]), int(header_rect[1]), int(header_rect[2]), int(header_rect[3]), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, title.format(self.censor.hit_rate() * 100))
        if self.world == "drbg" and (self.freeze or self.censor.hit_rate() > 0.6):
            painter.setFont(QFont("Inter", 20, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#f24f4f")))
            status_top = lanes_top - 42
            painter.drawText(int(left), int(status_top), int(width), 32, Qt.AlignmentFlag.AlignCenter, "censor locked this flow")
        if self.world == "drbg" and self.flowing_text:
            painter.setFont(QFont("Inter", 18, QFont.Weight.Bold))
            painter.setPen(QPen(PACKET_PASS))
            painter.drawText(int(left) + 12, int(header_rect[1] + 18), int(width), 30, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "traffic flowing")
        if self.world == "qrng" and self.source_label:
            painter.setPen(QPen(QColor(180, 190, 210)))
            painter.setFont(QFont("Inter", 13))
            painter.drawText(int(left), int(footer_rect[1]), int(width), int(FOOTER_H), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, self.source_label)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Attack Visual")
        self.resize(WINDOW_MIN_W, WINDOW_MIN_H)
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, BG)
        self.setPalette(palette)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        drbg_title = "DRBG (reseed every 200) – censor hit rate {0:4.1f}%"
        qrng_title = "QRNG (remote/cached) – censor hit rate {0:4.1f}%"
        self.drbg_panel = TrafficPanel(drbg_title, qrng_title, DRBGSource(), "drbg")
        self.qrng_panel = TrafficPanel(drbg_title, qrng_title, QRNGSource(), "qrng")
        layout.addWidget(self.drbg_panel)
        layout.addWidget(self.qrng_panel)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self.fullscreen = False
        shortcut = QShortcut(QKeySequence("F"), self)
        shortcut.activated.connect(self._toggle_fullscreen)
    def _toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self.fullscreen = not self.fullscreen
    def _tick(self):
        self.drbg_panel.step()
        self.qrng_panel.step()
        self.drbg_panel.flowing_text = self.qrng_panel.censor.hit_rate() < 0.45
        self.drbg_panel.update()
        self.qrng_panel.update()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
