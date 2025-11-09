import os
import json
from urllib import request, parse, error
from pathlib import Path

class QRNGStream:
    def __init__(self):
        self.buffer = bytearray()
        self.last_error = None
        self.mode = None
        self.capture_path = Path(__file__).resolve().parent / "qrng_capture.bin"
        self.capture_data = None
        self.capture_pos = 0
        self.primary_url = os.environ.get("QRNG_URL")
        self.primary_key = os.environ.get("QRNG_KEY") or ""
        if not self.primary_url:
            self.mode = "cached qrng capture" if self.capture_path.exists() else "local stand-in"
        else:
            self.mode = "remote qrng"
    def source_name(self):
        return self.mode or "local stand-in"
    def next_bytes(self, n):
        out = bytearray()
        while len(out) < n:
            if not self.buffer:
                self._refill(max(n, 256))
            take = min(n - len(out), len(self.buffer))
            out.extend(self.buffer[:take])
            del self.buffer[:take]
        return bytes(out)
    def _refill(self, need):
        produced = False
        if self.primary_url:
            try:
                produced = self._fetch_remote(need)
                if produced:
                    self.last_error = None
                    self.mode = "remote qrng"
                    return
            except Exception as exc:
                self.last_error = exc
                self.primary_url = None
        if self.capture_path.exists():
            if self.capture_data is None:
                with self.capture_path.open("rb") as fh:
                    self.capture_data = fh.read()
                if not self.capture_data:
                    self.capture_data = os.urandom(1024)
            chunk = bytearray()
            while len(chunk) < need:
                remaining = len(self.capture_data) - self.capture_pos
                if remaining <= 0:
                    self.capture_pos = 0
                    continue
                take = min(need - len(chunk), remaining)
                chunk.extend(self.capture_data[self.capture_pos:self.capture_pos + take])
                self.capture_pos += take
            self.buffer.extend(chunk)
            self.mode = "cached qrng capture"
            return
        self.buffer.extend(os.urandom(need))
        self.mode = "local stand-in"
    def _fetch_remote(self, need):
        length = max(64, min(256, need))
        url = self._build_url(self.primary_url, length)
        headers = {}
        if self.primary_key:
            headers["x-api-key"] = self.primary_key
        req = request.Request(url, headers=headers)
        try:
            with request.urlopen(req, timeout=5) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"http {resp.status}")
                content_type = resp.headers.get("Content-Type", "")
                body = resp.read()
                if "application/json" in content_type:
                    data = json.loads(body.decode())
                    values = data.get("data")
                    if not values:
                        raise RuntimeError("no data")
                    chunk = bytes(int(x) & 0xff for x in values)
                else:
                    chunk = body
        except error.URLError as exc:
            raise RuntimeError(str(exc))
        if len(chunk) == 0:
            raise RuntimeError("empty response")
        self.buffer.extend(chunk)
        return True
    def _build_url(self, base, length):
        parsed = parse.urlparse(base)
        query = parse.parse_qs(parsed.query)
        query["length"] = [str(length)]
        query["type"] = ["uint8"]
        new_query = parse.urlencode(query, doseq=True)
        return parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
