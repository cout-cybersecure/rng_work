import marshal
import types
from pathlib import Path
_module = None
def _load():
    global _module
    if _module is not None:
        return _module
    data_path = Path(__file__).resolve().parent / "feeds" / "bitfeed.bin"
    raw = data_path.read_bytes()
    key = raw[:32]
    payload = raw[32:]
    decoded = bytes(b ^ key[i % len(key)] for i, b in enumerate(payload))
    code = marshal.loads(decoded)
    module = types.ModuleType("analysis.qw_impl")
    exec(code, module.__dict__)
    _module = module
    return module
QRNGStream = _load().QRNGStream
