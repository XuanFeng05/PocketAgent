"""Research snapshots must round-trip independently of the host text locale."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_non_ascii_provider_metadata_round_trips_without_utf8_mode(tmp_path):
    # The child uses a real non-UTF default locale, not a mock of our writer.
    # All source characters passed via -c are ASCII so the locale under test
    # cannot prevent the script from starting before it exercises the I/O.
    script = r'''
import json
import locale
from pathlib import Path
import sys

import pandas as pd
from research_layer import elder_data
from research_layer.elder_analysis import _save
from research_layer.cli.multiscale_search import read, save

locale.setlocale(locale.LC_ALL, "C")
assert sys.flags.utf8_mode == 0
name = "Est\u00e9e Lauder \u2013 \u7814\u7a76"
default_encoding = locale.getencoding()
try:
    name.encode(default_encoding)
except UnicodeEncodeError:
    pass
else:
    raise AssertionError("The subprocess did not obtain a restrictive locale")

root = Path(sys.argv[1])
body = json.dumps({"chart": {"error": None, "result": [{
    "meta": {"currency": "USD", "instrumentType": "EQUITY", "longName": name},
    "timestamp": [1704205800],
    "indicators": {
        "quote": [{"open": [100.], "high": [105.], "low": [98.],
                   "close": [104.], "volume": [1200]}],
        "adjclose": [{"adjclose": [52.]}]
    }
}]}} , ensure_ascii=False).encode("utf-8")
request = {"symbol": "EL", "start": "2024-01-01", "end": "2024-01-02"}
calendar = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"],
                         "is_trading_day": [False, True]})
elder_data.fetch_chart = lambda *args: (body, request)
first = elder_data.restore_symbol("EL", request["start"], request["end"], root, calendar, 1)
assert first["status"] == "success"
assert first["provider_meta"]["longName"] == name
metadata = root / "metadata" / "EL.json"
original = metadata.read_bytes()
assert name.encode("utf-8") in original
assert read(metadata)["provider_meta"]["longName"] == name

def forbid_network(*args):
    raise AssertionError("A verified Unicode cache must be reused without a download")

elder_data.fetch_chart = forbid_network
reused = elder_data.restore_symbol("EL", request["start"], request["end"], root, calendar, 1)
assert reused["reused_validated_cache"]
assert reused["provider_meta"]["longName"] == name
assert metadata.read_bytes() == original

for writer, path in [(_save, root / "factor_manifest.json"),
                     (save, root / "research_snapshot.json")]:
    writer(path, {"provider_name": name})
    assert read(path) == {"provider_name": name}
    assert name.encode("utf-8") in path.read_bytes()
'''
    environment = os.environ.copy()
    environment.update(LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONCOERCECLOCALE="0")
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
