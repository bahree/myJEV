import json
from pathlib import Path
import httpx
request = json.loads(Path("examples/request.json").read_text())
response = httpx.post("http://127.0.0.1:8000/score", json=request, timeout=60)
response.raise_for_status()
print(response.json())
