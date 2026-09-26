"""Serve the dependency-free review interface and its cached demo runs."""

from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.loop import run
from core.pubchem import PubChemError, fetch_compound
from frontend.static_adapter import to_static_run


ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/pubchem/compound":
            super().do_GET()
            return
        name = parse_qs(parsed.query).get("name", [""])[0]
        try:
            compound = fetch_compound(name)
        except PubChemError as exc:
            self._json(404, {"error": str(exc)})
            return
        self._json(
            200,
            {
                "name": compound.name,
                "smiles": compound.smiles,
                "original_smiles": compound.original_smiles,
                "molecular_formula": compound.molecular_formula,
                "iupac_name": compound.iupac_name,
                "pubchem_cid": compound.cid,
                "source": f"PubChem CID {compound.cid}",
                "source_url": compound.source_url,
                "structure_url": (
                    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                    f"{compound.cid}/PNG"
                ),
                "target": "",
                "rationale": "",
                "objective": (
                    "Preserve target-relevant chemistry while improving "
                    "solubility, hERG safety, and synthetic accessibility."
                ),
                "bbb_goal": "neutral",
            },
        )

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/optimize":
            self._json(404, {"error": "Unknown endpoint."})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 65536)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            for field in ("name", "smiles", "target", "rationale", "objective"):
                if not str(payload.get(field, "")).strip():
                    raise ValueError(f"{field.replace('_', ' ').title()} is required.")
            native = run(
                seed=payload["smiles"],
                rounds=5,
                verbose=False,
                target=payload["target"],
                compound_name=payload["name"],
                rationale=payload["rationale"],
                objective=payload["objective"],
                bbb_goal=payload.get("bbb_goal", "neutral"),
                source=payload.get("source", "PubChem"),
            )
            response = to_static_run(native, payload["name"])
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._json(400, {"error": str(exc)})
            return
        except Exception as exc:
            self._json(500, {"error": f"Optimization failed: {exc}"})
            return
        self._json(200, response)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="allow configured Anthropic or Bedrock calls during new PubChem runs",
    )
    args = parser.parse_args()
    if not args.with_llm:
        os.environ["AGENT_LLM"] = "off"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(
        "AgentOptim static demo: "
        f"http://127.0.0.1:{args.port}/static/index.html"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
