import argparse
import json
import sys
from .inference import DecisionModel


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--artifact", required=True)
    score.add_argument("--revision")
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--input", default="-")
    score.add_argument("--jsonl", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--artifact", required=True)
    serve.add_argument("--revision")
    serve.add_argument("--device", default="cuda:0")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    if a.command == "serve":
        import os
        import uvicorn
        os.environ["MYJEV_ARTIFACT"] = a.artifact
        os.environ["MYJEV_DEVICE"] = a.device
        if a.revision:
            os.environ["MYJEV_REVISION"] = a.revision
        uvicorn.run("myjev.server:app_factory", factory=True, host=a.host, port=a.port, access_log=False, workers=1)
        return
    model = DecisionModel.load(a.artifact, revision=a.revision, device=a.device)
    source = sys.stdin if a.input == "-" else open(a.input)
    try:
        if a.jsonl:
            for line in source:
                if line.strip():
                    print(json.dumps(model.score(json.loads(line))), flush=True)
        else:
            print(json.dumps(model.score(json.load(source))))
    except (ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
    finally:
        if source is not sys.stdin:
            source.close()


if __name__ == "__main__":
    main()
