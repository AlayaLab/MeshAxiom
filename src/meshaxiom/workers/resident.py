"""Private line protocol: one clean model namespace and one resident stage."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["vertex", "topology"], required=True)
    parser.add_argument("--reply-fd", type=int, required=True)
    parser.add_argument("--reference", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    reply = os.fdopen(args.reply_fd, "w", buffering=1)
    function = None
    for line in sys.stdin:
        message = json.loads(line)
        with Path(message["log"]).open("a", buffering=1) as log:
            # Capture Python and native-library logs, while keeping the reply pipe separate.
            saved_out, saved_err = os.dup(1), os.dup(2)
            try:
                os.dup2(log.fileno(), 1)
                os.dup2(log.fileno(), 2)
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    if function is None:
                        module = "reference_" + args.stage if args.reference else args.stage
                        function = importlib.import_module("meshaxiom.workers." + module).run
                    from meshaxiom.vendor.recipe.runtime import read_request
                    result = function(read_request(message["request"]))
                    response = {"id": message["id"], "status": "complete", "report": result}
            except BaseException as exc:
                traceback.print_exc(file=log)
                response = {"id": message["id"], "status": "failed", "error": str(exc), "error_type": type(exc).__name__}
            finally:
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
                os.close(saved_out)
                os.close(saved_err)
        reply.write(json.dumps(response, ensure_ascii=False) + "\n")
        reply.flush()
        if response["status"] != "complete":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
