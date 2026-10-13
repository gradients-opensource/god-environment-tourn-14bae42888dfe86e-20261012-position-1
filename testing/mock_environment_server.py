"""Minimal local mock of an EnvTask environment sidecar (/reset + /step), for
testing the EnvTask rollout loop end-to-end while waiting for real
ENVIRONMENT_SERVER_URLS access from the validator.

This does NOT play a real game — it just keeps each episode alive for a fixed
number of turns then ends with a random reward, so you can verify the full
multi-turn generate -> step -> generate cycle in core/environment_rollout.py
actually runs (action parsing, message history, reward handling) without
needing the real validator's environment server.

Stdlib only (http.server), so it runs anywhere with no pip installs:
    python3 testing/mock_environment_server.py --port 9999
"""

import argparse
import json
import random
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


EPISODES: dict[str, dict] = {}
MAX_TURNS_BEFORE_DONE = 3


def make_observation(env_name: str, turn: int) -> str:
    if env_name == "intercode":
        return (
            f"[mock-intercode turn {turn}] You are in a bash shell. Working directory has a few "
            f"files. Respond with execute_bash(command=\"...\") to run a command, or submit() when done."
        )
    return (
        f"[mock-{env_name} turn {turn}] Board state placeholder. Legal moves: 0-7. "
        f"Respond with: Action: <number>"
    )


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_POST(self):
        try:
            body = self._read_json()
            if self.path == "/reset":
                self._handle_reset(body)
            elif self.path == "/step":
                self._handle_step(body)
            else:
                self._send_json({"error": f"unknown path {self.path}"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def _handle_reset(self, body: dict):
        env_name = self._infer_env_name(body)
        episode_id = str(uuid.uuid4())
        EPISODES[episode_id] = {"env_name": env_name, "turn": 0}
        print(f"[mock-env] /reset task_id={body.get('task_id')} seed={body.get('seed')} -> episode_id={episode_id}", flush=True)
        self._send_json({"episode_id": episode_id, "observation": make_observation(env_name, 0)})

    def _handle_step(self, body: dict):
        episode_id = body.get("episode_id")
        episode = EPISODES.get(episode_id)
        if episode is None:
            self._send_json({"error": f"unknown episode_id {episode_id}"}, status=400)
            return

        episode["turn"] += 1
        turn = episode["turn"]
        done = turn >= MAX_TURNS_BEFORE_DONE
        reward = round(random.uniform(-1.0, 1.0), 2) if done else 0.0
        observation = "" if done else make_observation(episode["env_name"], turn)

        print(f"[mock-env] /step episode_id={episode_id} action={body.get('action')!r} turn={turn} done={done} reward={reward}", flush=True)
        if done:
            del EPISODES[episode_id]

        self._send_json({"observation": observation, "reward": reward, "done": done})

    @staticmethod
    def _infer_env_name(body: dict) -> str:
        # The reset payload doesn't carry env_name explicitly (see
        # core/environment_rollout.py build_reset_payload) — games all send
        # "opponent": "mcts", intercode doesn't. Good enough to vary the mock
        # observation text; doesn't affect correctness of the rollout loop test.
        return "intercode" if "opponent" not in body else "game"

    def log_message(self, format, *args):
        pass  # quiet; we print our own structured lines above


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[mock-env] listening on 0.0.0.0:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
