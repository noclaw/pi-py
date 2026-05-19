"""webhook_agent.py — FastAPI endpoint that runs an agent task.

This is the noclaw migration pattern: replace subprocess calls to an external
CLI with a direct pi_agent call, keeping the same HTTP interface. The server
accepts a prompt, runs the agent, and returns structured JSON.

Requirements:
    pip install fastapi uvicorn

Usage:
    uv run uvicorn examples.webhook_agent:app --port 8080

    # Run a task
    curl -X POST http://localhost:8080/run \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "List the Python files in /tmp"}'

    # Resume a session
    curl -X POST http://localhost:8080/run \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Continue the task", "session_id": "abc12345"}'

    # Stream events as Server-Sent Events
    curl -N http://localhost:8080/run/stream \\
      -X POST -H "Content-Type: application/json" \\
      -d '{"prompt": "Summarise README.md"}'
"""
import asyncio
import json
import os
import time
from typing import AsyncGenerator

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
except ImportError:
    raise SystemExit(
        "FastAPI is required for this example.\n"
        "Install it with: pip install fastapi uvicorn"
    )

import pi_agent

app = FastAPI(title="pi-agent webhook", version="0.1.0")

# ── Configuration ──────────────────────────────────────────────────────────────

SESSIONS_DIR = os.environ.get("PI_SESSIONS_DIR", os.path.expanduser("~/.pi/webhook-sessions"))
DEFAULT_CWD   = os.environ.get("PI_CWD", os.getcwd())


# ── Request / response models ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    prompt: str
    session_id: str | None = None
    model: str | None = None          # "provider:model-id"
    cwd: str | None = None
    tools: bool = True
    system_prompt: str | None = None


class RunResponse(BaseModel):
    status: str                       # "success" | "error"
    output: str
    session_id: str
    tokens_in: int
    tokens_out: int
    cost: float
    elapsed_ms: int
    error: str | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_model(model_str: str | None):
    if not model_str:
        return None
    import pi_ai
    try:
        provider, model_id = model_str.split(":", 1)
        try:
            return pi_ai.get_model(provider, model_id)
        except Exception:
            m = pi_agent.find_custom_model(provider, model_id)
            if m:
                return m
    except ValueError:
        pass
    raise HTTPException(status_code=400, detail=f"Unknown model: {model_str!r}")


async def _build_harness(req: RunRequest) -> tuple[pi_agent.AgentHarness, str]:
    model = _resolve_model(req.model)
    cwd = os.path.abspath(req.cwd or DEFAULT_CWD)

    harness = await pi_agent.create_agent(
        model=model,
        cwd=cwd,
        sessions_dir=SESSIONS_DIR,
        session_id=req.session_id,
        tools="all" if req.tools else None,
        system_prompt=req.system_prompt,
    )
    meta = await harness._session.get_metadata()
    return harness, meta["id"]


# ── POST /run — buffered response ──────────────────────────────────────────────

@app.post("/run", response_model=RunResponse)
async def run(req: RunRequest) -> RunResponse:
    """Run a prompt and return the complete response as JSON."""
    start = time.monotonic()

    try:
        harness, session_id = await _build_harness(req)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    try:
        reply = await harness.prompt(req.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed_ms = int((time.monotonic() - start) * 1000)
    output = " ".join(
        b.text for b in (reply.content or [])
        if getattr(b, "type", None) == "text" and b.text
    )

    return RunResponse(
        status="error" if reply.stop_reason in ("error", "aborted") else "success",
        output=output,
        session_id=session_id,
        tokens_in=reply.usage.input,
        tokens_out=reply.usage.output,
        cost=reply.usage.cost.total if reply.usage.cost else 0.0,
        elapsed_ms=elapsed_ms,
        error=reply.error_message,
    )


# ── POST /run/stream — Server-Sent Events ─────────────────────────────────────

@app.post("/run/stream")
async def run_stream(req: RunRequest) -> StreamingResponse:
    """Run a prompt and stream AgentEvents as Server-Sent Events."""

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        def on_event(event: dict, *_) -> None:
            queue.put_nowait(event)

        try:
            harness, session_id = await _build_harness(req)
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
            return

        # Send the session ID first so callers can resume later
        yield f"data: {json.dumps({'type': 'session_start', 'session_id': session_id})}\n\n"

        harness.subscribe(on_event)

        # Run the agent in a background task, draining the queue while it runs
        async def _run() -> None:
            try:
                await harness.prompt(req.prompt)
            finally:
                queue.put_nowait(None)  # sentinel

        task = asyncio.create_task(_run())

        while True:
            event = await queue.get()
            if event is None:
                break
            try:
                yield f"data: {json.dumps(event, default=str)}\n\n"
            except Exception:
                pass  # skip non-serialisable events

        await task

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── GET /sessions ──────────────────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions(cwd: str | None = None) -> list[dict]:
    """List saved sessions."""
    from pi_agent.harness.env.python import PythonExecutionEnv
    from pi_agent.harness.session.jsonl_repo import JsonlSessionRepo

    if not os.path.exists(SESSIONS_DIR):
        return []

    env = PythonExecutionEnv(cwd=SESSIONS_DIR)
    repo = JsonlSessionRepo(fs=env, sessions_root=SESSIONS_DIR)
    sessions = await repo.list(cwd=cwd)
    return [
        {
            "id": s["id"],
            "created_at": s.get("createdAt"),
            "cwd": s.get("cwd"),
        }
        for s in sessions
    ]


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webhook_agent:app", host="0.0.0.0", port=8080, reload=True)
