import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from .inference import DecisionModel
from .schema import ScoreRequest

WARMUP = {"context": "warmup", "instructions": "Choose a route.", "candidates": [
    {"id": "a", "description": "First route"}, {"id": "b", "description": "Second route"}]}


def create_app(loader=None, queue_size=8, timeout=30.0):
    if queue_size < 1 or timeout <= 0:
        raise ValueError("queue size and timeout must be positive")

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        app.state.pending = 0
        app.state.executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_running_loop()
        load = loader or (lambda: DecisionModel.load(os.environ["MYJEV_ARTIFACT"],
            revision=os.getenv("MYJEV_REVISION"), device=os.getenv("MYJEV_DEVICE", "cuda:0")))
        try:
            app.state.model = await loop.run_in_executor(app.state.executor, load)
            await loop.run_in_executor(app.state.executor, app.state.model.score, WARMUP)
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            app.state.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"status": "alive"}

    @app.get("/readyz")
    @app.get("/health")
    async def ready():
        if not getattr(app.state, "ready", False):
            raise HTTPException(503, "model not ready")
        return {"status": "ready"}

    @app.post("/score")
    @app.post("/generate")
    async def score(request: ScoreRequest):
        if not app.state.ready:
            raise HTTPException(503, "model not ready")
        if app.state.pending >= queue_size:
            raise HTTPException(429, "queue full", headers={"Retry-After": "1"})
        app.state.pending += 1
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(app.state.executor, app.state.model.score, request)
        # A client timeout cannot interrupt a running CUDA kernel. Capacity is released
        # only on actual completion, preventing unbounded work after repeated timeouts.
        def completed(done):
            app.state.pending -= 1
            if not done.cancelled():
                done.exception()  # retrieve failures even after a client timeout
        future.add_done_callback(completed)
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout)
        except asyncio.TimeoutError:
            raise HTTPException(504, "inference deadline exceeded") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
    return app


def app_factory():
    return create_app(queue_size=int(os.getenv("MYJEV_QUEUE_SIZE", "8")),
                      timeout=float(os.getenv("MYJEV_TIMEOUT", "30")))
