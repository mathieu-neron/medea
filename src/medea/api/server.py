"""FastAPI HTTP wrapper around ``medea.model.infer.Predictor``.

Same logic as ``scripts/predict.py``; the heavy lifting (encoders + models)
is loaded once in the lifespan handler so the first request doesn't pay the
60-second cold-start.

Run:
    uvicorn medea.api.server:app --host 127.0.0.1 --port 8000

POST /predict
    { "url": "https://...", "k": 5, "model": "mlp" }
GET  /health
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medea.model.infer import Predictor

log = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    url: str = Field(..., description="YouTube URL or 11-char video id.")
    k: int = Field(5, ge=1, le=50, description="Number of neighbors to return.")
    model: Literal["mlp", "logreg"] = Field("mlp")
    explain: bool = Field(False, description="Generate a short rationale via Claude API.")


class NeighborOut(BaseModel):
    video_id: str
    label: int
    channel_id: int
    distance: float


class PredictResponse(BaseModel):
    video_id: str
    title: str | None
    upload_date: str | None
    score: float
    raw_score: float
    prior_cap: float | None
    model: str
    top_neighbors: list[NeighborOut]
    modality_attribution: dict[str, float]
    top_features: list[tuple[str, float]]
    rationale: str | None = None


_predictors: dict[str, Predictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("warming up MLP predictor")
    _predictors["mlp"] = Predictor(model="mlp")
    yield
    _predictors.clear()


app = FastAPI(title="Medea — AI YouTube detector", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "loaded_models": list(_predictors.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    predictor = _predictors.get(req.model)
    if predictor is None:
        # Lazily load the LogReg variant on first request — keeps cold start
        # tied to the primary (MLP) model only.
        log.info("loading on-demand predictor: %s", req.model)
        try:
            predictor = Predictor(model=req.model)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(500, f"failed to load model {req.model!r}: {e}") from e
        _predictors[req.model] = predictor

    try:
        pred = predictor.predict_url(req.url, k=req.k)
    except ValueError as e:  # bad url
        raise HTTPException(400, str(e)) from e
    except RuntimeError as e:  # download failure / too short etc.
        raise HTTPException(422, str(e)) from e

    rationale: str | None = None
    if req.explain:
        from medea.model.explain import explain as explain_fn

        rationale = explain_fn(pred)

    return PredictResponse(
        video_id=pred.video_id,
        title=pred.title,
        upload_date=pred.upload_date,
        score=pred.score,
        raw_score=pred.raw_score,
        prior_cap=pred.prior_cap,
        model=pred.model,
        top_neighbors=[NeighborOut(**asdict(n)) for n in pred.top_neighbors],
        modality_attribution=pred.modality_attribution,
        top_features=pred.top_features,
        rationale=rationale,
    )
