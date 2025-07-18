import base64
import os
import time
import uuid
from io import BytesIO
from typing import Optional, AsyncIterator
import pandas as pd
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from starlette.responses import FileResponse
import asyncio
from contextlib import asynccontextmanager

from model import preprocess, train_model, use_model
from utils import (MAX_FILE_SIZE_MB, DataColumns, ResultData,
                   MODEL_URL, Session, PERIODIC_CLEANUP_INTERVAL_SECONDS, SESSION_TIMEOUT_SECONDS, MODEL_FILE_EXT,
                   OUT_DIR)

sessions: dict[str, Session] = {}
active_sessions: set[str] = set()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    cleanup_task = asyncio.create_task(run_periodic_cleanup(interval=PERIODIC_CLEANUP_INTERVAL_SECONDS))

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

async def run_periodic_cleanup(interval: int):
    while True:
        await asyncio.sleep(interval)
        cleanup_old_sessions(SESSION_TIMEOUT_SECONDS)


def cleanup_old_sessions(session_timeout_seconds: int):
    current_time = time.time()
    for session_id, session in sessions.items():
        if (session_id not in active_sessions and
                current_time - session.last_activity > session_timeout_seconds):
            del sessions[session_id]
            os.remove(OUT_DIR + session_id + MODEL_FILE_EXT)


@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    sessions[session_id] = Session()
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        raise HTTPException(
            status_code=400,
            detail="Неверное расширение файла. Используйте .csv или .xlsx"
        )

    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Размер файла не должен превышать {MAX_FILE_SIZE_MB} Мб"
        )

    content = await file.read()

    if file.filename.endswith('.csv'):
        df = pd.read_csv(BytesIO(content))
    else:
        df = pd.read_excel(BytesIO(content))

    required_columns = [
        value for name, value in vars(DataColumns).items()
        if not name.startswith('__') and not callable(value)
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Отсутствуют обязательные колонки: {', '.join(missing)}"
        )

    background_tasks.add_task(preprocess_file, df, session_id)

    return JSONResponse(
        status_code=200,
        content={"session_id": session_id}
    )


@app.post("/analyze")
async def analyze(session_id: str, file: Optional[UploadFile] = File(None)):
    if session_id not in sessions.keys():
        raise HTTPException(
            status_code=404,
            detail="Не найдено загруженного файла"
        )

    if sessions[session_id].data is None:
        raise HTTPException(
            status_code=400,
            detail="Файл ещё не обработан"
        )

    active_sessions.add(session_id)
    if file is None:
        x, y, y_pred, model, metrics, feature_importance = train_model(sessions[session_id].data[0],
                                                                       sessions[session_id].data[1])
    else:
        content = await file.read()
        if file.filename.endswith('.pkl'):
            model = pickle.load(BytesIO(content))
            x, y, y_pred, metrics, feature_importance = use_model(model,
                                                                  sessions[session_id].data[0],
                                                                  sessions[session_id].data[1])
        elif file.filename.endswith('.joblib'):
            model = joblib.load(BytesIO(content))
            x, y, y_pred, metrics, feature_importance = use_model(model,
                                                                  sessions[session_id].data[0],
                                                                  sessions[session_id].data[1])
        else:
            active_sessions.discard(session_id)
            raise HTTPException(
                status_code=400,
                detail="Неверное расширение файла. Используйте .pkl или .joblib"
            )

    joblib.dump(model, OUT_DIR + session_id + MODEL_FILE_EXT)
    plot_results = generate_plots(y, y_pred, feature_importance)
    result = {
        ResultData.pred_graph_path: plot_results[0],
        ResultData.feat_graph_path: plot_results[1],
        ResultData.mae: metrics[ResultData.mae],
        ResultData.r2: metrics[ResultData.r2],
        ResultData.str_amount: x.shape[0],
        ResultData.features: x.columns.tolist(),
        ResultData.missing: int(x.isna().sum().sum()),
        ResultData.missing_by_cols: x.isna().sum().to_string(),
        ResultData.model_url: MODEL_URL
    }

    sessions[session_id].result = result
    active_sessions.discard(session_id)

    return JSONResponse(
        status_code=200,
        content={"session_id": session_id}
    )


@app.get("/results")
async def results(session_id: str):
    if session_id not in sessions.keys():
        raise HTTPException(
            status_code=400,
            detail="Не найдено загруженного файла"
        )

    if sessions[session_id].result is None:
        raise HTTPException(
            status_code=400,
            detail="Нет данных анализа"
        )

    return JSONResponse(
        status_code=200,
        content=sessions[session_id].result
    )


@app.get(MODEL_URL)
async def download_model(session_id: str):
    if not os.path.exists(OUT_DIR + session_id + MODEL_FILE_EXT):
        raise HTTPException(
            status_code=400,
            detail="Файл модели не найден"
        )

    return FileResponse(OUT_DIR + session_id + MODEL_FILE_EXT,
                        filename="model.joblib", media_type="application/octet-stream")


def preprocess_file(df: pd.DataFrame, session_id: str) -> None:
    active_sessions.add(session_id)
    sessions[session_id].data = preprocess(df)
    active_sessions.discard(session_id)


def generate_plots(y, y_pred, feature_importance):
    buf = BytesIO()

    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.savefig(buf, format="svg")
    buf.seek(0)
    pred = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    buf = BytesIO()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance.values(), y=feature_importance.keys())
    plt.title("Feature Importance")
    plt.savefig(buf, format="svg")
    buf.seek(0)
    feat = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return pred, feat
