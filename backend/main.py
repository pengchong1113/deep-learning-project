"""
FastAPI backend for the Fitness Assistant.

Usage:
    uvicorn backend.main:app --reload
"""

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.predictor import Predictor

STATIC_DIR = Path(__file__).parent.parent / 'frontend' / 'static'

predictor: Predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = Predictor()
    yield
    predictor.close()


app = FastAPI(title='Fitness Assistant API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.get('/')
def index():
    return FileResponse(str(STATIC_DIR / 'index.html'))


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail='File must be a video')

    suffix = Path(file.filename).suffix or '.mp4'

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predictor.predict(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if 'error' in result:
        raise HTTPException(status_code=422, detail=result['error'])

    return result
