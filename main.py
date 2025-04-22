import uvicorn
import torch
import os
import aiofiles

from fastapi import FastAPI, File, UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mhmr.run import HumanMeshRecovery

from dotenv import load_dotenv 
load_dotenv()

app = FastAPI(
    title="HP team",
    description="HealthPartner in 2025-1 DCU SW Capstone",
    version="1.0.0",
    contact={
        "name": "API git",
        "url": "https://github.com/SW-HP/hmr-anthropometry",
    },
    license_info={
        "name": "MIT License",
        "url": "https://github.com/SW-HP/hmr-anthropometry",
    }
)

# 이미지 업로드 경로
UPLOAD_DIRECTORY = "./uploaded_images"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# 데이터베이스 연결 설정
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r".*dinov2\.layers\.swiglu_ffn"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r".*dinov2\.layers\.attention"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r".*dinov2\.layers\.block"
)

renderer = HumanMeshRecovery(device=torch.device('cuda'))

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...), _fov: int=60):
    # 파일 저장
    file_location = os.path.join(UPLOAD_DIRECTORY, file.filename)
    async with aiofiles.open(file_location, "wb") as buffer:
        content = await file.read()
        await buffer.write(content)

    # 이미지 처리
    humans = renderer.process_image(file_location, fov=_fov)
    result = renderer.measure_human(humans)

    # 결과 반환
    return {"filename": file.filename, "measurements": result}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)