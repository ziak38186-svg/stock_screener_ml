import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd

from models.schemas import UploadResponse

router = APIRouter(prefix="/data", tags=["data"])

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{suffix}' not allowed. Use .csv, .xls, or .xlsx")

    dest_path = UPLOAD_DIR / file.filename
    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    try:
        if suffix in {".xls", ".xlsx"}:
            df = pd.read_excel(dest_path, nrows=5)
        else:
            df = pd.read_csv(dest_path, nrows=5)

        full_df = pd.read_csv(dest_path) if suffix == ".csv" else pd.read_excel(dest_path)
        row_count = len(full_df)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Could not parse file: {e}")

    return UploadResponse(
        filename=file.filename,
        rows=row_count,
        columns=list(df.columns),
        message=f"File '{file.filename}' uploaded successfully with {row_count:,} rows.",
    )


@router.get("/files")
async def list_uploaded_files():
    files = []
    for f in UPLOAD_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append({"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})
    return {"files": files}


@router.delete("/files/{filename}")
async def delete_file(filename: str):
    target = UPLOAD_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    return {"message": f"Deleted {filename}"}
