from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "uploads"))
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB max upload
ALLOWED_VIDEO = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4a', '.wav', '.mp3'}
ALLOWED_SUBTITLE = {'.srt', '.vtt', '.txt'}
ALLOWED_EXCEL = {'.xlsx', '.xls'}
PORT = int(os.getenv("PORT", 5000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
