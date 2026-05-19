from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ANALYSIS_FILE = Path(os.getenv("ANALYSIS_FILE", "50_analysis.xlsx"))
TIME_FILE = Path(os.getenv("TIME_FILE", "DATA_time.xlsx"))
UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "uploads"))
CACHE_TTL = int(os.getenv("CACHE_TTL", 300))
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
