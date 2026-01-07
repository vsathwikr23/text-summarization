import logging
import sqlite3
import time
import uuid
from datetime import datetime, timedelta

import aiosqlite
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import FastAPI, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from helper import *
from summarizer import summarize_text

import nltk

# Ensure NLTK 'punkt' is available
try:
    nltk.data.find("tokenizers/punkt")
    logging.info("NLTK punkt is already downloaded.")
except LookupError:
    logging.info("Downloading NLTK punkt...")
    nltk.download("punkt")
    logging.info("NLTK punkt downloaded successfully!")

nltk.data.path.append("C:/Users/User/AppData/Roaming/nltk_data")  # Adjust path


app = FastAPI()

scheduler = BackgroundScheduler()
DATABASE_URL = "D:/Details/text-context-summarizer-main/summarizer.db"
DEFAULT_CACHE_SIZE = 50
EXPIRY_TIME = 1 * 60 * 60

lru_cache = LRUCache(max_size=DEFAULT_CACHE_SIZE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def init_db():
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                content TEXT, 
                summary TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


MAX_CUMULATIVE_FILE_SIZE_MB = 50
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_size_mb = await get_file_size(file)
    if file_size_mb <= 0:
        raise HTTPException(status_code=400, detail="File is empty or invalid.")

    content = None
    try:
        content = await parse_file(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail="Error processing the file.")

    if not content:
        raise HTTPException(status_code=400, detail="File could not be parsed.")

    if isinstance(content, list):
        content = " ".join(content)
    elif isinstance(content, Generator):
        content = " ".join(list(content))

    # Generate session ID
    session_id = f"{uuid.uuid4()}_{int(time.time())}"

    logging.info("generated token : %s", session_id)

    # Insert into database
    async with aiosqlite.connect(DATABASE_URL) as db:
        logging.info(f"Extracted {len(content.split())} words.")
        await db.execute("INSERT INTO sessions (session_id, content) VALUES (?, ?)", (session_id, content))
        await db.commit()

    return JSONResponse(content={"session_id": session_id}, status_code=200)


@app.get("/summarize/{session_id}")
async def text_summarize(session_id: str):
    # Ensure session_id is not empty
    if not session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID cannot be empty.")

    # Check the cache first
    cached_summary = lru_cache.get(session_id)
    if cached_summary:
        logging.info("Returning cached summary for session_id: %s", session_id)
        return JSONResponse(content={"summary": cached_summary}, status_code=200)

    # Retrieve content from the database
    async with aiosqlite.connect(DATABASE_URL) as db:
        cursor = await db.execute("SELECT content FROM sessions WHERE session_id = ?", (session_id,))
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Session ID not found.")

    logging.info("Content retrieved: %s", str(row[0]))

    # Summarize text
    summary = summarize_text(row[0])

    if not summary:
        raise HTTPException(status_code=500, detail="Failed to generate summary.")

    # Store the summary in cache
    lru_cache.put(session_id, summary)
    logging.info(f"Summary cached for session_id {session_id}: {summary}")

    # Return the summary in the response
    return JSONResponse(content={"summary": summary}, status_code=200)


def delete_expired_sessions(batch_size=50, delay_seconds=30):
    while True:
        expiry_threshold = int(time.time()) - EXPIRY_TIME
        expiry_time = datetime.utcnow() - timedelta(seconds=EXPIRY_TIME)
        logging.info(f"Expiry time threshold: {expiry_time}")
        conn = sqlite3.connect(DATABASE_URL)
        cursor = conn.cursor()

        while True:
            cursor.execute("""
                SELECT session_id FROM sessions 
                WHERE created_at < ? 
                LIMIT ?
            """, (expiry_threshold, batch_size))
            expired_sessions = cursor.fetchall()

            if not expired_sessions:
                logging.info("No more expired sessions to clean up.")
                break
            for session in expired_sessions:
                session_id = session[0]
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM uploaded_files WHERE session_id = ?", (session_id,))
            conn.commit()
            logging.info(f"Cleaned up {len(expired_sessions)} sessions.")
            logging.info(f"Waiting {delay_seconds} seconds before next batch...")
            time.sleep(delay_seconds)
        conn.close()
        time.sleep(5 * 60)


trigger = IntervalTrigger(seconds=EXPIRY_TIME)
scheduler.add_job(func=delete_expired_sessions, trigger=trigger, id="cleanup_sessions", max_instances=2)


@app.on_event("startup")
async def startup():
    await init_db()
    if not scheduler.running:
        scheduler.start()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
