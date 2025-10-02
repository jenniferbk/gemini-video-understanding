"""
JSON Progress Wrapper for Electron Integration
Adds --json-progress flag to video_transcription_pipeline_v04.py
"""
import json
import sys
from datetime import datetime

# Global flag for JSON mode
JSON_PROGRESS_MODE = False

def report_progress(chunk_num: int, total_chunks: int, status: str, message: str = ""):
    """Output progress in JSON format for Electron consumption"""
    if JSON_PROGRESS_MODE:
        progress = {
            "type": "progress",
            "chunk": chunk_num,
            "total": total_chunks,
            "percent": int((chunk_num / total_chunks) * 100) if total_chunks > 0 else 0,
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        print(f"GVU_PROGRESS:{json.dumps(progress)}", flush=True)

def report_log(level: str, message: str):
    """Output log message in JSON format"""
    if JSON_PROGRESS_MODE:
        log = {
            "type": "log",
            "level": level,  # info, warning, error
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        print(f"GVU_LOG:{json.dumps(log)}", flush=True)

def report_complete(output_file: str, stats: dict):
    """Output completion message with stats"""
    if JSON_PROGRESS_MODE:
        completion = {
            "type": "complete",
            "output_file": str(output_file),
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        print(f"GVU_COMPLETE:{json.dumps(completion)}", flush=True)

def report_error(message: str, chunk: int = None, fatal: bool = True):
    """Output error in JSON format"""
    if JSON_PROGRESS_MODE:
        error = {
            "type": "error",
            "message": message,
            "chunk": chunk,
            "fatal": fatal,
            "timestamp": datetime.now().isoformat()
        }
        print(f"GVU_ERROR:{json.dumps(error)}", flush=True)

# Check for --json-progress flag
if '--json-progress' in sys.argv:
    JSON_PROGRESS_MODE = True
    # Remove the flag so it doesn't interfere with the main script
    sys.argv.remove('--json-progress')
