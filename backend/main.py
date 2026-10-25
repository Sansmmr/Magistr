"""
FastAPI backend for Multi-Agent Test Generation System
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from starlette.websockets import WebSocketDisconnect as StarletteWebSocketDisconnect
from file_parser import FileParser
from database_simple import db

app = FastAPI(title="Multi-Agent Test Generation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        stale_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except (WebSocketDisconnect, StarletteWebSocketDisconnect, RuntimeError):
                stale_connections.append(connection)
        for connection in stale_connections:
            self.disconnect(connection)

manager = ConnectionManager()

# Request models
class TestGenerationRequest(BaseModel):
    source_text: str
    source_chunks: Optional[list] = None
    original_question: Optional[str] = None
    original_options: Optional[list] = None
    max_iterations: Optional[int] = None
    question_count: Optional[int] = 1
    difficulty: Optional[str] = "medium"
    test_mode: Optional[bool] = False  # For testing without API keys
    configuration: Optional[str] = "M1"  # M1: без ітерацій, M2: з ітераціями

# WebSocket endpoint for real-time logs
@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for now - will be used for client commands
            await websocket.send_json({"type": "echo", "message": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Documents endpoints
@app.get("/api/documents")
async def get_documents():
    """Get all documents"""
    try:
        documents = db.get_all_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    """Get document by ID"""
    try:
        document = db.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# File upload endpoint (supports multiple formats)
@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and parse a file to extract text
    Supports: PDF, TXT, DOCX, CSV, Parquet, JSON
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Parse file using universal parser
        parser = FileParser()
        result = parser.parse_file(file_content, file.filename)
        
        if not result.get("text"):
            raise HTTPException(status_code=400, detail="Could not extract text from file")
        
        chunks = result.get("chunks") or []
        return {
            "filename": file.filename,
            "format": result.get("format"),
            "text": result.get("text"),
            "text_length": len(result.get("text", "")),
            "chunks": chunks,
            "chunk_count": result.get("chunk_count", len(chunks)),
            "rows": result.get("rows"),
            "columns": result.get("columns"),
            "column_names": result.get("column_names"),
            "preview": result.get("preview"),
            "truncated": result.get("truncated", False),
            "original_rows": result.get("original_rows"),
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Test generation endpoint
@app.post("/api/generate-test")
async def generate_test(request: TestGenerationRequest):
    """
    Generate a test question using multi-agent LangGraph system

    Configuration variants:
    - M1: No iterations (single pass, no critic)
    - M2: With iterations (standard)
    - M3: Limited iterations (max 2)
    - M4: Full system with all improvements
    """
    try:
        # Import LangGraph here to avoid circular imports
        from difficulty_utils import normalize_difficulty, max_iterations_for_difficulty
        from langgraph_graph import run_test_generation_graph

        difficulty = normalize_difficulty(request.difficulty or "medium")
        configuration = request.configuration or "M4"  # Default to M4

        # Fixed iteration limits per configuration (reproducible experiments)
        max_iterations = {"M1": 1, "M2": 1, "M3": 1, "M4": 2}.get(
            configuration, request.max_iterations or max_iterations_for_difficulty(difficulty)
        )

        # Create a log callback for WebSocket broadcasting
        async def log_callback(message: dict):
            await manager.broadcast({
                **message,
                "timestamp": datetime.now().isoformat()
            })
        
        # Run the LangGraph pipeline
        final_state = await run_test_generation_graph(
            source_text=request.source_text,
            source_chunks=request.source_chunks,
            original_question=request.original_question,
            original_options=request.original_options,
            max_iterations=max_iterations,
            question_count=request.question_count or 1,
            difficulty=difficulty,
            log_callback=log_callback,
            test_mode=request.test_mode or False,
            configuration=configuration
        )
        
        result = final_state.get("final_result")
        question_errors = [
            q["error"]
            for q in (result or {}).get("questions", [])
            if q.get("error")
        ]
        if question_errors:
            error_message = question_errors[0]
            await manager.broadcast({
                "type": "error",
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            })
            return {
                "success": False,
                "error": error_message,
                "result": result
            }
        
        # Send completion message
        await manager.broadcast({
            "type": "complete",
            "message": "Test generation completed",
            "result": final_state.get("final_result"),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "result": final_state.get("final_result")
        }
        
    except Exception as e:
        await manager.broadcast({
            "type": "error",
            "message": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------- лекційний конвеєр (повна конфігурація M4)
import asyncio, os, json as _json
from pydantic import Field

WEB_RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_runs")
LECTURE_MIN_Q, LECTURE_MAX_Q = 3, 30          # обмеження кількості завдань у вебзастосунку


class LectureRequest(BaseModel):
    text: str
    question_count: int = Field(12, ge=LECTURE_MIN_Q, le=LECTURE_MAX_Q)
    filename: Optional[str] = None


@app.post("/api/generate-lecture")
async def generate_lecture(request: LectureRequest):
    """Повна конфігурація методу (M4) для лекції користувача: граф lecture_graph.py
    (аналітик -> розподіл -> генератор -> критик із циклом доопрацювання). Події передаються через /ws/logs."""
    import critic as C
    import lecture_graph as LG
    text = (request.text or "").strip()
    if len(text) < 200:
        raise HTTPException(status_code=400, detail="Замало тексту: потрібно щонайменше 200 символів лекції.")
    try:
        C.check_lecture_size(text)
    except C.LectureTooLarge as e:
        raise HTTPException(status_code=400, detail=str(e))

    doc_id = db.create_document(text)                       # документ зберігається в SQLite (таблиця documents)
    outdir = os.path.join(WEB_RUNS, doc_id); os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "lecture.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(outdir, "request.json"), "w", encoding="utf-8") as f:
        _json.dump({"filename": request.filename, "question_count": request.question_count,
                    "created_at": datetime.now().isoformat()}, f, ensure_ascii=False, indent=1)

    loop = asyncio.get_running_loop()
    def log(event: dict):                                   # виклик із робочого потоку конвеєра
        asyncio.run_coroutine_threadsafe(manager.broadcast({**event, "run_id": doc_id,
                                                            "timestamp": datetime.now().isoformat()}), loop)
    await manager.broadcast({"type": "log", "agent": "System", "message": "Запуск конвеєра агентів (M4)",
                             "run_id": doc_id, "timestamp": datetime.now().isoformat()})
    try:
        summary = await asyncio.to_thread(LG.run_lecture, path, outdir, request.question_count, "m4", None, 6, log)
    except Exception as e:
        await manager.broadcast({"type": "error", "message": f"Помилка конвеєра: {e}", "run_id": doc_id,
                                 "timestamp": datetime.now().isoformat()})
        raise HTTPException(status_code=500, detail=f"Помилка конвеєра: {e}. Повторний запуск продовжить прогін.")

    result = LG.collect(outdir); result["run_id"] = doc_id
    try:                                                    # підсумкові версії завдань — у таблицю questions
        test_id = db.create_generated_test(doc_id)
        for it in result["items"]:
            m = it["final"]
            db.create_question(test_id, m["question"],
                               {"A": m["key"], "B": m["distractors"][0]["text"], "C": m["distractors"][1]["text"],
                                "D": m["distractors"][2]["text"]}, "A",
                               analysis={"block": it["block"], "block_title": it["block_title"], "target_level": it["target_level"]},
                               verification={"status": it["status"], "iterations": len(it["iterations"])})
    except Exception as e:
        print("Не вдалося записати завдання в базу:", e)
    await manager.broadcast({"type": "complete", "message": "Генерацію завершено", "result": result,
                             "run_id": doc_id, "timestamp": datetime.now().isoformat()})
    return {"success": True, "result": result}


@app.get("/api/lecture-runs/{run_id}")
async def get_lecture_run(run_id: str):
    """Результат збереженого прогону (каталог web_runs/<run_id>)."""
    import lecture_graph as LG
    outdir = os.path.join(WEB_RUNS, os.path.basename(run_id))
    if not os.path.isdir(outdir):
        raise HTTPException(status_code=404, detail="Прогін не знайдено")
    r = LG.collect(outdir); r["run_id"] = run_id
    return r


@app.post("/api/lecture-runs/{run_id}/resume")
async def resume_lecture_run(run_id: str):
    """Продовження перерваного прогону: вузли графа пропускають уже збережені кроки."""
    import lecture_graph as LG
    outdir = os.path.join(WEB_RUNS, os.path.basename(run_id)); path = os.path.join(outdir, "lecture.txt")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Прогін не знайдено")
    req = _json.load(open(os.path.join(outdir, "request.json"), encoding="utf-8"))
    loop = asyncio.get_running_loop()
    def log(event: dict):
        asyncio.run_coroutine_threadsafe(manager.broadcast({**event, "run_id": run_id, "timestamp": datetime.now().isoformat()}), loop)
    await asyncio.to_thread(LG.run_lecture, path, outdir, req["question_count"], "m4", None, 6, log)
    r = LG.collect(outdir); r["run_id"] = run_id
    await manager.broadcast({"type": "complete", "message": "Генерацію завершено", "result": r, "run_id": run_id,
                             "timestamp": datetime.now().isoformat()})
    return {"success": True, "result": r}


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server on http://localhost:8000")
    print("WebSocket endpoint: ws://localhost:8000/ws/logs")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
