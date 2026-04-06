# GUI Implementation Plan

## Stack

| Layer | Stack |
|-------|-------|
| Frontend | SwiftUI |
| Backend | Python, FastAPI, uvicorn |
| Bridge | HTTP `localhost:8765` |
| Vector DB | Qdrant embedded |

---

## Architecture

```
SwiftUI App
  ├── ServerProcess.swift  → spawns: uv run uvicorn src.server:app --port 8765
  ├── APIService.swift     → HTTP calls to localhost:8765
  └── Views                → Ingest | Sources | Search

FastAPI (src/server.py)
  ├── GET  /health
  ├── POST /ingest         → SSE stream
  ├── GET  /sources
  ├── GET  /search/keyword?q=
  └── GET  /search/semantic?q=
```

---

## Phase 1 — Python backend ✅

| File | Change |
|------|--------|
| `src/config.py` | Use `QDRANT_PATH` (absolute); removed Docker host/port |
| `src/database/vector_store.py` | `QdrantClient(path=QDRANT_PATH)` |
| `src/pipeline.py` | Added `on_progress: Callable[[str], None] = print` |
| `src/database/sqlite_store.py` | Added `list_sources()` |
| `src/server.py` | New — FastAPI app with 5 endpoints |
| `requirements.txt` | Added `fastapi`, `uvicorn[standard]` |

### `/ingest` SSE pattern

```python
@app.post("/ingest")
async def ingest_endpoint(body: IngestRequest):
    msg_queue: queue.Queue[str | None] = queue.Queue()

    def on_progress(msg): msg_queue.put(msg)
    def run():
        try:
            pipeline.ingest(body.url, body.language, body.force,
                            body.initial_prompt, on_progress=on_progress)
        except Exception as e:
            msg_queue.put(f"__error__:{e}")
        finally:
            msg_queue.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        loop = asyncio.get_running_loop()
        while True:
            msg = await loop.run_in_executor(None, msg_queue.get)
            if msg is None: break
            if msg.startswith("__error__:"):
                yield f"event: error\ndata: {json.dumps({'message': msg[10:]})}\n\n"
            else:
                yield f"data: {json.dumps({'message': msg})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
```

### Verify
```bash
uv pip install -r requirements.txt
uv run uvicorn src.server:app --port 8765

curl http://localhost:8765/health
curl http://localhost:8765/sources
curl "http://localhost:8765/search/keyword?q=test"
curl "http://localhost:8765/search/semantic?q=test"
```

---

## Phase 2 — SwiftUI app ⏳

```
PodcastExtractor/
├── PodcastExtractorApp.swift
├── Services/
│   ├── ServerProcess.swift   # Process() wrapper for uvicorn subprocess
│   └── APIService.swift      # URLSession async/await wrappers
├── Models/
│   ├── Source.swift
│   └── SearchResult.swift
└── Views/
    ├── ContentView.swift     # TabView: Ingest | Sources | Search
    ├── IngestView.swift
    ├── SourcesView.swift
    └── SearchView.swift
```

### Key gotchas

**`uv` not in subprocess PATH** — resolve it via a login shell first:
```swift
let whichTask = Process()
whichTask.executableURL = URL(fileURLWithPath: "/bin/zsh")
whichTask.arguments = ["-l", "-c", "which uv"]
// capture stdout → uvPath, then launch uvicorn with absolute path
```

**Server readiness** — poll `/health` before enabling the UI:
```swift
func waitUntilReady() async throws {
    for _ in 0..<20 {
        try? await Task.sleep(nanoseconds: 100_000_000)
        if let _ = try? await APIService.shared.health() { return }
    }
    throw ServerError.startupTimeout
}
```

**SSE error events:**
```swift
for try await line in stream.lines {
    if line.hasPrefix("event: error") { /* show error in UI */ }
    else if line.hasPrefix("data: ") { /* append to progress log */ }
}
```

---

## Phase 3 — Packaging (future)

Bundle FastAPI as a standalone binary via PyInstaller; models (~2GB) download on first launch to `~/Library/Application Support/PodcastExtractor/`.
