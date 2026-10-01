import os
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from src.pdf_processor import process_uploaded_pdfs
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine

load_dotenv()

app = FastAPI(title="DocuMind AI Direct Web & REST API", description="Direct Web Interface & REST API for DocuMind RAG AI")

vector_store = VectorStore()
rag_engine = RAGEngine()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class UploadWrapper:
    def __init__(self, filename: str, content: bytes):
        self.name = filename
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DocuMind AI — Direct Web App</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen font-sans">
    <div class="max-w-5xl mx-auto px-4 py-8">
        <!-- Header -->
        <header class="text-center mb-8">
            <h1 class="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-500 mb-2">
                📚 DocuMind AI
            </h1>
            <p class="text-gray-400">Direct Web & REST API Interface for Intelligent Document Q&A (RAG)</p>
        </header>

        <!-- Status Card -->
        <div class="bg-gray-800 rounded-xl p-4 mb-6 border border-gray-700 flex flex-wrap justify-between items-center gap-4">
            <div class="flex items-center space-x-3">
                <span id="status-dot" class="h-3 w-3 rounded-full bg-yellow-500"></span>
                <div>
                    <span class="text-sm font-semibold text-gray-300">LLM Provider:</span>
                    <span id="provider-name" class="text-sm text-blue-400 font-bold ml-1">Loading...</span>
                </div>
            </div>
            <div class="text-sm text-gray-300">
                <span>Vector Index:</span>
                <span id="index-status" class="font-bold text-gray-400 ml-1">Empty (0 chunks)</span>
            </div>
            <button onclick="resetSystem()" class="px-3 py-1 bg-red-950 text-red-400 border border-red-800 rounded-lg hover:bg-red-900 text-xs font-semibold">
                <i class="fa-solid fa-rotate-right mr-1"></i> Reset Data
            </button>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            <!-- Sidebar: Upload PDF -->
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700 h-fit">
                <h2 class="text-lg font-bold text-white mb-4"><i class="fa-solid fa-file-pdf text-red-500 mr-2"></i>Upload Documents</h2>
                
                <input type="file" id="pdf-input" accept=".pdf" multiple class="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 cursor-pointer mb-4" />
                
                <div class="space-y-3 text-xs text-gray-400 mb-4">
                    <div>
                        <label class="block font-medium mb-1">Chunk Size: <span id="chunk-size-val">1000</span></label>
                        <input type="range" id="chunk-size" min="500" max="2000" step="100" value="1000" oninput="document.getElementById('chunk-size-val').innerText=this.value" class="w-full accent-blue-500">
                    </div>
                    <div>
                        <label class="block font-medium mb-1">Chunk Overlap: <span id="chunk-overlap-val">200</span></label>
                        <input type="range" id="chunk-overlap" min="50" max="400" step="50" value="200" oninput="document.getElementById('chunk-overlap-val').innerText=this.value" class="w-full accent-blue-500">
                    </div>
                </div>

                <button onclick="uploadFiles()" id="upload-btn" class="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold rounded-lg transition shadow">
                    Process PDFs & Index
                </button>
                
                <div id="upload-msg" class="mt-3 text-xs"></div>
            </div>

            <!-- Main Chat Interface -->
            <div class="md:col-span-2 bg-gray-800 rounded-xl p-6 border border-gray-700 flex flex-col justify-between min-h-[480px]">
                <div id="chat-box" class="space-y-4 overflow-y-auto max-h-[420px] pr-2">
                    <div class="bg-gray-700/50 p-4 rounded-lg border border-gray-600">
                        <p class="text-sm text-gray-300">👋 Welcome to <b>DocuMind AI Direct Web Interface</b>! Upload your PDF documents on the left and ask questions to get grounded answers with source citations.</p>
                    </div>
                </div>

                <div class="mt-4 flex gap-2">
                    <input type="text" id="user-query" placeholder="Ask a question about your documents..." onkeydown="if(event.key==='Enter') sendQuery()" class="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-white focus:outline-none focus:border-blue-500" />
                    <button onclick="sendQuery()" id="send-btn" class="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg transition text-sm">
                        <i class="fa-solid fa-paper-plane mr-1"></i> Ask
                    </button>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function fetchStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('provider-name').innerText = data.provider_info.provider + ' (' + data.provider_info.model + ')';
                
                const dot = document.getElementById('status-dot');
                if (data.provider_info.status === 'Connected') {
                    dot.className = 'h-3 w-3 rounded-full bg-green-500';
                } else {
                    dot.className = 'h-3 w-3 rounded-full bg-yellow-500';
                }
                
                const idxStatus = document.getElementById('index-status');
                if (data.documents_processed) {
                    idxStatus.innerText = 'Ready (' + data.total_chunks + ' chunks, ' + data.file_names.length + ' files)';
                    idxStatus.className = 'font-bold text-green-400 ml-1';
                } else {
                    idxStatus.innerText = 'Empty (0 chunks)';
                    idxStatus.className = 'font-bold text-gray-400 ml-1';
                }
            } catch(e) { console.error(e); }
        }

        async function uploadFiles() {
            const files = document.getElementById('pdf-input').files;
            if(!files.length) { alert('Please select at least one PDF file.'); return; }
            
            const btn = document.getElementById('upload-btn');
            const msg = document.getElementById('upload-msg');
            btn.disabled = true;
            btn.innerText = 'Processing & Indexing...';
            msg.innerHTML = '<span class="text-blue-400">Extracting text & generating vector embeddings...</span>';

            const formData = new FormData();
            for(let i=0; i<files.length; i++) {
                formData.append('files', files[i]);
            }
            formData.append('chunk_size', document.getElementById('chunk-size').value);
            formData.append('chunk_overlap', document.getElementById('chunk-overlap').value);

            try {
                const res = await fetch('/api/upload', { method: 'POST', body: formData });
                const data = await res.json();
                if(res.ok) {
                    msg.innerHTML = `<span class="text-green-400">Indexed ${data.total_chunks} chunks successfully!</span>`;
                    fetchStatus();
                } else {
                    msg.innerHTML = `<span class="text-red-400">${data.detail || 'Upload failed'}</span>`;
                }
            } catch(e) {
                msg.innerHTML = `<span class="text-red-400">Error uploading files.</span>`;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Process PDFs & Index';
            }
        }

        async function sendQuery() {
            const input = document.getElementById('user-query');
            const q = input.value.trim();
            if(!q) return;

            const chatBox = document.getElementById('chat-box');
            chatBox.innerHTML += `
                <div class="bg-blue-900/40 border border-blue-700/50 p-3 rounded-lg text-sm text-blue-200">
                    <b class="text-blue-400">You:</b> ${q}
                </div>
            `;
            input.value = '';

            const loadingId = 'loading-' + Date.now();
            chatBox.innerHTML += `
                <div id="${loadingId}" class="bg-gray-700/40 border border-gray-600 p-3 rounded-lg text-sm text-gray-400 italic">
                    <i class="fa-solid fa-spinner fa-spin mr-2"></i>Searching document vectors & generating response...
                </div>
            `;
            chatBox.scrollTop = chatBox.scrollHeight;

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: q })
                });
                const data = await res.json();
                document.getElementById(loadingId).remove();

                let sourcesHtml = '';
                if(data.sources && data.sources.length) {
                    sourcesHtml = '<div class="mt-2 text-xs text-gray-400 border-t border-gray-600/50 pt-2 font-mono"><b>Sources:</b><ul class="list-disc pl-4 mt-1">';
                    data.sources.forEach((s, idx) => {
                        sourcesHtml += `<li>Source ${idx+1}: <code>${s.file_name}</code> (Page ${s.page_number}, Score: ${s.similarity_score.toFixed(3)})</li>`;
                    });
                    sourcesHtml += '</ul></div>';
                }

                chatBox.innerHTML += `
                    <div class="bg-gray-700/60 border border-gray-600 p-3.5 rounded-lg text-sm text-gray-200">
                        <b class="text-indigo-400 block mb-1">DocuMind AI:</b>
                        <div class="whitespace-pre-wrap">${data.answer}</div>
                        ${sourcesHtml}
                    </div>
                `;
            } catch(e) {
                document.getElementById(loadingId).remove();
                chatBox.innerHTML += `
                    <div class="bg-red-900/40 border border-red-700/50 p-3 rounded-lg text-sm text-red-300">
                        Error retrieving answer.
                    </div>
                `;
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        async function resetSystem() {
            if(!confirm('Reset vector database and clearing documents?')) return;
            await fetch('/api/reset', { method: 'POST' });
            document.getElementById('chat-box').innerHTML = `
                <div class="bg-gray-700/50 p-4 rounded-lg border border-gray-600 text-sm text-gray-300">
                    System state reset. Please upload new PDF files.
                </div>
            `;
            fetchStatus();
        }

        fetchStatus();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_TEMPLATE


@app.get("/api/status")
def get_status():
    provider_info = rag_engine.get_provider_info()
    return {
        "provider_info": provider_info,
        "documents_processed": vector_store.is_ready(),
        "total_chunks": len(vector_store.chunks),
        "file_names": (
            list(set(chunk["file_name"] for chunk in vector_store.chunks))
            if vector_store.is_ready()
            else []
        ),
    }


@app.post("/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    wrapped_files = []
    for f in files:
        content = await f.read()
        wrapped_files.append(
            UploadWrapper(filename=f.filename, content=content)
        )

    try:
        chunks, errors = process_uploaded_pdfs(
            uploaded_files=wrapped_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text found in uploaded PDFs.",
            )

        vector_store.build_index(chunks)
        return {
            "status": "success",
            "total_chunks": len(chunks),
            "files": [f.filename for f in files],
            "errors": errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
def query_documents(payload: QueryRequest):
    if not vector_store.is_ready():
        raise HTTPException(
            status_code=400, detail="Vector index is empty. Upload PDFs first."
        )

    q = payload.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    retrieved = vector_store.search(
        question=q, top_k=payload.top_k, minimum_score=0.15
    )
    answer = rag_engine.generate_answer(question=q, retrieved_chunks=retrieved)

    return {"question": q, "answer": answer, "sources": retrieved}


@app.post("/api/reset")
def reset_all():
    vector_store.index = None
    vector_store.chunks = []
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
