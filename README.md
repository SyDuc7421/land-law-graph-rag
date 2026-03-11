# Land Law GraphRAG 🇻🇳

Hệ thống **Retrieval-Augmented Generation (RAG)** dựa trên **Đồ thị Tri thức (Knowledge Graph)** dành cho **Luật Đất đai Việt Nam 2013 & 2024**. Hệ thống tự động phân tích, so sánh hai phiên bản luật, lưu trữ đồ thị quan hệ vào **Neo4j**, và cung cấp API truy vấn thông minh thông qua **FastAPI** và **LangChain**.

## 🌟 Tính năng nổi bật

- **Pipeline nạp dữ liệu hoàn chỉnh**: Tự động parse, chunk theo cấu trúc Chương – Mục – Điều – Khoản, so khớp điều khoản giữa 2 phiên bản (cross-mapping) và xây dựng Knowledge Graph trong Neo4j.
- **Cross-mapping thông minh**: Căn chỉnh các điều khoản tương ứng giữa 2013 và 2024, sinh `diff_summary` (tóm tắt thay đổi) bằng LLM cho từng mối quan hệ `SUPERSEDES`.
- **Hybrid Retrieval (KG + Vector)**: Mỗi loại truy vấn sử dụng chiến lược riêng — traversal đồ thị Cypher cho câu hỏi cấu trúc, semantic vector search (`text-embedding-3-small`) cho câu hỏi ngữ nghĩa, kết hợp fallback mở rộng token.
- **Phân loại truy vấn tự động**: LLM phân loại câu hỏi vào 4 loại (`so_sanh`, `tra_cuu`, `chu_the`, `thay_doi`) và trích xuất thực thể (số điều, năm luật, chủ thể pháp lý, từ khóa).
- **Suy luận pháp lý có trích dẫn**: Prompt chuyên biệt theo từng loại truy vấn, kết quả luôn kèm trích dẫn Điều – Khoản – Luật năm và đánh giá độ tin cậy.
- **Kiến trúc API hiện đại**: FastAPI theo mô hình MVC, tài liệu Swagger tự động.

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Web Framework | **FastAPI / Uvicorn** |
| Graph Database | **Neo4j** (via Docker) |
| Vector Index | **Neo4j Vector** (`article_embedding_index`, `clause_embedding_index`) |
| LLM | **OpenAI `gpt-4o-mini`** |
| Embeddings | **OpenAI `text-embedding-3-small`** (dim 1536) |
| Orchestration | **LangChain / LangChain-Community / LangChain-OpenAI** |
| Language | **Python 3.10+** |

---

## 🗺️ Kiến trúc Pipeline

```
Input (txt)
    │
    ▼
┌─────────────────────────────────────┐
│         INGESTION PIPELINE          │  ingest_law_kg.py
│                                     │
│  Phase 1: Parse + Chunk             │  parser.py → chunker.py
│    LandLaw2013.txt                  │  → chunks_2013.json
│    LandLaw2024.txt                  │  → chunks_2024.json
│                                     │
│  Phase 2: Cross-Mapping             │  cross_mapper.py
│    Article alignment 2013 ↔ 2024   │  → cross_mapping.json
│    LLM diff summary (optional)      │
│                                     │
│  Phase 3: Build Knowledge Graph     │  kg_builder.py
│    Nodes: Article, Clause, Concept  │  → Neo4j
│    Rels:  SUPERSEDES, HAS_CLAUSE,   │
│           DEFINES_CONCEPT           │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│           QUERY PIPELINE            │  chat_service.py
│                                     │
│  Step 1: Query Classifier           │  query_classifier.py
│    so_sanh / tra_cuu /              │  LLM + regex fallback
│    chu_the / thay_doi               │
│                                     │
│  Step 2: Hybrid Retriever           │  retriever.py
│    KG traversal (Cypher)            │
│    + Semantic vector search         │
│    + Broad keyword fallback         │
│                                     │
│  Step 3: Law Reasoner               │  reasoner.py
│    Type-specific prompt             │  gpt-4o-mini
│    → answer + citations             │
│    → confidence score               │
└─────────────────────────────────────┘
    │
    ▼
POST /chat  (FastAPI)
```

---

## 🚀 Hướng dẫn cài đặt và chạy

### 1. Yêu cầu môi trường
- Python >= 3.10
- Docker & Docker Compose

### 2. Cài đặt thư viện

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường (`.env`)

```env
OPENAI_API_KEY=sk-your-openai-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=my_secure_password
```

### 4. Khởi động Neo4j

```bash
docker-compose up -d
```

Giao diện quản trị Neo4j: http://localhost:7474

### 5. Nạp dữ liệu vào Knowledge Graph

```bash
# Full pipeline (parse + cross-map + build KG + embeddings + LLM diff)
python ingest_law_kg.py

# Tùy chọn:
python ingest_law_kg.py --skip-embeddings   # Bỏ qua bước tạo embeddings
python ingest_law_kg.py --skip-llm-diff     # Bỏ qua LLM diff summary
python ingest_law_kg.py --dry-run           # Chỉ parse + chunk + cross-map, không ghi Neo4j
python ingest_law_kg.py --load-cached       # Dùng lại chunks/cross-mapping đã có trong output/
python ingest_law_kg.py --no-clear          # Giữ nguyên dữ liệu Neo4j hiện có
```

> Dữ liệu đầu vào: `input/LandLaw2013.txt`, `input/LandLaw2024.txt`
> Dữ liệu trung gian: `output/chunks_2013.json`, `output/chunks_2024.json`, `output/cross_mapping.json`

### 6. Tạo Vector Index (nếu chưa có)

```bash
python -m app.scripts.create_vector_index
```

### 7. Khởi chạy API Server

```bash
uvicorn app.main:app --reload --port 8000
```

API Server: http://127.0.0.1:8000
Swagger UI: http://127.0.0.1:8000/docs

---

## 📚 API Endpoints

### `POST /chat`

Gửi câu hỏi pháp lý, nhận câu trả lời có trích dẫn từ AI.

**Request:**
```json
{
  "query": "UBND tỉnh có thẩm quyền gì trong việc thu hồi đất?",
  "top_k": 10
}
```

**Response:**
```json
{
  "answer": "Theo Khoản 2 Điều 79 Luật Đất đai 2024, UBND cấp tỉnh có thẩm quyền thu hồi đất...",
  "query_type": "chu_the",
  "intent_summary": "Thẩm quyền thu hồi đất của UBND tỉnh",
  "confidence": 0.87,
  "has_evidence": true,
  "citations": [
    {
      "law_year": 2024,
      "article_number": 79,
      "title": "Thẩm quyền thu hồi đất",
      "chapter_name": "THU HỒI ĐẤT, TRƯNG DỤNG ĐẤT",
      "source_type": "graph"
    }
  ],
  "cypher_queries": ["MATCH (a:Article {law_year: 2024}) ..."],
  "retrieval_stats": {
    "total_chunks": 8,
    "graph_chunks": 5,
    "vector_chunks": 3,
    "vector_used": true
  },
  "context_data": [ "..." ]
}
```

### `GET /health`

Kiểm tra trạng thái kết nối Neo4j và API server.

---

## 🔍 Chiến lược Retrieval theo loại truy vấn

| Loại | Mô tả | Chiến lược KG | Vector |
|---|---|---|---|
| `so_sanh` | So sánh điều khoản giữa 2013 và 2024 | `SUPERSEDES` relationship | Cả 2 phiên bản |
| `tra_cuu` | Tra cứu nội dung điều khoản | Số điều / keyword Cypher | Semantic fallback |
| `chu_the` | Quyền/nghĩa vụ/thẩm quyền chủ thể | `Clause` theo `clause_type` | Theo chủ thể |
| `thay_doi` | Thay đổi tổng quát Luật 2024 | Toàn bộ `SUPERSEDES` có `diff_summary` | Keyword mở rộng |

---

## 📁 Cấu trúc thư mục

```text
.
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── chat.py          # POST /chat endpoint
│   │   │   └── health.py        # GET /health endpoint
│   │   └── main.py              # Khai báo và gộp Routers
│   ├── core/
│   │   └── config.py            # Settings, load .env
│   ├── db/
│   │   ├── neo4j.py             # Neo4j connection
│   │   └── vector_store.py      # Neo4j Vector index wrapper
│   ├── law_processing/
│   │   ├── models.py            # Data models: LawDocument, Article, Clause, ...
│   │   ├── parser.py            # Parse txt → LawDocument hierarchy
│   │   ├── chunker.py           # Tạo Article chunks + lưu JSON
│   │   ├── cross_mapper.py      # Cross-mapping 2013 ↔ 2024 + LLM diff
│   │   └── kg_builder.py        # Build Knowledge Graph trong Neo4j
│   ├── schemas/
│   │   └── chat.py              # Pydantic models: ChatRequest, ChatResponse, CitationItem
│   ├── scripts/
│   │   └── create_vector_index.py  # Tạo Neo4j vector index
│   ├── services/
│   │   ├── chat_service.py      # RAG Pipeline Orchestrator (3 bước)
│   │   ├── query_classifier.py  # LLM query classifier (4 loại)
│   │   ├── retriever.py         # Hybrid retriever (KG + vector + fallback)
│   │   ├── reasoner.py          # LLM reasoner + citations + confidence
│   │   └── prompts.py           # System prompts
│   └── main.py                  # FastAPI app entry point
├── input/
│   ├── LandLaw2013.txt          # Văn bản Luật Đất đai 2013
│   └── LandLaw2024.txt          # Văn bản Luật Đất đai 2024
├── output/
│   ├── chunks_2013.json         # Chunks đã parse của 2013
│   ├── chunks_2024.json         # Chunks đã parse của 2024
│   └── cross_mapping.json       # Kết quả cross-mapping + diff
├── .env                         # Biến môi trường (không commit)
├── docker-compose.yml           # Neo4j container
├── ingest_law_kg.py             # Script nạp dữ liệu vào Neo4j KG
└── requirements.txt             # Python dependencies
```
