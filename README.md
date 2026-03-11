# Land Law GraphRAG 🇻🇳

Hệ thống Retrieval-Augmented Generation (RAG) dựa trên **Đồ thị Tri thức (Knowledge Graph)** dành riêng cho **Luật Đất đai 2024 của Việt Nam**. Dự án sử dụng mô hình trích xuất dữ liệu GraphRAG của Microsoft, lưu trữ đồ thị bằng **Neo4j**, và cung cấp API truy vấn thông minh thông qua **FastAPI** và **LangChain**.

## 🌟 Tính năng nổi bật

- **Trích xuất Đồ thị tự động**: Sử dụng pipeline của GraphRAG để đọc hiểu văn bản Luật Đất đai 2024 và tự động trích xuất các thực thể (Entity) cũng như mối quan hệ (Relationship) đặc thù của ngành luật (VD: `CHỦ_THỂ_QUẢN_LÝ`, `NGƯỜI_SỬ_DỤNG_ĐẤT`, `LOẠI_ĐẤT_CHI_TIẾT`...).
- **Cơ sở dữ liệu Đồ thị Neo4j**: Lưu trữ mạng lưới các điều luật, quy định phức tạp để cho phép LLM dễ dàng "dò đường" (traversal) và móc nối thông tin chính xác.
- **Dịch ngôn ngữ tự nhiên sang Cypher**: Sử dụng LLM do LangChain quản lý để tự động biên dịch câu hỏi tiếng Việt của người dùng thành câu truy vấn Cypher chuyên dụng cho Neo4j.
- **Kiến trúc API hiện đại**: Xây dựng bằng FastAPI chuẩn RESTful theo mô hình MVC thu gọn, cực nhanh và dễ mở rộng.
- **Hỗ trợ trích dẫn rõ ràng**: AI được thiết lập đặc biệt để hành xử như một chuyên gia tư vấn, luôn yêu cầu trích dẫn rõ ràng Chương, Điều khoản áp dụng.

## 🛠️ Công nghệ sử dụng

- **Python 3.10+**
- **Microsoft GraphRAG** (Trích xuất tri thức)
- **Neo4j** (Cơ sở dữ liệu Graph via Docker)
- **FastAPI / Uvicorn** (Web Framework Web Server)
- **LangChain / LangChain-Community** (Orchestration & Workflow)
- **OpenAI (gpt-4o-mini / text-embedding-3-small)** (LLMs)

---

## 🚀 Hướng dẫn cài đặt và chạy dự án

### 1. Yêu cầu môi trường
- Python >= 3.10
- Docker & Docker Compose (Để chạy Neo4j local)

### 2. Cài đặt các gói phụ thuộc
Tạo môi trường ảo (virtual environment) và cài đặt thư viện:

```bash
# Ưu tiên sử dụng môi trường ảo (venv hoặc conda)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường (`.env`)
Tạo một file `.env` ở thư mục gốc của dự án với nội dung như sau:
```env
OPEN_AI_API=sk-your-openai-api-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=my_secure_password123
```

### 4. Khởi động Neo4j Database
Mở Docker và chạy file `docker-compose.yml`:
```bash
docker-compose up -d
```
Bạn có thể truy cập giao diện quản trị Neo4j tại: http://localhost:7474

### 5. Nạp dữ liệu vào Đồ thị (Ingestion)
Sau khi quá trình `graphrag index` hoàn thành (dữ liệu mẫu đã được commit sẵn ở thư mục `output`), tiến hành chèn vector và nodes vào cơ sở dữ liệu Neo4j:
```bash
python ingest_neo4j.py
```

### 6. Khởi chạy API Server
```bash
uvicorn app.main:app --reload --port 8000
```
API Server sẽ chạy tại: http://127.0.0.1:8000

---

## 📚 API Endpoints

FastAPI tự động sinh tài liệu Swagger UI. Bạn vào trình duyệt truy cập: **http://127.0.0.1:8000/docs** để thử test API.

### `POST /chat`
Gửi tin nhắn để nhận câu trả lời tư vấn luật từ AI.

**Request Body:**
```json
{
  "query": "Ai là người giải quyết các tranh chấp về đất đai?",
  "top_k": 5
}
```

**Response Output:**
```json
{
  "answer": "Theo Khoản 1 Điều 236 của Luật Đất đai 2024, tranh chấp đất đai do Chủ tịch Ủy ban nhân dân cấp tỉnh, cấp huyện giải quyết...",
  "cypher_query": "MATCH (source:Entity)-[r:RELATED]-(target:Entity) WHERE ...",
  "context_data": [ ... ]
}
```

---

## 📁 Cấu trúc thư mục

```text
.
├── app/
│   ├── api/
│   │   ├── routes/          # Các API Endpoints (chat.py, health.py)
│   │   └── main.py          # Khai báo và gộp Routers
│   ├── core/
│   │   └── config.py        # Object thiết lập app, load .env
│   ├── db/
│   │   └── neo4j.py         # Hàm connection tới database
│   ├── schemas/
│   │   └── chat.py          # Data Models (Pydantic objects)
│   ├── services/
│   │   ├── chat_service.py  # Xử lý Logic Chat với mô hình Langchain
│   │   └── prompts.py       # Nơi lưu các System Prompts
│   └── main.py              # Root File khởi chạy FastAPI
├── input/                   # Dữ liệu txt đầu vào của đạo luật
├── output/                  # Kết quả xuất ra từ hệ thống GraphRAG Index
├── prompts/                 # System Prompts dùng cho GraphRAG Crawler
├── .env                     # File biến môi trường (Bỏ qua bởi git)
├── docker-compose.yml       # Cấu hình container Neo4j
├── ingest_neo4j.py          # Script migrate data từ folder output vào Neo4j
├── requirements.txt         # Liệt kê thư viện Python
└── settings.yaml            # Setting core của Microsoft GraphRAG Pipeline
```
