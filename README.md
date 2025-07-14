# DocReader: AI-Powered Document Extraction

DocReader is a robust, production-ready application designed to ingest invoices, waybills, and customs documents, and extract key business fields with high accuracy using state-of-the-art machine learning models.

## Features

*   **Document Ingestion**: Securely upload PDF or image documents.
*   **AI-Powered Extraction**: Utilizes advanced models like Donut and LayoutLMv3 for accurate field extraction.
*   **Scalable Architecture**: Built with FastAPI, Celery, Redis, and SQLite (for development) or PostgreSQL (for production) for high throughput and low latency.
*   **MLOps Ready**: Tracks model versions, scores, and provides observability with Prometheus/Grafana and OpenTelemetry.
*   **Containerized Deployment**: Docker and Helm charts for easy deployment on Kubernetes with NVIDIA GPU support.

## System Requirements

### Development Environment

*   **OS**: Linux (Fedora 39/40 recommended)
*   **GPU**: NVIDIA RTX-4070 Ti (12 GiB VRAM) or similar
*   **Software**:
    *   Python 3.10+
    *   CUDA 12.4+
    *   cuDNN
    *   gcc toolchain
    *   Poppler utilities
    *   ImageMagick

### Production Environment

*   **OS**: Linux
*   **GPU**: NVIDIA A4000 (20 GiB VRAM) or similar
*   **Software**:
    *   Docker
    *   Kubernetes with NVIDIA GPU Operator

## Quick Start (Development)

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/docreader.git
    cd docreader
    ```

2.  **Install system dependencies**:
    Ensure you have Python 3.10+, CUDA 12.4+, cuDNN, gcc, Poppler, and ImageMagick installed. Refer to your OS documentation for specific commands. For Fedora, see the initial setup instructions provided by the agent.

3.  **Create and activate a Conda environment (recommended)**:
    ```bash
    conda env create -f environment.yml
    conda activate docreader
    ```
    Alternatively, use `pip`:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Copy `.env.example` to `.env` and fill in your configurations.
    ```bash
    cp .env.example .env
    # Edit .env with your database connection string, Redis URL, etc.
    ```

5.  **Initialize the database (SQLite for development)**:
    ```bash
    alembic revision --autogenerate -m "Initial database setup (SQLite)"
    alembic upgrade head
    ```
    This will create a `docreader.db` file in your project root.

6.  **Run the application**:
    First, ensure **Redis** is running. You can run it via Docker:
    ```bash
    docker run --name docreader-redis -p 6379:6379 -d redis/redis-stack-server:latest
    ```
    **If you cannot use Docker due to permissions, you will need to install and run Redis manually on your system.**

    Then, start the FastAPI application:
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    And in a separate terminal, start the Celery worker. **Crucially, for GPU tasks, you need to set `CUDA_VISIBLE_DEVICES` if you have multiple GPUs and want to pin the worker to a specific one (e.g., GPU 0).**
    ```bash
    CUDA_VISIBLE_DEVICES=0 celery -A worker.tasks worker --loglevel=info --pool=solo # For development, use --pool=solo
    ```
    For production, you would use a different pool (e.g., `prefork` or `processes`) and manage `CUDA_VISIBLE_DEVICES` via your orchestration (e.g., Kubernetes).

## Sample cURL Calls

### Upload Document

```bash
curl -X POST "http://localhost:8000/upload" \
     -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     -F "file=@/path/to/your/document.pdf"
```

### Check Status

```bash
curl -X GET "http://localhost:8000/status/{task_id}" \
     -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Get Result

```bash
curl -X GET "http://localhost:8000/result/{document_id}" \
     -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Load Testing with Locust

1.  **Ensure the application (FastAPI and Celery worker) is running.**
2.  **Create a dummy PDF file** named `dummy.pdf` inside `tests/loadtest_locust/`. A simple way to create one is using `reportlab` (install with `pip install reportlab`) or any PDF creator.
3.  **Run Locust** from the project root:
    ```bash
    locust -f tests/loadtest_locust/locustfile.py
    ```
4.  **Open your browser** and navigate to `http://localhost:8089` (Locust UI).
5.  **Enter the number of users** to simulate and the **spawn rate** (users/second).
6.  **Set the host** (e.g., `http://localhost:8000`).
7.  **Start swarming!**

    You can also set the host and authentication token via environment variables:
    ```bash
    LOCUST_HOST=http://localhost:8000 LOCUST_AUTH_TOKEN="Bearer your_jwt_token_here" locust -f tests/loadtest_locust/locustfile.py
    ```

## Project Structure

```
.
├── app/                      # FastAPI application
│   ├── __init__.py
│   └── main.py               # Main FastAPI app
├── pipeline/                 # Document processing pipeline
│   ├── __init__.py
│   ├── prepare.py            # Pre-processing module (pdf2image, OpenCV)
│   ├── models/               # ML model wrappers
│   │   ├── __init__.py
│   │   ├── donut.py          # Donut model for end-to-end extraction
│   │   └── layoutlm_invoice.py # LayoutLMv3 inference
│   └── post.py               # Config-driven field post-processor
├── db/                       # Database layer
│   ├── __init__.py
│   ├── database.py           # Database session and engine
│   ├── models.py             # SQLModel definitions
│   └── migrations/           # Alembic migration scripts
├── worker/                   # Celery worker stack
│   ├── __init__.py
│   ├── celery_app.py         # Celery application setup
│   └── tasks.py              # Celery tasks
├── notebooks/                # Jupyter notebooks for training and experimentation
├── deploy/                   # Deployment configurations
│   └── helm/                 # Helm chart for Kubernetes deployment
├── tests/                    # Test suite
│   ├── loadtest_locust/      # Locust load testing scripts
│   │   └── locustfile.py
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore file
├── Dockerfile                # Dockerfile for dev and prod builds
├── LICENSE                   # Project license
├── README.md                 # Project README
├── environment.yml           # Conda environment file
└── requirements.txt          # Pip requirements file
```

## Contributing

*(Details to be added later)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
