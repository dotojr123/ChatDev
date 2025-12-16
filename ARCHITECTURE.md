# Architecture Documentation (Dec 2025)

## Overview
This document describes the modernized architecture of ChatDev, refactored for production readiness, scalability, and maintainability.

## Core Components

### 1. Asynchronous Event Loop
The core orchestration engine (`ChatChain`) and agent interaction (`RolePlaying`) have been migrated to an **AsyncIO** model. This allows for:
- Non-blocking I/O operations (LLM API calls, Vector DB operations).
- Higher concurrency potential (future support for parallel agents).
- Better integration with modern web frameworks (FastAPI).

### 2. Memory System (Vector Store)
We replaced the legacy JSON-based memory with a proper **Vector Store** interface.
- **Interface**: `chatdev.memory.base.VectorStore` (Abstract Base Class).
- **Implementation**: `QdrantVectorStore` using `qdrant-client` (Async).
- **Data Model**: Pydantic models `MemoryItem` ensure type safety.
- **Storage**: Qdrant (supports both local `:memory:`/disk and server mode).

### 3. Configuration Management
Configuration is now handled via **Pydantic Settings** (`chatdev.settings.Settings`).
- Reads from Environment Variables and `.env` files.
- Provides type validation and default values.
- Centralized configuration for API keys, DB URLs, and feature flags.

### 4. API Service
A new **FastAPI** service (`run_service.py`) exposes ChatDev as a REST API.
- `POST /tasks`: Trigger a new software generation task (async background job).
- `GET /tasks/{job_id}`: Check status of a job.
- `GET /health`: Health check.

### 5. Infrastructure
- **Docker**: Optimized Multi-stage build (distroless/slim).
- **Docker Compose**: Orchestrates ChatDev API + Qdrant Vector DB.
- **CI/CD**: GitHub Actions for automated testing and linting.

## Directory Structure
```
.
├── chatdev/
│   ├── memory/          # New Memory Module
│   │   ├── base.py      # Interfaces
│   │   └── qdrant.py    # Qdrant Implementation
│   ├── settings.py      # Pydantic Settings
│   ├── chat_chain.py    # Async Chain Orchestrator
│   ├── phase.py         # Async Phases
│   └── ...
├── tests/               # New Test Suite
├── run_service.py       # FastAPI Entrypoint
├── run.py               # CLI Entrypoint (Async wrapper)
├── Dockerfile           # Multi-stage Dockerfile
└── docker-compose.yml   # Dev Environment
```

## Running the Project

### CLI Mode
```bash
python run.py --task "Create a snake game" --name "Snake"
```

### API Service Mode
```bash
python run_service.py
```

### Docker
```bash
docker-compose up --build
```
