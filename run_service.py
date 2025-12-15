from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import uuid
import structlog
from contextlib import asynccontextmanager

from chatdev.chat_chain import ChatChain
from chatdev.settings import settings
from camel.typing import ModelType

logger = structlog.get_logger()

# Global state for running jobs
jobs = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ChatDev API Service")
    yield
    # Shutdown
    logger.info("Shutting down ChatDev API Service")

app = FastAPI(title="ChatDev API", version=settings.VERSION, lifespan=lifespan)

class TaskRequest(BaseModel):
    task_prompt: str
    project_name: str = "Project"
    org_name: str = "Org"
    model: str = "GPT_3_5_TURBO"

class TaskResponse(BaseModel):
    job_id: str
    status: str

async def run_chatdev_task(job_id: str, req: TaskRequest):
    try:
        jobs[job_id]["status"] = "running"

        # Map model string to ModelType
        model_type_map = {
            "GPT_3_5_TURBO": ModelType.GPT_3_5_TURBO,
            "GPT_4": ModelType.GPT_4,
            "GPT_4_TURBO": ModelType.GPT_4_TURBO,
            "GPT_4O": ModelType.GPT_4O,
            "GPT_4O_MINI": ModelType.GPT_4O_MINI,
        }
        model_type = model_type_map.get(req.model, ModelType.GPT_3_5_TURBO)

        # Initialize Chain
        # Note: We rely on default configs for simplicity in API for now
        # In a real prod scenario, we might accept config overrides in the request
        config_path, config_phase_path, config_role_path = (
            "CompanyConfig/Default/ChatChainConfig.json",
            "CompanyConfig/Default/PhaseConfig.json",
            "CompanyConfig/Default/RoleConfig.json"
        )

        chain = ChatChain(
            config_path=config_path,
            config_phase_path=config_phase_path,
            config_role_path=config_role_path,
            task_prompt=req.task_prompt,
            project_name=req.project_name,
            org_name=req.org_name,
            model_type=model_type
        )

        # Disable human interaction mode for API execution to prevent blocking
        # Ideally this would be handled via configuration injection
        # chain.config["chain"] = [p for p in chain.config["chain"] if "Human" not in p["phase"]]

        await chain.pre_processing()
        chain.make_recruitment()
        await chain.execute_chain()
        chain.post_processing()

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result_path"] = chain.log_filepath

    except Exception as e:
        logger.error("Job failed", job_id=job_id, error=str(e))
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)

@app.post("/tasks", response_model=TaskResponse)
async def create_task(req: TaskRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "id": job_id,
        "status": "pending",
        "request": req.model_dump()
    }

    background_tasks.add_task(run_chatdev_task, job_id, req)

    return TaskResponse(job_id=job_id, status="pending")

@app.get("/tasks/{job_id}")
async def get_task_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": settings.VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
