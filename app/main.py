from fastapi import FastAPI
from app.routers.inference_router import inference_router

app = FastAPI(title="My API", description="This is a Learning Models API")



app.include_router(inference_router)


