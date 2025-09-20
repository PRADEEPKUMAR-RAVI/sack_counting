from fastapi import APIRouter, UploadFile, File
from app.services.inference_service import InferenceService

inference_router = APIRouter(tags=["Inference"])

inference_service = InferenceService()

@inference_router.get("/health", description="Health Check Endpoint")
def health_check():
    return{
        "status":"ok",
        "msg":"Backend is up and running"
    }


@inference_router.post("/video", description="Video Inference Endpoint")
def video_inference(file: UploadFile = File(...)):

    try:
        
        return inference_service.process_video(file=file)
    
    except Exception as e:
        return {"error": str(e)}