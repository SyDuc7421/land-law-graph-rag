from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    """Kiểm tra trạng thái server"""
    return {"message": "ok"}
