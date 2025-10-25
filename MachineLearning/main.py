from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import voice_register, voice_verify, train_model

app = FastAPI(title="Voice Authentication Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(voice_register.router, prefix="/voice", tags=["register"])
app.include_router(voice_verify.router, prefix="/voice", tags=["verify"])
app.include_router(train_model.router, prefix="/model", tags=["training"])
# app.include_router(model_manage.router, prefix="/model", tags=["model"])

@app.get("/")
async def root():
    return {"message": "Voice Authentication Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)