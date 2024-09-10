import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from detect_food import predict_food

app = FastAPI()

# Load the MobileNetV2 model globally
model = MobileNetV2(weights='imagenet')

# CORS Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/detect_food')
async def detect_food(file: UploadFile = File(None)):
    if file is None:
        return {"status": "failure", "message": "No image detected."}
    # Read the uploaded image
    image_data = await file.read()
    img = Image.open(io.BytesIO(image_data)).resize((224, 224))
    # Make a prediction
    predictions = predict_food(img, model)
    return {"predictions": predictions}

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Detection API"}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)