from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
import base64
from PIL import Image
import io
from infer import infer

app = FastAPI()

class RequestData(BaseModel):
    img_data: str 
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    with open("templates/dashboard.html", "r", encoding="utf-8") as f:
        return f.read()
@app.post("/predict/")
async def predict(req: RequestData):
    # Decode base64 to image
    img_bytes = io.BytesIO(base64.b64decode(req.img_data))
    img = Image.open(img_bytes)

    # Run prediction
    pred = infer(img)

    # ---- Save the image ----
    import os
    from datetime import datetime

    save_dir = "X-Ray-Classification\data\saved_images"
    os.makedirs(save_dir, exist_ok=True)

    # اسم ملف فريد حسب الوقت
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pred}.png"
    save_path = os.path.join(save_dir, filename)

    # حفظ الصورة
    img.save(save_path)

    # Return prediction + save path
    return {
        "prediction": pred,
        "saved_path": save_path
    }

