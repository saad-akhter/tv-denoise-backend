from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
import numpy as np
from PIL import Image
from skimage.restoration import denoise_tv_chambolle
from skimage import img_as_float, img_as_ubyte

app = FastAPI(title="TV Image Denoising API")

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 8 * 1024 * 1024  # 8 MB


def read_image(file: UploadFile):
    try:
        img_bytes = file.file.read()
        if len(img_bytes) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")


@app.post("/denoise")
async def denoise_image(file: UploadFile = File(...), weight: float = Form(0.12)):
    # read image
    img = read_image(file)

    # convert to numpy
    arr = np.asarray(img)
    arr_float = img_as_float(arr)

    # TV denoise each channel
    denoised = np.zeros_like(arr_float)
    for c in range(3):
        denoised[..., c] = denoise_tv_chambolle(arr_float[..., c], weight=weight)

    # convert back to uint8
    denoised_uint8 = img_as_ubyte(denoised)
    output_img = Image.fromarray(denoised_uint8)

    # return as PNG
    buf = BytesIO()
    output_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
