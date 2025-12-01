# TV Image Denoising Backend (FastAPI)

This backend exposes a `/denoise` endpoint where users upload an image and receive a denoised image using Total Variation (TV) filtering.

### Run locally:

pip install -r requirements.txt
uvicorn app:app --reload --port 8000

### Endpoint:

POST /denoise
Form data:

file: image

weight: float (0.0–1.0)
