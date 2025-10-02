import os
import asyncio
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import torch
import import_ipynb
from inference import infer, inference_transforms, Model, save_mask_as_dicom

app = FastAPI()

# === загрузка модели ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(torch.load("model2.pth", map_location=device))
model.eval()

# === очередь задач ===
task_queue = asyncio.Queue()

async def worker():
    while True:
        task = await task_queue.get()
        try:
            result = await run_inference(task["input_path"])
            task["future"].set_result(result)
        except Exception as e:
            task["future"].set_exception(e)
        finally:
            task_queue.task_done()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())

# === функция инференса ===
async def run_inference(input_path: str):
    # запустить модель
    outs = infer(model, input_path, inference_transforms)

    results = []
    for i, out in enumerate(outs):
        seg = out["segmentation"]

        seg_path = input_path.replace(".dcm", f"_seg_{i}.dcm")
        save_mask_as_dicom(
            seg, seg_path,
            study_uid=out.get("study_uid"),
            series_uid=out.get("series_uid"),
        )

        # добавляем путь к сохранённому сегу
        out["segmentation_file"] = seg_path
        # саму маску в JSON отдавать не будем (слишком жирная)
        out.pop("segmentation", None)

        results.append(out)

    return results

# === эндпоинты ===
@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    tmp_id = str(uuid4())
    os.makedirs("tmp", exist_ok=True)
    input_path = f"tmp/{tmp_id}.dcm"

    with open(input_path, "wb") as f:
        f.write(await file.read())

    if not input_path.lower().endswith(".dcm"):
        return JSONResponse(status_code=400, content={"error": "Только DICOM файлы!"})

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await task_queue.put({"input_path": input_path, "future": future})

    results = await future
    return JSONResponse(content={"results": results})
