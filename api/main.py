import uvicorn
from fastapi import FastAPI, File, UploadFile
from api import prediction



app = FastAPI()

@app.post('/api/predictImage')
async def predict_image(file:UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    # Read the file
    image = prediction.read_image(await file.read())
    # Do preprocessing
    image = prediction.preprocess(image, prediction.image_size)
    # Predict
    result = prediction.predict(image)
    print(result)

    return result


if __name__ =="__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000

    )