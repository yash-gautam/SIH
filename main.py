import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from typing import Optional, Any, List
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
import uvicorn as uvicorn
# import starlette.responses as _responses

from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploadedFile: UploadFile


# @app.get('/')
# async def root():
#     return _responses.RedirectResponse("/docs")

@app.post("/files")
async def create_file(file: bytes = File(...)):
    global uploadedFile
    uploadedFile = file
    return {"file_size": len(file)}


@app.get("/predict")
async def predictDep():
    global uploadedFile
    model = load_model('SIH_Depression_Detection.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    path = r'Internal storage\Download'
    file = r'\face.png'
    # loc = os.path.abspath(r'C:\Users\visha\Downloads\S037_006_00000019.png')
    loc = path + file

    # img = image.load_img(uploadedFile.filename, target_size=(64, 64))
    img = image.load_img(loc, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    ans = classes[0]
    print(classes)
    j = 0
    arr = ['anger', 'contempt', 'digust', 'fear', 'happy', 'sadness', 'surprise']
    for i in ans:
        # print(int(i))
        x = int(i)
        if x == 1:
            print(j)
            break
        j = j + 1
    print(arr[j])
    return arr[j]
    # return {"file uploaded: ": uploadedFile.filename}


@app.post("/uploadfile")
async def create_upload_file(file: UploadFile):
    global uploadedFile
    uploadedFile = file
    return {"filename": file.filename}

@app.post("/uploadfileTest")
async def create_upload_file_test(image: UploadFile = File(...)):
    global uploadedFile
    print(image.file)
    try:
        # os.mkdir("images")
        print(os.getcwd())
    except Exception as e:
        print(e)
    dir_name = os.getcwd() + "/images/"
    image.save("sample.png", "")
    file_name = os.getcwd() + "/images/" + image.filename.replace(" ", "-")
    file = jsonable_encoder({"imagePath": file_name})
    print(file_name)
    # new_image = await add_image(file)
    # return {"filename": new_image}
    uploadedFile = image
    return {"filename": image.filename}


if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=5000)
