from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from base import *
import uvicorn
from typing import List
import shutil
app = FastAPI()

# Copy your face recognition functions here
@app.get("/known_face_names")
def get_known_face_names():
    id, all_face_features, known_face_names = load_data(data_path)
    return {"id": id, "name": known_face_names, 'feature': all_face_features}

@app.post("/recognize")
async def recognize_faces(file: UploadFile = File(...)):
    try:
        # Save the uploaded image to a temporary file
        img_path = "temp_image.jpg"
        with open(img_path, "wb") as image_file:
            image_file.write(file.file.read())

        # Perform face recognition on the uploaded image
        result = recognition(img_path)
        if (result == None):
            return JSONResponse(content={"No detect face"})
        else:
            # Return the result as JSON
            return JSONResponse(content={"result": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/recognize_folder/")
async def recognize_folder(folder_path: str = Form(...)):
    try:
        # Perform face recognition on the folder
        result = recognition_folder(folder_path)

        # Return the result
        return {"result": result}

    except Exception as e:
        # Return error message if an exception occurs
        return {"error": str(e)}

@app.post("/recognize_list")
async def recognize_faces(files: List[UploadFile] = File(...)):
    try:
        results = []

        for file in files:
            # Save each uploaded image to a temporary file
            img_path = f"temp_image_{files.index(file)}.jpg"
            with open(img_path, "wb") as image_file:
                image_file.write(file.file.read())

            # Perform face recognition on the uploaded image
            result = recognition(img_path)

            # Append the result to the list of results
            results.append({"filename": file.filename, "result": result})

        # Return the results as JSON
        return JSONResponse(content={"results": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   

@app.post("/add_person_by_folder_default/{folder_path}")
async def read_and_update_folder(folder_path: str):
    for person_folder in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person_folder)

        # Đọc và cập nhật dữ liệu từ thư mục
        person = add_person(person_path)
        add_user_to_json(data_path, person)

        
    return {"message": f"Data for {folder_path} updated successfully"}

@app.post("/add_person/{person_name}")
async def read_and_update_folder(person_name: str, files: List[UploadFile] = File(...)):
    # Đường dẫn tới thư mục của người
    person_folder_path = f"./person_data/{person_name}"
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(person_folder_path, exist_ok=True)
    # Lưu ảnh vào thư mục
    for file in files:
        with open(os.path.join(person_folder_path, file.filename), "wb") as image:
            shutil.copyfileobj(file.file, image)
    # Đọc và cập nhật dữ liệu từ thư mục
    person = add_person(person_folder_path)
    add_user_to_json(data_path, person)

    if os.path.isdir(person_folder_path):
        shutil.rmtree(person_folder_path)
    return {"message": f"Data for {person_name} updated successfully"}

@app.delete("/remove_person/{ID}")
async def remove_person(ID: str):
    try:
        remove_person_data(data_path,ID)
        return {"message": f"Data for id: {ID} removed successfully"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
if __name__ == "__main__":
    
    # Start the FastAPI app using Uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
