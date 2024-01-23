import face_recognition
import cv2
import os
import numpy as np
import json
import random
#load đặc trưng đã lưu từ pkl or sql
def load_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
    except:
        # Handle the case where the file doesn't exist yet
        data = []
    # Retrieve the variables from the loaded data
    all_face_features = [dt["feature"] for dt in data]
    known_face_names = [dt["name"] for dt in data]
    id = [dt["id"] for dt in data]
    return id, all_face_features, known_face_names

data_path = "./label/features.json"
detect = 0

def add_person(person_name):
    face_features_list = []
    id, all_face_features, known_face_names = load_data(data_path)

    for filename in os.listdir(person_name):
        if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(person_name, filename)
            img = face_recognition.load_image_file(img_path)
            # Kích thước ban đầu của hình ảnh
            original_height, original_width = img.shape[:2]

            # Tính toán kích thước mới (1/2 kích thước ban đầu)
            new_width = int(original_width / 4)
            new_height = int(original_height / 4)

            # Thay đổi kích thước hình ảnh
            img = cv2.resize(img, (new_width, new_height))
            face_encodings = face_recognition.face_encodings(img)
            
            if len(face_encodings) > 0:
                face_features_list.append(face_encodings[0])

    if len(face_features_list) > 0:
        aggregated_features = np.mean(face_features_list, axis=0).tolist()
        all_face_features.append(aggregated_features)
        known_face_names.append(os.path.basename(person_name))

        new_user = {
            # dùng uuid 
            # import uuid

            # # Tạo một UUID mới
            # my_uuid = uuid.uuid4()

            # # In ra giá trị UUID
            # print("New UUID:", my_uuid)
            'id': id[-1] + 1 if len(id) > 0 else 1,
            'name': os.path.basename(person_name),
            'feature': aggregated_features
        }
    return new_user

def add_user_to_json(data_path, new_user):
    try:
        with open(data_path, 'r') as file:
            data = json.load(file)
    except:
        # Handle the case where the file doesn't exist yet
        data = []

    data.append(new_user)

    with open(data_path, 'w') as file:
        json.dump(data, file)

def recognition(img_path):
    id, all_face_features, known_face_names = load_data(data_path)

    rgb_img = face_recognition.load_image_file(img_path)
    # Neu co yolo cat rgb_img = rgb_img[x, y, h, w]
    # Kích thước ban đầu của hình ảnh
    original_height, original_width = rgb_img.shape[:2]

    # Tính toán kích thước mới (1/2 kích thước ban đầu)
    new_width = int(original_width / 4)
    new_height = int(original_height / 4)

    # Thay đổi kích thước hình ảnh
    rgb_img = cv2.resize(rgb_img, (new_width, new_height))
    rgb_img = np.ascontiguousarray(rgb_img[:,:,::-1])
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
    # face_encodings = face_recognition.face_encodings(rgb_img)
    face_names = []
    face_ids = []
    face_confidence = []
    if len(face_encodings) > 0 :
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(all_face_features, face_encoding)
            min_distance_index = np.argmin(face_distances)
            # print(face_distances[min_distance_index])
            if face_distances[min_distance_index] < 0.35 :
                confidences = round((1 - face_distances[min_distance_index]) ,2)
                face_confidence.append(confidences)
                face_names.append(known_face_names[min_distance_index])
                face_ids.append(id[min_distance_index])
            else:
                face_names.append("Unknown")
                face_ids.append(-1)
                face_confidence.append(0)

        # Luu anh neu tim thay guong mat
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Vẽ hộp đường viền khuôn mặt
                cv2.rectangle(rgb_img, (left, top), (right, bottom), (0, 0, 255), 2)
                # Hiển thị tên khuôn mặt
                cv2.putText(rgb_img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        output_image_path = f"./detect_history/detect{random.randint(1,1000)}.jpg"
        cv2.imwrite(output_image_path, rgb_img)
        
        return [face_names,face_confidence]
        # return [face_ids,face_confidence]

        
    else:
        return None

def recognition_folder(folder_path):
    folder_result = {}
    id, all_face_features, known_face_names = load_data(data_path)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPG") or filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            folder_result[filename] = recognition(img_path)

    return folder_result

def remove_person_data(data_path, person_id):
    # id, all_face_features, known_face_names = load_data(data_path)
    try:
        with open(data_path, 'r') as file:
            data = json.load(file)
    except:
        # Handle the case where the file doesn't exist yet
        data = []
    # Find the index of the person with the given id
    for person in data:
        if (person["id"] == int(person_id)):
            data.remove(person)
            print(f"Removed person with id {person_id}: {person}")
            with open(data_path, 'w') as file:
                json.dump(data, file)
            return person
    return None

    