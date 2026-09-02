import os
import shutil
import numpy as np
from face_recognition import load_yunet, detect_faces

RESULTS_DIR = "./results/"
model_yn = load_yunet()

def start_analysis(source_dir,target_path):
    target_embedding = detect_faces(model_yn, target_path)[0]
    analyze_folder(source_dir, target_embedding)


def analyze(target_embedding, embedding):
    "We compute the embedding difference with the euclidean metric."
    diff = np.linalg.norm(target_embedding - embedding) / 2

    embedding = embedding.ravel()
    target_embedding = target_embedding.ravel()
    similarity = np.dot(embedding, target_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(target_embedding))
    print(diff)
    if diff < 3.0 or similarity > 0.65:
        return True

    return False

def analyze_folder(folder_path, target_embedding):
    print("Analyzing folder : ", folder_path)
    files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(".jpg"):
            files.append(f)
    files.sort()

    for i, file in enumerate(files, start=1):
        image_path = os.path.join(folder_path, file)
        embeddings = detect_faces(model_yn, image_path)

        for embedding in embeddings:
            if analyze(target_embedding, embedding):
                shutil.copy(image_path, RESULTS_DIR)
                break

    subfolders = [
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    for sub in subfolders:
        sub_path = os.path.join(folder_path, sub)
        analyze_folder(sub_path, target_embedding)

