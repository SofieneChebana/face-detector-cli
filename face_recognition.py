import cv2
print(cv2.__version__)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    "",
    0
    )

def load_yunet():
    model = cv2.FaceDetectorYN.create(
        model="models/face_detection_yunet_2023mar.onnx",
        config="",
        input_size=(320, 320),
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=5000
    )
    return model

def resize(img, max_dim=1600):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    return img


def detect_faces(model, image_path):
    print(image_path)
    embeddings = []
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    model.setInputSize((w, h))
    faces = model.detect(img)

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = face[:4].astype(int)

            if x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                print("❌ Face off bound → ignored")
                continue

            if w < 40 or h < 40:
                print("❌ Face too small → ignored")
                continue

            aligned = recognizer.alignCrop(img, face)
            if aligned is None:
                print("❌ alignCrop a échoué")
                continue

            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            embedding = recognizer.feature(aligned).ravel()
            embeddings.append(embedding)
            print("Embedding 128D :", embedding.shape)


    return embeddings
