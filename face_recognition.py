import cv2
print(cv2.__version__)

recognizer = cv2.FaceRecognizerSF.create(
    "models/face_recognition_sface_2021dec.onnx",
    "",
    0
    )

def load_yunet():
    # Charger YuNet
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
    # Charger image
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Adapter la taille d’entrée
    model.setInputSize((w, h))

    # Détection
    faces = model.detect(img)

    if faces[1] is not None:
        for face in faces[1]:
            x, y, w, h = face[:4].astype(int)

            # Vérification des limites
            if x < 0 or y < 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                print("❌ Face hors limites → ignorée")
                continue

            # Vérification taille visage
            if w < 40 or h < 40:
                print("❌ Visage trop petit → ignoré")
                continue
            
            
            #if w < 112 or h < 112:
            #    img = cv2.resize(img, None, fx=2, fy=2)
            #    x, y ,w, h = x*2, y*2, w*2, h*2

            aligned = recognizer.alignCrop(img, face)
            if aligned is None:
                print("❌ alignCrop a échoué")
                continue

            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

            embedding = recognizer.feature(aligned).ravel()
            embeddings.append(embedding)
            print("Embedding 128D :", embedding.shape)

    #cv2.imshow("YuNet", img)
    #cv2.waitKey(0)

    return embeddings

#model = load_yunet()
#detect_faces(model,"target.jpg")