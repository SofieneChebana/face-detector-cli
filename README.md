# face-detector-cli
A command-line tool that detects a target face in a reference image and finds matching photos across a folder of images using OpenCV (YuNet + SFace).

<h2>How to use</h2>
Install requirements (Numpy and OpenCV) :

``
pip install -r requirements.txt
``
<br><br>
Lauch the tool :

```
python3 main.py -i image.jpg -f folder/
```
Where 
- `-i` / `image.jpg`: Reference image containing the face of the person you are looking for.
- `-f` / `folder/`: Directory of images to search for matching faces.
<h2>Exemple</h2>

  <h3>Input:</h3>
  <img width="363" height="517" alt="pexels-kampus-8777825" src="https://github.com/user-attachments/assets/9588734a-00ae-4003-a8b7-9b1dcadf5fef" />
  
  <h3>Output:</h3>
  <div style="display:flex; gap:20px;">
    <img width="300" alt="pexels-kampus-8777820" src="https://github.com/user-attachments/assets/a7086fe5-a27e-416c-9db9-0c274eff79c9" />
    <img width="300" alt="pexels-kampus-8777826" src="https://github.com/user-attachments/assets/bc6a1ead-22b9-4741-a23e-638fb646c018" />
    <img width="300" alt="pexels-kampus-8777844" src="https://github.com/user-attachments/assets/81d6826b-3eb3-4f34-9284-1f551f97cd2f" />
  </div>
  
<h2>References & Credits</h2>


This project uses models from **OpenCV Zoo** (`opencv/opencv_zoo`) :
> W. Wu, H. Peng, and S. Yu, *"Yunet: A tiny millisecond-level face detector,"* *Machine Intelligence Research*, vol. 20, no. 5, pp. 656–665, 2023.

```bibtex
@article{wu2023yunet,
  title={Yunet: A tiny millisecond-level face detector},
  author={Wu, Wei and Peng, Hanyang and Yu, Shiqi},
  journal={Machine Intelligence Research},
  volume={20},
  number={5},
  pages={656--665},
  year={2023},
  publisher={Springer}
}
```
- **Face Detection (YuNet)**: `face_detection_yunet` via [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT License, by Shiqi Yu).
- **Face Recognition (SFace)**: `face_recognition_sface_2021dec.onnx` via [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) (Apache 2.0 License, based on the work on SFace by Yaoyao Zhong).
