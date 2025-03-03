import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Initialize Face Detector (MTCNN) and FaceNet Model
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
similar=False

def preprocess_image(image_path):
    """ Load an image and detect the face, returning its embedding. """
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)

    if face is None:
        raise ValueError(f"No face detected in {image_path}")

    face_embedding = resnet(face.unsqueeze(0))  # Get embedding
    return face_embedding

def preprocess_imageArray(image_array):
    """
    Convert an OpenCV (NumPy) image to a PIL Image, detect face, and get the embedding.
    """
    img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))  # Convert NumPy to PIL
    face = mtcnn(img)

    if face is None:
        raise ValueError("No face detected in the image")

    face_embedding = resnet(face.unsqueeze(0))  # Get embedding
    return face_embedding

def compare_faces(inputImageArray, refImage, threshold=0.5):#array,ref image path
    try:
        """ Compare two face images based on embeddings. """
        emb1 = preprocess_imageArray(inputImageArray)
        emb2 = preprocess_image(refImage)

        # Compute Euclidean Distance
        distance = torch.norm(emb1 - emb2).item()

        # Alternative: Cosine Similarity
        # cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()

        print(f"Euclidean Distance: {distance:.4f}")
        # print(f"Cosine Similarity: {cos_sim:.4f}")

        if distance < threshold:
            global similar
            similar =True
        # else:
        #     print("Faces are NOT similar.")
    except Exception as e:
        print(e)


# Example Usage
# compare_faces("image1.jpg", "image2.jpg")

#pip install facenet-pytorch torch torchvision numpy opencv-python
