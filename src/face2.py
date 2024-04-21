import dlib
import cv2
import numpy as np

# Load the pre-trained facial landmark predictor
predictor_path = "src/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Load the face detector
detector = dlib.get_frontal_face_detector()

# Function to detect facial landmarks and reconstruct the face
def detect_and_reconstruct_face(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(image)
    
    for face in faces:
        # Detect facial landmarks
        landmarks = predictor(image, face)
        
        # Reconstruct the face using the detected landmarks
        points = np.zeros((68, 2), dtype=int)
        for i in range(68):
            points[i] = (landmarks.part(i).x, landmarks.part(i).y)
        
        # Draw facial landmarks on the image
        for (x, y) in points:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    
    return image

def run_construct(image_path):
    # image = cv2.imread(image_path)
    reconstructed_image = detect_and_reconstruct_face(image_path)
    cv2.imshow("Facial Landmark Detection and Reconstruction", reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
