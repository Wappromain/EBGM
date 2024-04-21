import numpy as np
import cv2
from sklearn.decomposition import PCA, IncrementalPCA
from ebgm import *

from face import *
from face2 import detect_and_reconstruct_face, run_construct

def pca_dimension_reduction(image_data, n_components=150):
    gray_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
    print(gray_image.shape)
    image_sum = gray_image.sum(axis=2)
    image_bw = image_sum / image_sum.max()
    print(image_bw.shape)
    # Perform PCA
    ipca = IncrementalPCA(n_components=n_components)
    reconstructed_image = ipca.inverse_transform(ipca.fit_transform(image_bw))

    # Create a FacialLandmarks object and detect landmarks
    run_construct(gray_image)

    return reconstructed_image

def process_image(input_image_path):
    # Read the input image
    image = cv2.imread(input_image_path)

    # Perform dimension reduction using PCA
    reconstructed_image = pca_dimension_reduction(image)
    return reconstructed_image
