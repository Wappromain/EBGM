from face import *
from utils import *
from ebgm import *
from pca import *


if __name__ == "__main__":
    # sample data

    # probe_image = ImageLoader.load_image("00001_930831_fa_a.ppm")
    probe_image = process_image(input_image_path="src/rough/grayscale/00002qr010_940928.tif")

    gallery_image = "src/rough/grayscale"

    # Create EBGMFaceRecognition instance with gallery images from "input_images" folder
    ebgm_face_recognition = EBGMFaceRecognition(gallery_image)

    # graphs and jets for each folder should be assigned to unique gallery_image folder
    directory = "src/output_images/tests"
    file_names = extract_filenames_from_folder(folder_path=gallery_image)
    # Recognize face
    match_percentages = ebgm_face_recognition.recognize_face(probe_image, output_csv=f"{directory}/csv/result.csv", file_names=file_names)

    # Save graphs and jets
    ebgm_face_recognition.save_graphs(ebgm_face_recognition.gallery_graphs, directory, file_names=file_names)
    ebgm_face_recognition.save_jets(ebgm_face_recognition.gallery_jets, directory, file_names=file_names)

    # Print match percentages
    print(type(match_percentages))
    for i, percentage in enumerate(match_percentages):
        print(f"Match percentage with image {file_names[i]}: {percentage:.2f}%")

    
    # Load an image
    image = cv2.imread('src/rough/grayscale/00002qr010_940928.tif')
