import sys
from face import *
from utils import *
from ebgm import *
from pca import *


if __name__ == "__main__":
    # sample data

    # probe_image = ImageLoader.load_image("00001_930831_fa_a.ppm")
    probe_image = process_image(input_image_path=sys.argv[2])

    gallery_image = sys.argv[1]

    # Create EBGMFaceRecognition instance with gallery images from "input_images" folder
    ebgm_face_recognition = EBGMFaceRecognition(gallery_image)

    # file names in folder
    file_names = extract_filenames_from_folder(folder_path=gallery_image)
    # Recognize face
    match_percentages = ebgm_face_recognition.recognize_face(
        probe_image,
        output_csv="src/output_images/main/csv/result.csv",
        file_names=file_names
    )

    # graphs and jets for each folder should be assigned to unique gallery_image folder
    directory = "src/output_images/main/"

    # Save graphs and jets
    ebgm_face_recognition.save_graphs(ebgm_face_recognition.gallery_graphs, directory, file_names=file_names)
    ebgm_face_recognition.save_jets(ebgm_face_recognition.gallery_jets, directory, file_names=file_names)

    similarity_ratio = ebgm_face_recognition.compute_similarity_ratio(ebgm_face_recognition.gallery_graphs, ebgm_face_recognition.gallery_jets, probe_image)
    print(similarity_ratio)
    # ebgm_face_recognition.compute_similarity_score()

    # Print match percentages
    print(type(match_percentages))
    for i, percentage in enumerate(match_percentages):
        print(f"Match percentage with image {file_names[i]}: {percentage:.2f}%")

    
    # Load an image
    image = cv2.imread(sys.argv[2])
