import os
import cv2
import numpy as np


def extract_keypoints(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return

    orb = cv2.ORB_create()
    keypts, des = orb.detectAndCompute(image, None)

    np.savez(output_path, keypoints=[kp.pt for kp in keypts], descriptors=des)

    print(f"Keypoints and descriptors extracted for {image_path} and saved to {output_path}")


def process_markers(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + "_keypoints.npz")
            extract_keypoints(image_path, output_path)


if __name__ == "__main__":
    input_dir = "data/markers/"
    output_dir = "data/marker_keypoints/"
    process_markers(input_dir, output_dir)
