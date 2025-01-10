import cv2
import numpy as np
import matplotlib.pyplot as plt

marker_image = cv2.imread("data/markers/marker_1.png", cv2.IMREAD_GRAYSCALE)

data = np.load("data/marker_keypoints/marker_1_keypoints.npz")


marker_descriptors = data["descriptors"]
marker_keypoints = [cv2.KeyPoint(x, y, 1) for x, y in data["keypoints"]]

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

    if frame_descriptors is not None and marker_descriptors is not None:
        matches = bf.match(frame_descriptors, marker_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        frame_with_matches = cv2.drawMatches(marker_image, marker_keypoints, gray_frame, frame_keypoints, matches[:10], None, flags=2)

        cv2.imshow('matches', frame_with_matches)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
