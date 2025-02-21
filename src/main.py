import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from camera_utils import projection_matrix
from render import render
from obj_loader import OBJ


MIN_MATCHES = 90
FOCAL_LENGTH = 800
Cx, Cy = 640, 360 # for 1280x720 resolution

K = np.array([[FOCAL_LENGTH, 0, Cx], [0, FOCAL_LENGTH, Cy], [0, 0, 1]], dtype=np.float32)

H = None

marker_image = cv2.imread("data/markers/marker_1.png", 0)

obj = OBJ("data/models/Cow.obj", swapyz=True)

data = np.load("data/marker_keypoints/marker_1_keypoints.npz")


marker_descriptors = data["descriptors"]
marker_keypoints = [cv2.KeyPoint(x, y, 1) for x, y in data["keypoints"]]

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)

for _ in range(10):
    if cap.isOpened():
        print("Camera opened")
        break
    print("Camera not opened")
    time.sleep(10)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Empty frame received, skipping...")
        continue

    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    frame_keypoints, frame_descriptors = orb.detectAndCompute(frame, None)


    if frame_descriptors is not None and marker_descriptors is not None and frame_keypoints is not None:

        matches = bf.match(frame_descriptors, marker_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > MIN_MATCHES:
        # differentiate distance between source and destination keypoints
            src_pts = np.float32([marker_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            h, w = marker_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, H)
            frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # Compute projection matrix
        if H is not None:
            projection = projection_matrix(H, K)
            frame = render(frame, obj, projection, marker_image, False)

        frame_with_matches = cv2.drawMatches(frame, frame_keypoints, marker_image, marker_keypoints, matches[:MIN_MATCHES], 0, flags=2)


        cv2.imshow('frame_with_matches', frame_with_matches)
        print(f"Number of matches: {len(matches)}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
