import numpy as np
import math


def projection_matrix(H, K):

    if np.linalg.det(H) < 0:
        H = -H

    # Decomposte into rotation and translation conponents
    rotation_and_translation = np.linalg.inv(K) @ H

    # rotation_and_translation = np.dot(np.linalg.inv(K), H)
    col_1 = rotation_and_translation[:, 0]
    col_2 = rotation_and_translation[:, 1]
    col_3 = rotation_and_translation[:, 2]

    l = math.sqrt(np.linalg.norm(col_1, 2) * np.linalg.norm(col_2, 2))

    rot_1 = col_1 / l
    rot_2 = col_2 / l
    translation = col_3 / l

    c = rot_1 + rot_2
    p = np.cross(rot_1, rot_2)
    d = np.cross(c, p)

    rot_1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
    rot_3 = np.cross(rot_1, rot_2)

    projection = np.stack((rot_1, rot_2, rot_3, translation)).T
    projection = K @ projection

    return projection
