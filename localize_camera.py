# The visualization of the detection result referenced: https://www.pyimagesearch.com/2020/11/02/apriltag-with-python/
# the computation of the detection result referenced: https://raceon.io/localization/#apriltag-detection
# pupil-apriltags official documentation: https://pypi.org/project/pupil-apriltags/

from detect_apriltag import TagDetector
from tag import Tag

import math
import numpy as np
import pickle
import cv2 as cv
import argparse

# only run the below algorithm if localize_camera is being run directly
if __name__ == '__main__':

    # opens the file in binary form for reading
    # read in the given camera parameters
    with open('cam_params.pickle', 'rb') as f:
        camera_matrix, distortion_coefficients = pickle.load(f)

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True)
    args = vars(ap.parse_args())
    img_path = args['image']
    original_image = cv.imread(img_path)
    cv.imshow('Original Image', original_image)

    # create a Tag object, and add tags with global parameters
    tags = Tag(tag_size=0.084, family='tagStandard41h12')
    tags.add_tag(id=0, location=(0.1016, 0, 1.395), orientation=(-90, 0, 0))
    # array([[1,0,0],[0,0,1],[0,-1,0]]) corresponds to (-90,0,0) in euler angles

    # create a TagDetector object and run detection
    tagDetector = TagDetector(
        camera_matrix, distortion_coefficients, tags.family)
    undistorted_img, detection_results = tagDetector.detect_tags(
        img_path, tags.tag_size)

    # store the positions and angles of all detected apriltags
    # and compute their average at the end
    avg_pos = np.zeros((3,))
    avg_angles = np.zeros((3,))

    # process the detection results one by one
    for result in detection_results:
        # result is the current detected tag

        # Visualization
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = result.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv.line(undistorted_img, ptA, ptB, (0, 255, 0), 4)
        cv.line(undistorted_img, ptB, ptC, (0, 255, 0), 4)
        cv.line(undistorted_img, ptC, ptD, (0, 255, 0), 4)
        cv.line(undistorted_img, ptD, ptA, (0, 255, 0), 4)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(result.center[0]), int(result.center[1]))
        cv.circle(undistorted_img, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tagFamily = result.tag_family.decode("utf-8")
        cv.putText(undistorted_img, tagFamily, (ptA[0], ptA[1] + 50),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Computation
        # first, compute the camera position in the tag frame
        # extract the rotational matrix and translation vector in the tag frame
        R_matrix_tag = result.pose_R
        T_vector_tag = (result.pose_t).reshape(3,)
        camera_pos_tag = R_matrix_tag.T @ (-T_vector_tag)

        # next, compute the camera position and angle in the global frame
        detected_id = result.tag_id
        if detected_id in tags.locations.keys():
            # extract the rotational matrix and translation vector in the global frame
            R_matrix_global = tags.orientations[detected_id]
            T_vector_global = tags.locations[detected_id]

            # compute the camera position in the global frame
            camera_pos_global = R_matrix_global @ camera_pos_tag + T_vector_global
            avg_pos += camera_pos_global

            # compute the camera angles in the global frame
            total_R_matrix = R_matrix_tag @ R_matrix_global
            camera_angles_global = tags.rotationMatrixToEulerAngles(
                total_R_matrix)
            avg_angles += camera_angles_global

        else:
            print("Tag is not added")

    # save the undistorted image and display it
    destination_dir = 'images/undistorted_orientation/'
    cv.imwrite(destination_dir + img_path.split("/")[-1], undistorted_img)
    cv.imshow("Undistorted Image", undistorted_img)

    avg_pos /= len(detection_results)
    avg_angles /= len(detection_results)
    avg_angles = avg_angles * 180 / math.pi     # convert radians to degrees

    np.set_printoptions(precision=3)
    print()
    print("################################################")
    print(img_path.split("/")[-1])
    print()
    print("Camera global position: ", avg_pos)
    print("Camera global angles: ", avg_angles)
    print("################################################")
    print()

cv.waitKey(0)
