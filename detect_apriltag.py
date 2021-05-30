# The undistortion process referenced: https://www.programcreek.com/python/example/89299/cv2.getOptimalNewCameraMatrix
# pupil-apriltags official documentation: https://pypi.org/project/pupil-apriltags/

import cv2 as cv
import pupil_apriltags


class TagDetector():
    """
    A TagDetector object that smooths the original distorted image,
    and detect all AprilTags in the image
    """

    def __init__(self, camera_matrix, distortion_coefficients, tag_family):
        """
        Initialize an actual dector from the pupil_apriltags package
        """
        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.detector = pupil_apriltags.Detector(families=tag_family)

    def undistort_image(self, img):
        """
        Return the undistorted version of the camera matrix and image
        - img: the original image with distortion
        """
        h, w = img.shape[:2]
        new_camera_matrix, _ = cv.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coefficients, (w, h), 1, (w, h))
        undistorted_img = cv.undistort(img, self.camera_matrix,
                                       self.distortion_coefficients, None, new_camera_matrix)
        return undistorted_img, new_camera_matrix

    def detect_tags(self, img_path, tag_size):
        """
        Detect all Apriltags in the input image
        - img_path: path to the original image/photo
        - tag_size: tag size in meters
        Return the undistorted version of the image and the detection results
        """
        # load the input image, perform undistortion, and convert it to gray scale
        img = cv.imread(img_path)
        undistorted_img, new_camera_matrix = self.undistort_image(img)
        gray_img = cv.cvtColor(undistorted_img, cv.COLOR_BGR2GRAY)

        # from the new camera matrix, extract the camera intrinsic parameters as an array
        camera_params = [new_camera_matrix[0, 0], new_camera_matrix[1, 1],
                         new_camera_matrix[0, 2], new_camera_matrix[1, 2]]

        # detect the Apriltags
        detection_results = self.detector.detect(gray_img, estimate_tag_pose=True,
                                                 camera_params=camera_params, tag_size=tag_size)

        return undistorted_img, detection_results
