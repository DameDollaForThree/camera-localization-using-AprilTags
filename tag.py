# Class definition for a Tag object
# Referenced https://github.com/teddylew12/race_on_cv/blob/master/tag.py
# and https://learnopencv.com/rotation-matrix-to-euler-angles/

import math
import numpy as np


class Tag():
    """
    Each Tag object may contain several tags
    """

    def __init__(self, tag_size, family):
        self.tag_size = tag_size
        self.family = family
        self.locations = {}     # dictionary to store the locations of all tags
        self.orientations = {}      # dictionary to store the orientations of all tags

    def add_tag(self, id, location, orientation):
        """
        add a new tag to the Tag object
        """
        self.locations[id] = location
        # convert orientation angles from degrees to radians
        orientation = (np.array(orientation)).reshape(3,)
        orientation = orientation / 180 * math.pi
        self.orientations[id] = self.eulerAnglesToRotationMatrix(orientation)

    def eulerAnglesToRotationMatrix(self, theta):
        """
        Calculates Rotation Matrix given euler angles.
        - theta: tuple/list (theta_x,theta_y,theta_z) euler angles (in radians)
        """

        R_x = np.array([[1,         0,                  0],
                        [0,         math.cos(theta[0]), -math.sin(theta[0])],
                        [0,         math.sin(theta[0]), math.cos(theta[0])]
                        ])

        R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])],
                        [0,                     1,      0],
                        [-math.sin(theta[1]),   0,      math.cos(theta[1])]
                        ])

        R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                        [math.sin(theta[2]),    math.cos(theta[2]),     0],
                        [0,                     0,                      1]
                        ])

        R = np.dot(R_z, np.dot(R_y, R_x))

        return R

    def isRotationMatrix(self, R):
        """
        Checks if a matrix is a valid rotation matrix.
        """
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def rotationMatrixToEulerAngles(self, R):
        """
        Calculates rotation matrix to euler angles
        The result is the same as MATLAB except the order
        of the euler angles ( x and z are swapped ).
        """

        assert(self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])
