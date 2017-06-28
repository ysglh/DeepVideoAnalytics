"""
Implements frame/segment/image analysis algorithms implemented using Open CV
"""
import cv2


def blurryness(image):
    """

    :param image:
    :return:
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()
