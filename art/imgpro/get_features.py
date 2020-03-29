"""
Functionality for getting features from an single jpg image
"""

import cv2

TEST_IMG = "/Users/bradwest/drive/msu/stat575/paap/data/img/christies/test.jpg"


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("No image loaded for path {}".format(path))
    return img


def prop_dark(img, threshold):
    """
    Get the proportion of pixels with intensity less than the threshold
    :param np.array img: grayscale image
    :param int threshold: max intensity
    """
    if len(img.shape) != 2:
        raise ValueError("img must be two dimensional. Shape: {}".format(img.shape))
    return (img < threshold).sum() / (img.shape[0] * img.shape[1])


if __name__ == "__main__":
    img_features = {}
    img = load_img(TEST_IMG)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_features["prop_dark"] = prop_dark(gray_img, 65)
