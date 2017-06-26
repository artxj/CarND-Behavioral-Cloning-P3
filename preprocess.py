
import cv2

def clahe_image(image):
    """
    Applies OpenCV CLAHE to increase contrast on dark images
    http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
