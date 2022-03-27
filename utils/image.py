import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_COLOR)


def write_image(img: np.ndarray, path: str):
    cv2.imwrite(path, img)


def detect_circles(cv_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    res = cv2.HoughCircles(
        image=gray,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=100,
        param2=50,
        minRadius=0,
        maxRadius=0,
    )
    return res[0] if res is not None else None


def segment_cuboid_main_face(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    face_brightness = np.argmax(hist.flatten())
    _, thresh1 = cv2.threshold(gray, int(2.1 * face_brightness), 1, cv2.THRESH_BINARY)
    mask = thresh1.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    assert num_labels > 0
    idx, area = max(enumerate(stats[1:]), key=lambda x: x[1][cv2.CC_STAT_AREA])
    final_mask = labels == (idx + 1)
    return (final_mask * 255).astype(np.uint8), area


def detect_cuboid_face(cv_img):
    mask, area = segment_cuboid_main_face(cv_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, area


def circle_area(circle):
    return np.pin * circle[2] ** 2
