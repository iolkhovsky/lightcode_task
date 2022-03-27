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
    hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    histogram_sorted = sorted(enumerate(hist), key=lambda x: x[1], reverse=True)
    main_face_brightness = histogram_sorted[1][0] * 16
    _, binary_img = cv2.threshold(gray, int(main_face_brightness), 1, cv2.THRESH_BINARY)
    binary_img = binary_img.astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, 4, cv2.CV_32S)
    assert num_labels > 0
    idx, stat = max(enumerate(stats[1:]), key=lambda x: x[1][cv2.CC_STAT_AREA])
    mask = labels == (idx + 1)
    return (mask * 255).astype(np.uint8), stat[cv2.CC_STAT_AREA]


def detect_cuboid_face(cv_img):
    mask, area = segment_cuboid_main_face(cv_img)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, area


def circle_area(circle):
    return np.pi * circle[2] ** 2
