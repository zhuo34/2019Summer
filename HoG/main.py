import cv2

import hog

if __name__ == '__main__':
    img = hog.cv2.imread('c.png')

    my_hog = hog.Hog((48, 48), (2, 2), (1, 1), (4, 4))

    v, canvas = my_hog.compute(img, visualize=True)
    cv2.imshow("HOG", canvas * 3)
    cv2.imwrite("0.png", canvas * 3)
    cv2.waitKey(0)

