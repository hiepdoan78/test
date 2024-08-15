import cv2
import numpy as np
from process_image import *
import matplotlib.pyplot as plt


capture = cv2.VideoCapture(r"C:\Users\Admin\PycharmProjects\last_hope\21-09-17-12-42-08.mp4")
def test():
    #print("Chuong trinh nhan dien lan duong")
    #print("Nhap v de xu ly tren video,nhap t de xu ly truc tiep:")
    #a = input()
    #while (a != "t") and (a != "v"):
    #    print("Vui long nhap lai t hoac v:")
    #    a = input()
    #if a == 't':
    #    capture = cv2.VideoCapture(0)
    #elif a == "v":
    #    print("Nhap ten video:")
    #    b = input()
    #    capture = cv2.VideoCapture(b)

    while True:

        ret, img = capture.read()

        # combine denoise frame and gaussian blur to remove noise from the frames
        blur = denoise_frame(img)

        # select region of interest
        region = region_selection(blur)

        # perspective tranform
        bird_eye_view = warp_perspective(blur)

        # change color from RGB to Binary
        binary_image1 = binary(region)
        binary_image3 = binary(bird_eye_view)

        # find contour
        lines_1 = detect_line(binary_image1)
        lines_3 = detect_line(binary_image3)

        # draw line
        result3 = draw(bird_eye_view, lines_3)
        result1 = draw(img, lines_1)

        result1 = line_detect(img, lines_1)
        #result3 = offset(bird_eye_view, lines_3)
        result3 = find_offset(bird_eye_view, lines_3)

        cv2.imshow("draw_image", result1)
        cv2.imshow("cc", result3)
        #plt.imshow(bird_eye_view)
        #plt.imshow(find_offset(bird_eye_view, lines_3))
        #print(find_offset(bird_eye_view, lines_3))
        plt.show()

        key = cv2.waitKey(25)
        if key == 27:
            break


if __name__ == '__main__':
    test()