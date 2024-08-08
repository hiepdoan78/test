import cv2
import numpy as np
from process_image import *
import matplotlib.pyplot as plt


capture = cv2.VideoCapture(r"C:\Users\Admin\PycharmProjects\detect_lane\video test\21-09-17-12-42-08.mp4")
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
        binary_image = binary(bird_eye_view)

        # find contour
        lines = detect_line(binary_image)

        # apply sliding_window
        #sli = sliding(binary_image)

        # draw line
        result = draw(bird_eye_view, lines)

        cv2.imshow("regio", bird_eye_view)
        cv2.imshow("region", img)
        #plt.imshow(img)
        #plt.show()
        print(lines)
        key = cv2.waitKey(25)
        if key == 27:
            break

if __name__ == '__main__':
    test()