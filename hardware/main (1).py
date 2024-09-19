import cv2
from process_image import *
import matplotlib.pyplot as plt
from detect import *

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


def test():
    while True:

        ret, img = cap.read()

        # combine denoise frame and gaussian blur to remove noise from the frames
        blur = denoise_frame(img)

        # perspective tranform
        bird_eye_view = warp_perspective(blur)

        # change color from RGB to Binary
        binary_image3 = binary(bird_eye_view)

        # find contour
        lines_3 = detect_line(binary_image3)

        # draw line
        result3 = draw(bird_eye_view, lines_3)

        # define direction to move
        result3 = find_offset(bird_eye_view, lines_3)

        # show result

        cv2.imwrite("img.jpg", result3)

        main()
        #print(steering_angle(bird_eye_view, lines_3))

        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    test()



