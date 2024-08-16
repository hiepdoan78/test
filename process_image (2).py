import cv2
import numpy as np
import matplotlib.pyplot as plt

present_std_left, present_mean_left, left_x, present_std_right, present_mean_right, right_x = [0], [0], [], [0], [0], []


def binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    return thresh


def denoise_frame(image):
    kernel = np.ones((3, 3), np.float32) / 9
    denoiseframe = cv2.filter2D(image, -1, kernel)
    blur = cv2.GaussianBlur(denoiseframe, (5, 5), 0)
    return blur


def detect_edges(image):
    # first threshold for the hysteresis procedure
    low_t = 50
    # second threshold for the hysteresis procedure
    high_t = 150

    # Apply Canny edge detection function with thresh ratio 1:3
    canny_edges = cv2.Canny(image, low_t, high_t)

    return canny_edges


def region(image):
    # creating a polygon to focus only on the road in the picture
    height, width = image.shape[:2]
    bottom_left = [width * 0, height * 1]
    top_left = [width * 0, height * 0.5]
    bottom_right = [width * 1, height * 1]
    top_right = [width * 1, height * 0.5]
    return [bottom_left, top_left, bottom_right, top_right]


def region_selection(image):
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        # color of the mask polygon (white)
        ignore_mask_color = 255
        # created this polygon based on where placed camera
    vertices = np.array([region(image)], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def detect_line(thresh):
    contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        return []
    return [cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01) for cnt in contours]


def find_from_to_xy(img, line):
    ht, wd, dp = img.shape
    [vx, vy, x, y] = line

    # y = ax + b
    x1, y1 = int(x-vx*int(ht/4)), int(y-vy*int(ht/4))
    x2, y2 = int(x+vx*int(ht/4)), int(y+vy*int(ht/4))

    # kiem tra bien
    x1 = x1 if x1 <= wd else x1 if x1 >= 0 else 0
    x2 = x2 if x2 <= wd else x2 if x2 >= 0 else 0
    y1 = y1 if y1 <= ht else y1 if y1 >= int(ht/4) else int(ht/4)
    y2 = y2 if y2 <= ht else y2 if y2 >= int(ht/4) else int(ht/4)
    return x1, y1, x2, y2


def find_angle(img, line):
    x1, y1, x2, y2 = find_from_to_xy(img, line)
    dx, dy = x2 - x1, y2 - y1
    angle = np.arctan2(dy, dx) * (180 / np.pi)
    return angle, x1, y1, x2, y2


def draw(img, lines):
    for line in lines:
        angle, x1, y1, x2, y2 = find_angle(img, line)
        if (angle <= 20) and (angle >= - 20):
            return img
        cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
    return img


def fillpoly(img, lines):
    wt, wd, dp = img.shape
    bl, tl, br, tr = [], [], [], []
    for line in lines:
        angle, x1, y1, x2, y2 = find_angle(img, line)
        if x1 > wd/2:
            if y1 > wt/2:
                tl.append(x1)
                tl.append(y1)
            else:
                bl.append(x1)
                bl.append(y1)
        else:
            if y1 > wt/2:
                tr.append(x1)
                tr.append(y1)
            else:
                br.append(x1)
                br.append(y1)
        if x2 > wd / 2:
            if y2 > wt / 2:
                tl.append(x2)
                tl.append(y2)
            else:
                bl.append(x2)
                bl.append(y2)
        else:
            if y2 > wt / 2:
                tr.append(x2)
                tr.append(y2)
            else:
                br.append(x2)
                br.append(y2)
    vertices = np.array([bl, tl, tr, br], dtype=np.int32)
    cv2.fillPoly(img, [vertices], (0, 0, 255))
    return img


def warp_perspective(image):

    # Get image size
    height, width = image.shape[:2]

    # Offset for frame ratio saving
    offset = 50

    # Perspective points to be warped
    pts1 = np.float32(region(image))

    # Window to be shown
    pts2 = np.float32([[0, 240], [0, 0], [320, 240], [320, 0]])

    # Matrix to warp the image for bird_eye_view window
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Final warping perspective
    bird_eye_view = cv2.warpPerspective(image, matrix, (width, height))

    return bird_eye_view


def sliding_window(img, nwindows=9, margin=150, minpix=1, draw_windows=True):

    global left_a, left_b, left_c, right_a, right_b, right_c
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # find peaks of left and right halves
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int_(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 3)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 3)
            # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))

    #        if len(good_right_inds) > minpix:
    #            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
    #        elif len(good_left_inds) > minpix:
    #            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
    #        if len(good_left_inds) > minpix:
    #            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
    #        elif len(good_right_inds) > minpix:
    #            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])

    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])

    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])

    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
    right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


def sliding_window_polyfit(img):
    """
    This is used to split an image multiple windows. After spliting windows, drawing a histogram for each windows
    highest peak can be found.

    :param img - unwarped image
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img, axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int_(histogram.shape[0] // 2)
    quarter_point = np.int_(midpoint // 2)
    # Previously the left/right base was the max of the left/right half of the histogram
    # this changes it so that only a quarter of the histogram (directly to the left/right) is considered
    leftx_base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
    rightx_base = np.argmax(histogram[midpoint:(midpoint + quarter_point)]) + midpoint

    # print('base pts:', leftx_base, rightx_base)

    # Choose the number of sliding windows
    nwindows = 10
    # Set height of windows
    window_height = np.int_(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Rectangle data for visualization
    rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit, right_fit = (None, None)
    # Fit a second order polynomial to each
    if len(leftx) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit = np.polyfit(righty, rightx, 2)

    visualization_data = (rectangle_data, histogram)

    return left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data


def text_line_detect(image, lines):
    wt, wd, dp = image.shape
    left_point, right_point, center_x = point_x(image, lines)
    if len(left_point) > 0:
        cv2.putText(image, 'Left Lane Detected', (int(wt*3/4), int(wd/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'No Left Lane Detected', (int(wt*3/4), int(wd/3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    if len(right_point) > 0:
        cv2.putText(image, 'Right Lane Detected', (int(wt*3/4), int(wd/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'No Right Lane Detected', (int(wt*3/4), int(wd/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
                    cv2.LINE_AA)

    return image


def find_min(arr):
    x_min = arr[0]

    for i in range(len(arr)):
        if x_min > arr[i]:
            x_min = arr[i]

    return x_min


def find_max(arr):
    x_max = arr[0]
    a = 0
    for i in range(len(arr)):
        if abs(x_max) < abs(arr[i]):
            x_max = arr[i]
            a = i
    return x_max, a


def find_0(arr):
    a = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            a += 1
    if a == len(arr):
        return 1
    else:
        return 0


def point_x(image, lines):
    wt, wd, dp = image.shape
    center_x = wd / 2
    left_point = []
    right_point = []
    for line in lines:
        [vx, vy, x, y] = line
        if x < center_x:
            left_point.append(x)
        if x > center_x:
            right_point.append(x)
    return left_point, right_point, center_x


def offset(image, lines):
    wt, wd, dp = image.shape
    left_point, right_point, center_x = point_x(image, lines)
    left_mean, right_mean = 0, 0
    if len(left_point) > 0 and len(right_point) > 0:
        left_mean = np.mean(left_point)
        if left_mean > 100:
            left_mean = 0
        right_mean = np.mean(right_point)
        if 40 < left_mean:
            cv2.putText(image, 'Turn right', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
        elif right_mean < 270:
            cv2.putText(image, 'Turn left', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)
    if len(left_point) < 1 and len(right_point) > 0:
        right_mean = np.mean(right_point)
        if right_mean < 270:
            cv2.putText(image, 'Turn left', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2, cv2.LINE_AA)
    if len(left_point) > 0 and len(right_point) < 1:
        left_mean = np.mean(left_point)
        if left_mean > 100:
            left_mean = 0
        if 40 < left_mean:
            cv2.putText(image, 'Turn right', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2, cv2.LINE_AA)
    return image


def average(lst):
    return sum(lst) / len(lst)


def find_offset(image, lines):

    wt, wd, dp = image.shape
    center_x = wd / 2
    cv2.line(image, (wd // 2, wt), (wd // 2, 0), [230, 43, 239], 2)
    x_left, x_right, y_left, y_right = [], [], [], []
    offset_left, offset_right, y_offset_left, y_offset_right = [], [], [], []

    for line in lines:
        angle, x1, y1, x2, y2 = find_angle(image, line)
        if x1 < (center_x - 20):
            x_left.append(x1)
            y_left.append(y1)
        if x1 > (center_x + 20):
            x_right.append(x1)
            y_right.append(y1)
        if x2 < (center_x - 20):
            x_left.append(x2)
            y_left.append(y2)
        if x2 > (center_x + 20):
            x_right.append(x2)
            y_right.append(y2)
    print(x_left, y_left, x_right, y_right)

    left_x_empty = find_0(x_left)
    right_x_empty = find_0(x_right)
    left_y_empty = find_0(y_left)
    right_y_empty = find_0(y_right)
    print(left_x_empty, left_y_empty, right_x_empty, right_y_empty)

    if left_x_empty == 0 and right_x_empty == 0 and left_y_empty == 0 and right_y_empty == 0:
        line_left = np.polyfit(x_left, y_left, 1)
        for i in range(len(y_right)):
            line_left[1] -= y_right[i]
            x_line_left = np.roots(line_left) // 1
            line_left[1] += y_right[i]
            a = np.int_(x_line_left[0])
            dist_left = abs(np.int_(a - x_right[i]))
            center_lane_left = abs(np.int_(dist_left // 2))
            if a > 0:
                center_lane_left += a

            offset_left.append(np.int_(center_x - center_lane_left))
            y_offset_left.append(y_right[i])

            #cv2.line(image, (a, y_right[i]), (x_right[i], y_right[i]), [230, 43, 239], 2)
            #plt.plot(mid_lane_left, y_right[i], 'o')

        line_right = np.polyfit(x_right, y_right, 1)
        for i in range(len(y_left)):
            line_right[1] -= y_left[i]
            x_line_right = np.roots(line_right) // 1
            line_right[1] += y_left[i]
            b = np.int_(x_line_right[0])
            dist_right = abs(np.int_(b - x_left[i]))
            center_lane_right = abs(np.int_(dist_right // 2))
            if b < wd:
                center_lane_right += (wd - b)

            offset_right.append(np.int_(center_x - center_lane_right))
            y_offset_right.append(y_left[i])


            #cv2.line(image, (b, y_left[i]), (x_left[i], y_left[i]), [230, 43, 239], 2)
            #plt.plot(mid_lane_right, y_right[i], 'o')

        max_center_line_left, i = find_max(offset_left)
        max_center_line_right, j = find_max(offset_right)

        if abs(max_center_line_left) > 20 or abs(max_center_line_right) > 20:
            if abs(y_offset_left[i]) > abs(y_offset_right[j]):
                cv2.putText(image, 'Turn left', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2, cv2.LINE_AA)
            elif abs(y_offset_left[i]) < abs(y_offset_right[j]):
                cv2.putText(image, 'Turn right', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2, cv2.LINE_AA)
        elif abs(max_center_line_left) <= 20 or abs(max_center_line_right) <= 20:
            cv2.putText(image, 'Straight', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)

    if left_x_empty == 1 or left_y_empty == 1:
        #for i in range(len(y_right)):
        #    x_left.append(0)
        #    dist = abs(np.int_((x_left[i]) - (x_right[i])))
        #    center_lane = abs(np.int_(dist // 2))

        cv2.putText(image, 'Turn left', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)

    if right_x_empty == 1 or right_y_empty == 1:
        #for i in range(len(y_left)):
        #    x_right.append(wd)
        #    dist = abs(np.int_((x_left[i]) - (x_right[i])))
        #    center_lane = abs(np.int_(dist // 2))
        cv2.putText(image, 'Turn right', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2, cv2.LINE_AA)


    #average_offset = find_max(offset)
    #print(offset)
    #if abs(average_offset) > 30:
    #    if average_offset < 0:
    #        cv2.putText(image, 'Turn right', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                    (0, 255, 0), 2, cv2.LINE_AA)
    #    elif average_offset > 0:
    #        cv2.putText(image, 'Turn left', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                    (0, 255, 0), 2, cv2.LINE_AA)
    #else:
    #    cv2.putText(image, 'Straight', (int(wt * 3 / 4), int(wd / 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                (0, 255, 0), 2, cv2.LINE_AA)
    return image



