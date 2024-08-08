import cv2
import numpy as np
import matplotlib as plt


left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []


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

    # Convert frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection function with thresh ratio 1:3
    canny_edges = cv2.Canny(gray, low_t, high_t)

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


def hough_transform(image):

    # Distance resolution of the accumulator in pixels.
    rho = 1
    # Angle resolution of the accumulator in radians.
    theta = np.pi / 180
    # Only lines that are greater than threshold will be returned.
    threshold = 20
    # Line segments shorter than that are rejected.
    minLineLength = 20
    # Maximum allowed gap between points on the same line to link them
    maxLineGap = 500
    # function returns an array containing dimensions of straight lines
    # appearing in the input image
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def detect_line(thresh):
    contours, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours is None:
        return []
    return [cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01) for cnt in contours]


def __find_from_to_xy(img, line):
    ht, wd, dp = img.shape
    [vx, vy, x, y] = line

    # y = ax + b
    x1, y1 = int(x-vx*int(ht/2)), int(y-vy*int(ht/2))
    x2, y2 = int(x+vx*int(ht/2)), int(y+vy*int(ht/2))

    # kiem tra bien
    x1 = x1 if x1 <= wd else x1 if x1 >= 0 else 0
    x2 = x2 if x2 <= wd else x2 if x2 >= 0 else 0
    y1 = y1 if y1 <= ht else y1 if y1 >= int(ht/2) else int(ht/2)
    y2 = y2 if y2 <= ht else y2 if y2 >= int(ht/2) else int(ht/2)
    return x1, y1, x2, y2


def __find_angle(img, line):
    x1, y1, x2, y2 = __find_from_to_xy(img, line)
    dx, dy = x2 - x1, y2 - y1
    angle = np.arctan2(dy, dx) * (180 / np.pi)
    return angle, x1, y1, x2, y2


def draw(img, lines):
    for line in lines:
        angle, x1, y1, x2, y2 = __find_angle(img, line)
        if (angle <= 20) and (angle >= - 20):
            return img
        cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
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


def extract_visualize_data(original_img):
    """
    An utilty function to visualize splited window images.

    :param original_img - Raw road line image
    """
    original_img_bin, Minv = pipeline(original_img)

    left_fit, right_fit, left_lane_inds, right_lane_inds, visualization_data = sliding_window_polyfit(original_img_bin)

    h = original_img.shape[0]
    left_fit_x_int = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
    right_fit_x_int = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]
    # print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

    rectangles = visualization_data[0]
    histogram = visualization_data[1]

    # Create an output image to draw on and  visualize the result
    out_img = np.uint8(np.dstack((original_img_bin, original_img_bin, original_img_bin)) * 255)
    # Generate x and y values for plotting
    ploty = np.linspace(0, original_img_bin.shape[0] - 1, original_img_bin.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    for rect in rectangles:
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (rect[2], rect[0]), (rect[3], rect[1]), (0, 255, 0), 2)
        cv2.rectangle(out_img, (rect[4], rect[0]), (rect[5], rect[1]), (0, 255, 0), 2)
        # Identify the x and y positions of all nonzero pixels in the image
    nonzero = original_img_bin.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [100, 200, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)


def sliding(img):
    # Histogram
    histogram = np.sum(img, axis=0)
    midpoint = np.int_(histogram.shape[0] / 2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Sliding Window
    y = 210
    lx = []
    rx = []

    msk = img.copy()

    # Creating x-coordinates
    while y > 0:

        # Left threshold
        img1 = img[y - 20:y, left_base - 30:left_base + 30]
        contours1, _ = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            M1 = cv2.moments(contour)
            if M1["m00"] != 0:
                cx1 = int(M1["m10"] / M1["m00"])
                cy1 = int(M1["m01"] / M1["m00"])
                lx.append(left_base - 30 + cx1)
                left_base = left_base - 30 + cx1

        # Right threshold
        img2 = img[y - 20:y, right_base - 30:right_base + 30]
        contours, _ = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour2 in contours:
            M2 = cv2.moments(contour2)
            if M2["m00"] != 0:
                cx2 = int(M2["m10"] / M2["m00"])
                cy2 = int(M2["m01"] / M2["m00"])
                rx.append(right_base - 30 + cx2)
                right_base = right_base - 30 + cx2

        cv2.rectangle(msk, (left_base - 30, y), (left_base + 30, y - 20), (255, 255, 255), 1)
        cv2.rectangle(msk, (right_base - 30, y), (right_base + 30, y - 20), (255, 255, 255), 1)
        y -= 20
    return msk

