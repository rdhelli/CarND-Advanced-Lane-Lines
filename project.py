#%%
# ----------------------------------------------------------------------
# Imported libraries
# ----------------------------------------------------------------------
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

#%%
# ----------------------------------------------------------------------
# Camera calibration using chessboard images
# ----------------------------------------------------------------------
# chessboard dimensions of black square intersections
nx = 9
ny = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((ny*nx, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('./camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    # If found, add object points, image points
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        print(fname, ": found corners")
        # Draw corners and save image
        img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imwrite(r"./output_images/" +
                    fname.split('\\')[-1].split('.')[0] +
                    "_1chbrd.jpg", img)
    # If not found, display message
    else:
        print(fname, ": corners not found")
        cv2.putText(img, "corners not found", (100, 100),
                    cv2.FONT_HERSHEY_TRIPLEX, 3, (0, 0, 255))
        cv2.imwrite(r"./output_images/" +
                    fname.split('\\')[-1].split('.')[0] +
                    "_1chbrd.jpg", img)

#%%
# ----------------------------------------------------------------------
# Correcting for distortion
# ----------------------------------------------------------------------
# perform camera calibration, to extract distortion and transformation matrix
cal_shape = cv2.imread('./camera_cal/calibration1.jpg').shape[1::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,
                                                   imgpoints,
                                                   cal_shape,
                                                   None, None)
# perform undistortion transformation on calibration images
for fname in images:
    img = cv2.imread(fname)
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    print(fname, ": undistorted")
    cv2.imwrite(r"./output_images/" +
                fname.split('\\')[-1].split('.')[0] +
                "_2undist.jpg", undistorted)


#%%
# ----------------------------------------------------------------------
# Applying perspective transformation on chessboard images
# ----------------------------------------------------------------------
def corners_unwarp(img, nx, ny, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    p = 100
    if ret:
        src = np.float32([corners[0], corners[nx-1],
                          corners[-1], corners[-nx]])
        img_size = gray.shape[::-1]
        dst = np.float32([[p, p], [img_size[0]-p, p],
                          [img_size[0]-p, img_size[1]-p], [p, img_size[1]-p]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, M, img_size)
        return warped, M
    else:
        return None, None


# perform perspective transformation on calibration images
for fname in images:
    img = cv2.imread(fname)
    warped, _ = corners_unwarp(img, nx, ny, mtx, dist)
    if warped is not None:
        print(fname, ": warped")
        cv2.imwrite(r"./output_images/" +
                    fname.split('\\')[-1].split('.')[0] +
                    "_3warped.jpg", warped)
    else:
        print(fname, ": no success warping")
        cv2.putText(img, "no success warping", (100, 100),
                    cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0))
        cv2.imwrite(r"./output_images/" +
                    fname.split('\\')[-1].split('.')[0] +
                    "_3warped.jpg", img)

#%%
# ----------------------------------------------------------------------
# Applying perspective transformation on road images = "bird eye view"
# ----------------------------------------------------------------------
images = glob.glob('./test_images/*.jpg')
img_size = cv2.imread(images[0]).shape[1::-1]
p = 300  # hyperparameter to edit (horizontal) size of region of interest

# points order: top left, top right, bottom right, bottom left
# source area = relevant part of the road (known dimensions)
src = np.float32([[578, 460], [706, 460], [1120, 720], [190, 720]])
# destination area = middle rectangle (no cropping necessary)
dst = np.float32([[p, 0], [img_size[0]-p, 0],
                  [img_size[0]-p, img_size[1]], [p, img_size[1]]])
pts1 = np.array(src, np.int32).reshape(-1, 1, 2)
pts2 = np.array(dst, np.int32).reshape(-1, 1, 2)

# store transformation matrix
M = cv2.getPerspectiveTransform(src, dst)


# the undist mtx and dist factors shall be predetermined at camera calibration
def perspective_transform(img, mtx, dist, M):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped


# perform perspective transformation on test images
warped_images = []
for fname in images:
    img = cv2.imread(fname)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # draw the source polygon on a copy
    undist_rect = np.copy(undist)
    undist_rect = cv2.polylines(undist_rect, [pts1], True, (0, 0, 255), 3)
    warped = cv2.warpPerspective(img, M, img_size)
    warped_images.append(cv2.cvtColor(warped,cv2.COLOR_BGR2RGB))
    # draw the destination polygon on a copy
    warped_rect = np.copy(warped)
    warped_rect = cv2.polylines(warped_rect, [pts2], True, (0, 0, 255), 6)
    print(fname, ": warped")
    cv2.imwrite(r"./output_images/" +
                fname.split('\\')[-1].split('.')[0] +
                "_1roi.jpg", undist_rect)
    cv2.imwrite(r"./output_images/" +
                fname.split('\\')[-1].split('.')[0] +
                "_1warped.jpg", warped_rect)

#%%
# ----------------------------------------------------------------------
# Applying gradient and color channel filters to produce binary images
# ----------------------------------------------------------------------
images = glob.glob('./test_images/*.jpg')


def grad_n_color_filter(img):
    # thresholding for H channel
    h_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 0]
    h_thr_min = 30
    h_thr_max = 105
    h_bin = np.zeros_like(h_channel)
    h_bin[(h_channel > h_thr_min) & (h_channel <= h_thr_max)] = 1
    # thresholding for S channel
    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]
    s_thr_min = 120
    s_thr_max = 255
    s_bin = np.zeros_like(s_channel)
    s_bin[(s_channel >= s_thr_min) & (s_channel <= s_thr_max)] = 1
    # gradient thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    thr_min = 35
    thr_max = 100
    sxbin = np.zeros_like(scaled_sobel)
    sxbin[(scaled_sobel >= thr_min) & (scaled_sobel <= thr_max)] = 1
    # generate colored image for debugging
    color_binary = np.dstack((s_bin, sxbin, h_bin)) * 255
    # apply combined thresholds
    # H term only used to filter S term during shadowy noise, hence the '&'
    combined_binary = np.zeros_like(sxbin)
    combined_binary[(h_bin == 1) & (s_bin == 1) | (sxbin == 1)] = 1
    return combined_binary, color_binary


# perform combined thresholding on test images
binary_warped_images = []
for i, img in enumerate(warped_images[2:]):
    combined_binary, color_binary = grad_n_color_filter(img)
    binary_warped_images.append(combined_binary)
    # BGR conversion:
    combined_bgr = np.dstack((combined_binary,
                              combined_binary,
                              combined_binary)) * 255
    print("image binarized")
    cv2.imwrite(r"./output_images/" +
                "test" + str(i+1) +
                "_2color.jpg", color_binary)
    cv2.imwrite(r"./output_images/" +
                "test" + str(i+1) +
                "_2combined.jpg", combined_bgr)

#%%
# ----------------------------------------------------------------------
# Lane finding from scratch with sliding windows
# ----------------------------------------------------------------------


def find_lane_sliding_windows(binary_warped):
    # hyperparameters
    nwindows = 12  # number of sliding windows
    margin = 100  # width of windows
    minpix = 50  # minimum no. of pixels to recenter window
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    # Set height of windows based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)
                          ).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)
                           ).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window to mean pos
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    return leftx, lefty, rightx, righty, out_img


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    # Fit a second order polynomial to each side
    if leftx.shape[0] == 0 or rightx.shape[0] == 0:
        return None, None, None
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except TypeError:
        return None, None, None
    # left_fit_values.append(left_fit)
    # right_fit_values.append(right_fit)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    # Calc both polynomials
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty


left_fit_values = []
right_fit_values = []
for i, img in enumerate(binary_warped_images):
    leftx, lefty, rightx, righty, lane_pixels = find_lane_sliding_windows(img)
    left_fitx, right_fitx, ploty = fit_poly(lane_pixels.shape,
                                            leftx, lefty, rightx, righty)
    pts_left = np.vstack((left_fitx, ploty)).astype(np.int32).T
    pts_right = np.vstack((right_fitx, ploty)).astype(np.int32).T
    lane_pixels = cv2.polylines(lane_pixels, [pts_left],
                                False, (255, 255, 0), 5)
    lane_pixels = cv2.polylines(lane_pixels, [pts_right],
                                False, (255, 255, 0), 5)
    print("lane lines searched with sliding windows")
    cv2.imwrite(r"./output_images/" +
                "test" + str(i+1) +
                "_3slidingwindows.jpg", lane_pixels)

# print(left_fit_values)
# print(right_fit_values)

#%%
# ----------------------------------------------------------------------
# Lane finding based on previous positions with polynomial bands
# ----------------------------------------------------------------------
# dummy values defined based on sliding window outputs
# note: these are 'current' fit values instead of 'previous' fit values
left_fit_values = [[ 1.79358557e-04, -2.31047843e-01,  4.11672545e+02],
                   [-2.33136840e-04,  3.18521448e-01,  2.63627058e+02],
                   [ 9.85184549e-05, -2.75844772e-01,  4.75840582e+02],
                   [ 1.29855137e-04, -1.56620135e-01,  4.05587276e+02],
                   [ 3.26011285e-04, -4.07986611e-01,  4.14093246e+02],
                   [ 2.36987443e-04, -4.11018910e-01,  5.23322026e+02]]
right_fit_values = [[ 2.26475912e-04, -3.35719274e-01,  1.12120702e+03],
                    [-3.10019132e-04,  3.80019261e-01,  8.99742065e+02],
                    [ 1.95901368e-04, -2.95628681e-01,  1.11093862e+03],
                    [ 3.15256252e-04, -3.09655817e-01,  1.10818462e+03],
                    [ 2.02975026e-04, -2.61373779e-01,  1.08996055e+03],
                    [ 3.47888733e-04, -3.64620239e-01,  1.14392108e+03]]


def find_lane_around_poly(binary_warped, left_fit, right_fit):
    # hyperparameter
    margin = 60  # width of band around previous polynomial
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set the area of search
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) +
                                   left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                   left_fit[1]*nonzeroy +
                                   left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) +
                                    right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) &
                       (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                    right_fit[1]*nonzeroy +
                                    right_fit[2] + margin)))
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    return leftx, lefty, rightx, righty, out_img


# Run image through the pipeline
# Note that in your project, you'll also want to feed in the previous fits
margin = 60  # width of band around previous polynomial
for i, img in enumerate(binary_warped_images):
    leftx, lefty, rightx, righty, out_img = \
        find_lane_around_poly(img, left_fit_values[i], right_fit_values[i])
    left_fitx, right_fitx, ploty = \
        fit_poly(img.shape, leftx, lefty, rightx, righty)
    window_img = np.zeros_like(out_img)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(
            np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(
            np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(
            np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    print("lane lines searched with polynomial band")
    cv2.imwrite(r"./output_images/" +
                "test" + str(i+1) +
                "_4polyband.jpg", result)
    # Plot the polynomial lines onto the image
    plt.figure()
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.imsave(r"./output_images/" +
               "test" + str(i+1) +
               "_4polyband.jpg", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.close()

#%%
# ----------------------------------------------------------------------
# Calculate offset and curvature radius, then apply reverse transformation
# ----------------------------------------------------------------------
M_inv = np.linalg.inv(M)

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


def measure_curvature(ploty, x_values):
    # If no pixels were found return None
    y_eval = np.max(ploty)
    # Fit new polynomials to x, y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix +
                      fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad


def measure_offset(img_shape, last_x):
    # compute the offset from the center
    return (last_x - img_shape[1]/2) * xm_per_pix


def lanes_to_road(img, left_fitx, right_fitx, ploty):
    pts_left = np.array([np.vstack(
            (left_fitx, ploty)).astype(np.float32).T[::-1]])
    pts_right = np.array([np.vstack(
            (right_fitx, ploty)).astype(np.float32).T])
    pts_left = cv2.perspectiveTransform(pts_left, M_inv).astype(np.int32)
    pts_right = cv2.perspectiveTransform(pts_right, M_inv).astype(np.int32)
    pts = np.hstack((pts_left, pts_right))
    lane_lines = np.zeros_like(img)
    lane_lines = cv2.fillPoly(lane_lines, [pts], (0, 100, 0))
    lane_lines = cv2.polylines(lane_lines, pts_left,
                               False, (0, 255, 255), 12)
    lane_lines = cv2.polylines(lane_lines, pts_right,
                               False, (0, 255, 255), 12)
    return lane_lines


#%%
# ----------------------------------------------------------------------
# Define a class to receive the characteristics of each lane detection
# ----------------------------------------------------------------------
class Line():
    def __init__(self):
        self.detected = False  # was the line detected in the last iteration?
        self.recent_xfitted = []  # x values of the last n fits of the line
        self.bestx = None  # average x values of the last n lines
        self.best_fit = [None, None, None]  # average coeffs of last n fits
        self.current_fit = []  # actual coeffs of recent fit
        self.radius = None  # radius of curvature of the line in some units
        self.offset = None  # distance of vehicle center from the line
        self.diffs = np.array([0, 0, 0], dtype='float')  # diff in fit coeffs
        self.allx = None  # x values for detected line pixels
        self.ally = None  # y values for detected line pixels


#%%
# ----------------------------------------------------------------------
# Define pipeline for all image processing
# ----------------------------------------------------------------------
Left = Line()
Right = Line()
n = 5


def process_image(img):
    warped = perspective_transform(img, mtx, dist, M)
    combined_binary, color_binary = grad_n_color_filter(warped)
    if Left.detected and Right.detected:
        Left.allx, Left.ally, Right.allx, Right.ally, lane_pixels = \
                find_lane_around_poly(combined_binary,
                                      Left.current_fit, Right.current_fit)
        if Left.allx.shape[0] >= 1000 and Right.allx.shape[0] >= 1000:
            Left.detected, Right.detected = True, True
        else:
            Left.detected, Right.detected = False, False
    if not Left.detected and not Right.detected:
        Left.allx, Left.ally, Right.allx, Right.ally, lane_pixels = \
                find_lane_sliding_windows(combined_binary)
        if Left.allx.shape[0] >= 1000 and Right.allx.shape[0] >= 1000:
            Left.detected, Right.detected = True, True
        else:
            Left.detected, Right.detected = False, False
    if Left.detected and Right.detected:
        Left.current_fit, Right.current_fit, ploty = \
                fit_poly(lane_pixels.shape,
                         Left.allx, Left.ally,
                         Right.allx, Right.ally)
        if len(Left.recent_xfitted) >= n and len(Right.recent_xfitted) >= n:
            Left.recent_xfitted.pop(0)
            Right.recent_xfitted.pop(0)
        Left.recent_xfitted.append(Left.current_fit)
        Right.recent_xfitted.append(Right.current_fit)
        Left.best_fit = np.average(Left.recent_xfitted, axis=0)
        Right.best_fit = np.average(Right.recent_xfitted, axis=0)
        Left.offset = measure_offset(lane_pixels.shape, Left.best_fit[-1])
        Right.offset = measure_offset(lane_pixels.shape, Right.best_fit[-1])
    else:
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    if any(Left.best_fit) and any(Right.best_fit):
        Left.radius = measure_curvature(ploty, Left.best_fit)
        Right.radius = measure_curvature(ploty, Right.best_fit)
        radius = ((Left.radius + Right.radius) / 2).astype(np.int32)
        if radius > 5000:
            radius = "straight"
        offset = round(Left.offset + Right.offset, 2)
        lane_lines = lanes_to_road(img, Left.best_fit, Right.best_fit, ploty)
        message1 = "Radius of curvature: {}".format(radius)
        message2 = "Offset from center: {}".format(offset)
        out_img = cv2.addWeighted(img, 1, lane_lines, 0.6, 0)
        cv2.putText(out_img, message1, (100, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), thickness=2)
        cv2.putText(out_img, message2, (100, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), thickness=2)
    else:
        out_img = img
    return out_img


#%%
# ----------------------------------------------------------------------
# Apply pipeline on test images to retrieve geometric data of lanes
# ----------------------------------------------------------------------
images = glob.glob('./test_images/test*.jpg')
Left = Line()
Right = Line()
n = 5

for fname in images:
    img = cv2.imread(fname)
    warped = perspective_transform(img, mtx, dist, M)
    combined_binary, color_binary = grad_n_color_filter(
            cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    leftx, lefty, rightx, righty, lane_pixels = \
        find_lane_sliding_windows(combined_binary)
    left_fitx, right_fitx, ploty = \
        fit_poly(lane_pixels.shape, leftx, lefty, rightx, righty)

    print(fname, ": lane lines plotted on bird eye image")
    pts_left = np.vstack((left_fitx, ploty)).astype(np.int32).T
    pts_right = np.vstack((right_fitx, ploty)).astype(np.int32).T
    warped = cv2.polylines(warped, [pts_left],
                           False, (255, 255, 0), 5)
    warped = cv2.polylines(warped, [pts_right],
                           False, (255, 255, 0), 5)
    print("lane lines searched with sliding windows")
    cv2.imwrite(r"./output_images/" +
                fname.split('\\')[-1].split('.')[0] +
                "_5birdeyelanes.jpg", warped)

    print(fname, ": pipeline run on image")
    # Apply pipeline and save output
    left_offset = measure_offset(lane_pixels.shape, left_fitx[-1])
    right_offset = measure_offset(lane_pixels.shape, right_fitx[-1])
    left_radius = measure_curvature(ploty, left_fitx)
    right_radius = measure_curvature(ploty, right_fitx)
    radius = ((left_radius + right_radius) / 2).astype(np.int32)
    if radius > 5000:
        radius = "straight"
    offset = round(left_offset + right_offset, 2)
    lane_lines = cv2.cvtColor(lanes_to_road(cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB), left_fitx, right_fitx, ploty),
            cv2.COLOR_BGR2RGB)
    message1 = "Radius of curvature: {}".format(radius)
    message2 = "Offset from center: {}".format(offset)
    out_img = cv2.addWeighted(img, 1, lane_lines, 0.6, 0)
    cv2.putText(out_img, message1, (100, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), thickness=2)
    cv2.putText(out_img, message2, (100, 140), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), thickness=2)
    cv2.imwrite(r"./output_images/" +
                fname.split('\\')[-1].split('.')[0] +
                "_result.jpg", out_img)

#%%
# ----------------------------------------------------------------------
# Process video images with pipeline
# ----------------------------------------------------------------------
video_titles = [("output.mp4", "project_video.mp4"),
                ("output_challenge.mp4", "challenge_video.mp4"),
                ("output_harder_challenge.mp4", "harder_challenge_video.mp4")]

for title in video_titles[:1]:
    Left = Line()
    Right = Line()
    output_video = title[0]
    clip1 = VideoFileClip(title[1])
    output_clip = clip1.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)
