 # Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration2_1chbrd.jpg "checkered chess board"
[image2]: ./output_images/calibration2_2undist.jpg "undistorted chess board"
[image3]: ./test_images/straight_lines2.jpg "distorted road"
[image4]: ./output_images/straight_lines2_1roi.jpg "undistorted road"
[image5]: ./output_images/straight_lines2_1warped.jpg "top-down perspective"
[image6]: ./output_images/test5_1warped.jpg "before binary thresholding"
[image7]: ./output_images/test5_2color.jpg "after binary thresholding"
[image8]: ./output_images/test3_3slidingwindows.jpg "sliding window based search"
[image9]: ./output_images/test3_4polyband.jpg "polynomial band based search"
[image10]: ./output_images/test3_5birdeyelanes.jpg "resulting polynomials"
[image11]: ./output_images/test3_result.jpg "results transformed back to road image"
[video1]: ./output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### 1. Writeup / README

#### 1.1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! Although I provided examples for all steps, I encourage the viewer to check the ./output_images for all the calibration images, and the road test images. They provide a quick way to grasp the whole of the process described below.

### 2. Camera Calibration

#### 2.1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell (line 13) of the file called `project.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
![alt text][image2]

### 3. Pipeline (single images)

#### 3.1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image3]
![alt text][image4]
As the same camera was used to capture the chessboard images, we can apply the already determined camera calibration and distortion coefficients (line 116). The same openCV function can be used as well. I drew a red rectangle near the region of interest, as we will use it later.

#### 3.2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 136 through 161 in the file `project.py`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as the Matrix (`M`) which could be obtained by source and and destination points and the `getPerspectiveTransform` openCV function .  I chose to hardcode the source and destination points in the following manner:

```python
p = 300
src = np.float32([[578, 460], [706, 460], [1120, 720], [190, 720]])
dst = np.float32([[p, 0], [img_size[0]-p, 0],
                  [img_size[0]-p, img_size[1]], [p, img_size[1]]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image (line 156 to 161) The `src` rectangle can be found above, and the `dst` rectangle below. I finetuned the parameters to get parallel lane lines and the relevant portion of the road.

![alt text][image5]

#### 3.3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

As learned during the lessons, it proved to be useful to convert the image to HLS representation. I used a combination of color thresholds, consisting of an 'H' and an 'S' channel filtering. The 'S' channel in itself was working okay, so I only used the 'H' channel as a broad filter to auxiliate the selection, when the 'S' became noisy. The gradient threshold was also useful to detect otherwise non-recognised lane line segments (thresholding steps at lines 172 through 197 in `project.py`).  Here's an example of my output for this step, which seems to be the most difficult image, with a lot of brightness changes and shadows. The 'S' channel bursts a lot of fake detections, marked blue, but with the help of the 'H' channel, marked red, the common area, resulting in pink, resembles the lane line direction well:

![alt text][image6]
![alt text][image7]

The gradient thresholding, marked green, seemingly discovers other parts of the lane lines.

#### 3.4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
To approximate the lane pixels with a second order polynomial, without prior knowledge of the lane line positions, I used the sliding window technique, line 224 to 294. This means, that the image is divided in two, and the search is started at the bottom, based on the histogram peak values. Then, a customizable sized rectangle is used to cover the continuity of those pixels. When the guess is confident, meaning a significant number of pixels are discovered, the center of the next rectangle is shifted to the mean position of the pixels. After the lane pixels have been further narrowed down, the pixel clouds are approximated with second order polynomial functions, with the following result:

![alt text][image8]

When we observe a number of subsequent images, we can store information about the whereabouts of the lane lines. This way, the search can be sped up, if we use the previous polynomial and search within a margin around it for the lane pixels (line 356 to 385). This process is depicted in the following image:

![alt text][image9]

#### 3.5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I measured the curvature based on formula derived based on the derivative of the second order polyinomial function, in lines 439 through 446 in my code in `project.py`. To acquire the radius value at the point closest to the car, the function is evaluated with the maximum y value. For the number to be meaningful, the calculations are converted with the coefficient needed to convert to real distance units.

```python
curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```

For the offset from the center of the road, I have written a function to return the difference between the image center and the identified lane line starts (line 449 to 451). The sum of these values results in the signed offset value, again converted with the coefficient (line 538).

```python
(last_x - img_shape[1]/2) * xm_per_pix
```

#### 3.6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

When the polynomials are acquired, the warped image can be compared with the positions:
![alt text][image10]

In order to make use of the positions, we have to transform the lines back, with the reverse perspective transformation. I implemented this step in lines 454 through 468 in my code in `project.py` in the function `lanes_to_road()`.  Here is an example of my result on a test image:

![alt text][image11]
---

### 4. Pipeline (video)

#### 4.1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.mp4)

---

### 5. Discussion

#### 5.1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. The presented technologies can unfortunately be prone to the various light conditions and sharp turns, that can be seen in the challenge videos, as they change the respective position of the lines a great deal. Mostly I focused on implementing the techniques taught during the lessons, but there were some hints to make the system more robust.

The most important improvement was to implement a low pass filter to average the measurements and not react to the untrustable results of a single frame. Averaging out 5 different values seemed a good start (line 495). The above presented second searching method, also has a similar effect, as in taking into account the history of measurements. Practically we know that the line cannot change drastically from frame to frame, therefore we can implement that knowledge in such ways.

I was not fully satisfied with the accuracy of the curvature and offset measurements, as they were based on rough estimations and overly generic assumptions (plane ground). A next step could be to utilize the pitch angle of the vehicle to compensate for the various angles the road is viewed. Another improvement would be to update filter parameters according to the mean brightness level, according to previously optimized lookup tables. Adaptive histogram equalization could be used, to make the edges more prominent.
