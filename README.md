# Planar homography
Program that calculate the planar homography matrix. Implementation include the BRIEF descriptor, DoG keypoints detector, brute force keypoints matching and robust RANSAC estimation. 

## Introduction
In this implementation, I wrote a DoG keypoints detector and BRIEF descriptors and use hamming distance as metric for feature matching. In addition, robust RANSAC estimator is implemented to reject any outliners occured during the matching process.
Using above functions, I implemented a function that calcuate the planar homography and use it to generate simple panorama.

## Result
### Feature matching using BRIEF descriptor
![alt text](https://github.com/bilaer/Planar-homography/blob/master/match.jpg)

### RANSAC
![alt text](https://github.com/bilaer/Planar-homography/blob/master/ransac.jpg)

### Image Stitching
![alt text](https://github.com/bilaer/Planar-homography/blob/master/InclineL.jpg) ![Right Image](https://github.com/bilaer/Planar-homography/blob/master/InclineR.jpg)
![alt text](https://github.com/bilaer/Planar-homography/blob/master/final.jpg)

## Libraries
* Use [PIL](http://www.numpy.org/) to open, save and draw matching lines on image
* Use [Numpy](https://pillow.readthedocs.io/en/latest/) to do scientific calculation
* I use my own computer vision algorithms implementation [PythonCV](https://github.com/bilaer/PythonCV) to do gaussian smoothing and convolution.
* Since I haven't implement my own image warping function yet, I use [OpenCV](https://docs.opencv.org/2.4/index.html) to warp the image after I obtain the homography matrix from my program


## Images Source
Testing images are taken from http://16720.courses.cs.cmu.edu/lec.html
