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
Left Image:
![alt text](https://github.com/bilaer/Planar-homography/blob/master/InclineL.jpg) 

Right Image:
![Right Image](https://github.com/bilaer/Planar-homography/blob/master/InclineR.jpg)

Result image:
![alt text](https://github.com/bilaer/Planar-homography/blob/master/final.jpg)




## Libraries

## Images Source
