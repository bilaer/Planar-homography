from cv import image
from cv import Filter
import numpy as np
import PIL
import random
import math
import itertools
import cv2

# DoG descriptor that is used to calculate and store the key points in a 
# given image
class DoGDecteor(object):
    def __init__(self, image, height, width, contrastTh, rTh):
        self.img = image
        self.imgWidth = width
        self.height = height
        self.appFilter = Filter()
        self.level = tuple([-1, 0, 1, 2, 3, 4])
        self.k = 2**0.5
        self.contrastTh = contrastTh
        self.rTh = rTh
        self.sigma0 = 1
        self.gauPyramid, self.dogPyramid = self.CalGauAndDoGPraymid(image, height, width)
        self.pattern = self.GetExtrema(image, height, width, contrastTh, rTh)

    # Return a RxCxW matrix where RxC is the size of the image and 
    # W is the total number of the pyramids
    def CalGauAndDoGPraymid(self, targetImage, height, width):
        gauPyramid = np.zeros((height, width, len(self.level)))
        dogPyramid = np.zeros((height, width, len(self.level) - 1))
        
        # Obtain the gaussian smoothing image by with different sigma value
        print("Gaussian smoothing...")
        # Normalize the image so that the scale of value is between 0 and 1
        # Which is the correspondent to the BRIEF paper
        targetImage = self.normalize(targetImage)
        for i in range(len(self.level)):
            sigma = self.sigma0*(self.k**self.level[i])
            gaul = np.array(self.appFilter.GaussianSmoothing(targetImage, width, height, sigma))
            gauPyramid[:, :, i] = gaul
        # Obtain dog pyramid by substrating gaussian smoothing image with image 
        # on the last level

        print("Calculating the difference...")
        for i in range(1, len(self.level)):
            dogPyramid[:, :, i - 1] = gauPyramid[:, :, i] - gauPyramid[:, :, i - 1]
        return gauPyramid, dogPyramid

    def normalize(self, image):
        minValue = np.amin(image)
        valueRange = np.amax(image) - minValue
        result = np.zeros(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = image[i][j] - minValue
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i][j] = image[i][j]/valueRange
        return result

    # Edge suppression for remove less important interest points
    def GetExtrema(self, image, height, width, conTh, rTh):
        Dx, Dy, Dxx, Dxy, Dyx, Dyy = (np.zeros((height, width, len(self.level) - 1)) for i in range(6))
        for i in range(len(self.level) - 1):
            # Add padding
            Dy[:, :, i], Dx[:, :, i] = np.gradient(self.dogPyramid[:, :, i])

            # Get the second derivative of image
            print("Calculating the second derivatives of %d level of DoG" %i)
            Dxy[:, :, i], Dxx[:, :, i] = np.gradient(Dx[:, :, i])
            Dyy[:, :, i], Dyx[:, :, i] = np.gradient(Dy[:, :, i])


        # Result is a set of keypoints that satisfy the demands, i.e it is local extrema and has response
        # larger than contrast threshold and principal curvature larger than rTh
        print("Thresholding")
        result = []
        count = 0
        for i in range(height):
            for j in range(width):
                for k in range(0, len(self.level) - 1):
                    # Get the neighbors of current scale and scale below and above current pyramid
                    currentNeb = self.dogPyramid[max(0, i - 1):min(height, i + 2), max(0, j - 1):min(width, j + 2), k]
                    if k - 1 < 0:
                        lowNeb = None
                        heiNeb = self.dogPyramid[max(0, i - 1):min(height, i + 2), max(0, j - 1):min(width, j + 2),
                                 k + 1]
                    elif k + 1 == len(self.level) - 1:
                        heiNeb = None
                        lowNeb = self.dogPyramid[max(0, i - 1):min(height, i + 2),
                                                 max(0, j - 1):min(width, j + 2), k - 1]
                    else:
                        currentNeb = self.dogPyramid[max(0, i - 1):min(height, i + 2),
                                                     max(0, j - 1):min(width, j + 2), k]
                        lowNeb = self.dogPyramid[max(0, i - 1):min(height, i + 2),
                                                 max(0, j - 1):min(width, j + 2), k - 1]
                        heiNeb = self.dogPyramid[max(0, i - 1):min(height, i + 2),
                                                 max(0, j - 1):min(width, j + 2), k + 1]
                    # Test if the current point is local extrema
                    # If current point is a local extrema, then test if its principal curvature exceed
                    # the principal curvature. In addition, test if the DoG pyramid magnitude exceed
                    # the contrast threshold
                    midi, midj = self.getMid(currentNeb, i, j, height, width)
                    if self.isLocalExtrema(lowNeb, heiNeb, currentNeb, midi, midj) and currentNeb[midi][midj] > conTh:
                        # Calculate the principal curvature of current point and compare it with the
                        if self.CheckPrincipalCurvature(Dxx[i][j][k], Dxy[i][j][k], Dyx[i][j][k], Dyy[i][j][k]):
                            result.append((i, j, k))
                        else:
                            count = count + 1
        print("filter out %d points" %count)

        return np.array(result)

    def getMid(self, array, i, j, height, width):
        midi, midj = 1, 1
        if i - 1 < 0:
            midi = 0
        if j - 1 < 0:
            midj = 0
        return midi, midj

    def isLocalExtrema(self, lowNeb, hiNeb, curNeb, midi, midj):
        if not lowNeb is None and hiNeb is None:
            if np.amax(lowNeb) < curNeb[midi][midj] and np.amax(curNeb) == curNeb[midi][midj]:
                temp = np.sort(np.reshape(curNeb, (curNeb.shape[0]*curNeb.shape[1])))
                if temp[-2] != np.amax(curNeb):
                    return True
                else:
                    return False
            elif np.amin(lowNeb) > curNeb[midi][midj] and np.amin(curNeb) == curNeb[midi][midj]:
                temp = np.sort(np.reshape(curNeb, (curNeb.shape[0] * curNeb.shape[1])))
                if temp[1] != np.amin(curNeb):
                    return True
                else:
                    return False
            else:
                return False
        if not hiNeb is None and lowNeb is None:
            if np.amax(hiNeb) < curNeb[midi][midj] and np.amax(curNeb) == curNeb[midi][midj]:
                temp = np.sort(np.reshape(curNeb, (curNeb.shape[0]*curNeb.shape[1])))
                if temp[-2] != np.amax(curNeb):
                    return True
                else:
                    return False
            elif np.amin(hiNeb) > curNeb[midi][midj] and np.amin(curNeb) == curNeb[midi][midj]:
                temp = np.sort(np.reshape(curNeb, (curNeb.shape[0] * curNeb.shape[1])))
                if temp[1] != np.amin(curNeb):
                    return True
                else:
                    return False
            else:
                return False
        else:
            if np.amax(lowNeb) < curNeb[midi][midj] and np.amax(hiNeb) < curNeb[midi][midj] and \
               np.amax(curNeb) == curNeb[midi][midj]:
                temp = np.sort(np.reshape(curNeb, (curNeb.shape[0]*curNeb.shape[1])))
                if temp[-2] != np.amax(curNeb):
                    return True
                else:
                    return False
            elif np.amin(lowNeb) > curNeb[midi][midj] and np.amin(hiNeb) > curNeb[midi][midj] and \
                 np.amin(curNeb) == curNeb[midi][midj]:
                temp = np.sort(np.reshape(curNeb, (curNeb.shape[0] * curNeb.shape[1])))
                if temp[1] != np.amin(curNeb):
                    return True
                else:
                    return False
            else:
                return False

    def drawKeyPoints(self, image, isGrey=False):
        r = 1
        color = (255, 0, 0)
        assert(len(self.pattern) != 0)
        draw = PIL.ImageDraw.Draw(image)
        for points in self.pattern:
            y, x = points[0], points[1]
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
        image.show()
        del draw

    # Return the principal curvature of a point given its second derivatives
    def CheckPrincipalCurvature(self, Dxx, Dxy, Dyx, Dyy):
        Hessian = np.array([[Dxx, Dxy],[Dyx, Dyy]])
        R = (np.trace(Hessian)**2)/np.linalg.det(Hessian)
        return R < self.rTh

    def GetKeyPoints(self):
        return self.pattern

    def GetKeyPointByIndex(self, index):
        assert(index < len(self.pattern))
        return self.pattern[index]

    def GetKeyPointX(self, index):
        assert(index < len(self.pattern))
        return self.pattern[index][1]

    def GetKeyPointY(self, index):
        assert(index < len(self.pattern))
        return self.pattern[index][0]

    def GetKeyPointGauLevel(self, index):
        assert(index < len(self.pattern))
        return self.pattern[index][2]

    # Get the value of pixel of certain pixels at given level of DoG pyramid
    def GetValueOfGau(self, x, y, level):
        assert(level < len(self.gauPyramid))
        return self.gauPyramid[y][x][level]




# FAST Detector
class FASTDetector(object):
    def __init__(self):
        pass

# The patterns that BRIEF descriptor will used in collecting the samples
class Pattern(object):
    __slots__ = ["xIndex", "yIndex"]

    def __init__(self, width, nbits):
        self.xIndex, self.yIndex = self.makeTestPattern(width, nbits)

    # Randomly generate the sampling patterns on 9X9 area
    # This patterns will be apply to all the keypoints that is detected
    # Will use uniform distribution, the variance is equal to S**2/25, according to
    # the original paper where S is the size of image patch
    def makeTestPattern(self, width, nbits):
        xIndex, yIndex = [], []
        # Randomly choose nbits from gaussian distribution points for comparision
        for i in range(nbits):
            xx, xy = np.random.uniform(-width/2, width/2, 2)
            xIndex.append((xx, xy))
            yx, yy = np.random.uniform(-width/2, width/2, 2)
            yIndex.append((yx, yy))
        return (xIndex, yIndex)

    def getPattern(self):
        return (self.xIndex, self.yIndex)

    def getLen(self):
        return len(self.xIndex)

    def getXCoord(self, i):
        assert(i < len(self.xIndex))
        return self.xIndex[i]

    def getYCoord(self, i):
        assert(i < len(self.yIndex))
        return self.yIndex[i]

# BRIEF descriptor Notice that BRIEF is not rotation invariant
class BRIEF(object):
    __slots__ = ["width", "nbits", "pattern", "descriptor"]

    def __init__(self, width, nbits, detector, pattern, imageHeight, imageWidth):
        self.width = width
        self.nbits = nbits
        self.pattern = pattern
        # Descriptor is a dictionary of which the locations and level of gaussian pyramid is the key and
        # the descriptor of that point is the value
        self.descriptor = self.computeBRIEF(detector, imageWidth, imageHeight)

    # Return the descriptor given certain point
    def getDescriptorByPoint(self, point):
        return self.descriptor[point]

    # Return the descriptor of all points
    def getDescriptor(self):
        return self.descriptor

    # Base on the pattern randomly generated from makeTestPattern function,
    # apply this pattern to every keypoints detected by detector
    def computeBRIEF(self, det, imageWidth, imageHeight):
        result = dict()
        #xIndex, yIndex = self.pattern.getPattern()
        for i in range(len(det.GetKeyPoints())):
            temp = np.zeros(self.nbits)
            for j in range(self.pattern.getLen()):
                (xx, xy) = map(int, self.pattern.getXCoord(j))
                (yx, yy) = map(int, self.pattern.getYCoord(j))
                pairXx, pairXy = det.GetKeyPointX(i) + xx, det.GetKeyPointY(i) + xy
                pairYx, pairYy = det.GetKeyPointX(i) + yx, det.GetKeyPointY(i) + yy
                GauLevel = det.GetKeyPointGauLevel(i)
                # Check if this key point is valid for computing the brief, e.g not on the edge of image
                if pairXx >= 0 and pairXx < imageWidth and pairXy >= 0 and pairXy < imageHeight and \
                   pairYx >= 0 and pairYx < imageWidth and pairYy >= 0 and pairYy < imageHeight:
                    if det.GetValueOfGau(pairXx, pairXy, GauLevel) > det.GetValueOfGau(pairYx, pairYy, GauLevel):
                        temp[j] = 1
                    else:
                        temp[j] = 0
                else:
                    temp = None
                    break
            if not temp is None:
                result[tuple(det.GetKeyPointByIndex(i))] = temp
        return result

    # Function use to calculate the similarity between two descriptor
    # Descriptors are represented as vectors
    # Return the number of different bits in two vector
    def hammingDistance(self, descriptorOne, descriptorTwo):
        count = 0
        for i in range(len(descriptorOne)):
            if descriptorOne[i] != descriptorTwo[i]:
                count = count + 1
        return count

    # Brutal force matching method
    # Calculating the difference of descriptor of matching points
    # return a list of point that satisfy the matching threshold
    # keypointsTwo is dictionary of which the key is location of points
    # and value is the descriptor of this point
    # Return a dictionary of matching points, the structure of returning point
    # is as follow: {(y, x, level):(y1, x1, level)...}
    def matching(self, keypointsTwo, errorTh):
        result = dict()
        hasAdd = set()
        for pointOne in self.descriptor:
            missMatch = dict()
            for pointTwo in keypointsTwo:
                if pointTwo not in hasAdd:
                    distance = self.hammingDistance(self.descriptor[pointOne], keypointsTwo[pointTwo])
                    missMatch[distance] = pointTwo

            # Get the match point with least hamming distance
            if len(hasAdd) == len(keypointsTwo):
                return result
            matchPoint = missMatch[(sorted(missMatch.keys())[0])]
            if sorted(missMatch.keys())[0] < errorTh*self.nbits:
                result[pointOne] = matchPoint
            hasAdd.add(matchPoint)
        return result

###########################################################################
#                         Homography functions                            #
###########################################################################

# Function that use SVD to calculate the homography matrix
# correspondences must be at least four pair points
# [[(x1, y1),(x2, y2)]...]
# this function normalize the homography matrixso that the
# solutiion is more stable The transformation matrix is given by
# T = s*[[1, 0, - mean(u)], [0, 1, -mean(v)], [0, 0, 1/s]] where
# X1 is before transformation, X2 is after transformation
# s is given by s = 2**0.5*n/sum((ui - mean(u)**2 + (vi - mean(v))**2)**0.5
# reference: http://www.ele.puc-rio.br/~visao/Homographies.pdf
def calHomography(correspondences):
    assert(len(correspondences) >= 4)
    meanX1, meanY1, meanX2, meanY2 = 0, 0, 0, 0
    for point in correspondences:
        meanX1 = point[0][1] + meanX1
        meanX2 = point[1][1] + meanX2
        meanY1 = point[0][0] + meanY1
        meanY2 = point[1][0] + meanY2
    meanX1, meanY1 = meanX1 / len(correspondences), meanY1 / len(correspondences)
    meanX2, meanY2 = meanX2 / len(correspondences), meanY2 / len(correspondences)

    # Calculate the scale factor
    s1, s2 = 0, 0
    for point in correspondences:
        s1 = s1 + ((point[0][1] - meanX1)**2 + (point[0][0] - meanY1)**2)**0.5
        s2 = s2 + ((point[1][1] - meanX2)**2 + (point[1][0] - meanY2)**2)**0.5
    s1 = (2**0.5)*len(correspondences)/s1
    s2 = (2**0.5)*len(correspondences)/s2

    # Get the transformation matrix
    T1 = s1 * np.array([[1, 0, -meanX1], [0, 1, -meanY1], [0, 0, 1 / s1]])
    T2 = s2 * np.array([[1, 0, -meanX2], [0, 1, -meanY2], [0, 0, 1 / s2]])


    # Calculate the homography matrix of normalized the coordinates
    A = []
    for i in range(len(correspondences)):
        # correspondences has form [[(y, x, level),(y, x, level)]....]
        norm1 = np.array([[correspondences[i][0][1]], [correspondences[i][0][0]], [1]])
        norm2 = np.array([[correspondences[i][1][1]], [correspondences[i][1][0]], [1]])
        p1 = np.matmul(T1, norm1)
        p2 = np.matmul(T2, norm2)
        x1, y1, x2, y2 = p1[0][0], p1[1][0], p2[0][0], p2[1][0]
        A = A + [[ 0, 0, 0, -x2, -y2, -1,  x2*y1, y2*y1, y1],
                 [ x2, y2, 1, 0, 0, 0, -x2*x1, -y2*x1, -x1]]

    # Get transpose A multiple A
    A = np.array(A)
    B = np.matmul(np.transpose(A), A)
    # Get the SVD decomposition of ATA
    u, s, vh = np.linalg.svd(B)
    # The solution is the smallest eigenvalue and it is corresponding
    # eigenvector
    # solution has form h1, h2 ... h9 and reshape the vector into
    # a matrix
    H = np.reshape(vh[8, :], (3, 3))

    # Multiple transformation matrix to get true homography
    H = np.matmul(np.matmul(np.linalg.inv(T1), H), T2)

    return H

# Function that turns the match dictionary into the a array
def parseMatchData(matches):
    result = []
    for point in matches:
        result.append([list(point[:2]), list(matches[point][:2])])
    return result

# Algorithm that help remove the outliners of matches
class RANSAC(object):
    def __init__(self, matches):
        self.matches = matches

    # Use the given homography to calculate the a set
    # of inliners given threshold.
    def calInliner(self, homographyMat):
        result = set()
        disList = list()
        thr = 20
        for point in self.matches:
            normp1 = np.array([[point[0][1]], [point[0][0]], [1]])
            normp2 = np.array([[point[1][1]], [point[1][0]], [1]])
            exmpoint = np.matmul(homographyMat, normp2)
            exmpoint1 = np.matmul(np.linalg.inv(homographyMat), normp1)
            exmpoint = np.array([[exmpoint[0][0]/exmpoint[2][0]], [exmpoint[1][0]/exmpoint[2][0]]])
            exmpoint1 = np.array([[exmpoint1[0][0]/exmpoint1[2][0]], [exmpoint1[1][0]/exmpoint1[2][0]]])
            normp1 = np.array([[point[0][1]], [point[0][0]]])
            normp2 = np.array([[point[1][1]], [point[1][0]]])
            dis = np.linalg.norm(exmpoint - normp1) + np.linalg.norm(exmpoint1 - normp2)
            if dis < thr:
                result.add(tuple(tuple(cor) for cor in point))
                disList.append(dis)

        # Calculate the threshold
        mean, std = 0, 10**9
        if len(disList) != 0:
            mean = sum(list(disList))/len(disList)
            std = sum([(x - mean)**2 for x in list(disList)])/len(disList)
        # Get the inliners
        return result, std

    # Compute the best Homography matrix using RANSAC algortihm
    # matches is the dictionary of paired keypoints
    # Here I use adaptive RANSAC
    # reference: http://www.uio.no/studier/emner/matnat/its/UNIK4690/v16/forelesninger/
    # lecture_4_3-estimating-homographies-from-feature-correspondences.pdf, page 5
    def computeHRANSAC(self):
        # Setup the initial parameters
        p = 0.99
        minStd = 10**5
        N = 10**5
        iter = 0
        prevInliner = set()
        # Get the combination of all correspondences
        while iter < N: #and len(com) != 0:
            # Randomly select four correspondences
            ranCorr = []
            hasAdd = set()
            count = 0
            while count < 4:
                index = random.randint(0, len(self.matches) - 1)
                if index not in hasAdd:
                    ranCorr.append(self.matches[index])
                    hasAdd.add(index)
                    count = count + 1

            # Calculate the homography based on given correspondences
            h = calHomography(ranCorr)

            # Get the inliner
            inliner, std = self.calInliner(h)

            # If the the size of returned inliner is larger than currently largest inliners,
            # or if the have the same size but the deviation is smaller, update the inliner set
            if len(inliner) > len(prevInliner) or \
               (len(prevInliner) != 0 and len(inliner) == len(prevInliner) and std < minStd):
                print("current num of inliner: %d" %len(inliner))
                print("total: %d" %len(self.matches))
                # Update the max inliners
                prevInliner = inliner
                # Update the min std of largest the inliners
                minStd = std
                # Calculate the epsilon and update the iteration N
                epsilon = 1 - len(inliner)/len(self.matches)
                N = int(math.log(1 - p)/math.log(1 - (1 - epsilon)**4))
                iter = -1
            iter = iter + 1
            print("iter: %d, N: %d, cur: %d" %(iter, N, len(prevInliner)))

        # Recalculate the H using all the inliners
        H = calHomography(list(prevInliner))
        # Look for additional matches
        reca, std = self.calInliner(H)
        return list(prevInliner), list(reca), H


# Function that is used to connect matched points in two images by drawing lines
def getMatch(imageOnePath, imageTwoPath, conTh, rTh, pattern, width, nbits, errorTh, isDraw=False):
    # Initialize DoG detectors for two images
    print("get key points")
    imageOne = image(imageOnePath)
    imageTwo = image(imageTwoPath)
    detOne = DoGDecteor(imageOne.GetGrey(), imageOne.GetHeight(), imageOne.GetWidth(), conTh, rTh)
    detTwo = DoGDecteor(imageTwo.GetGrey(), imageTwo.GetHeight(), imageTwo.GetWidth(), conTh, rTh)
    print("# of key points in first image %d" %len(detOne.GetKeyPoints()))
    print("# of key points in second image %d" %len(detTwo.GetKeyPoints()))
    # Get the parameters of two images
    imageOneHeight, imageOneWidth = imageOne.GetHeight(), imageOne.GetWidth()
    imageTwoHeight, imageTwoWidth = imageTwo.GetHeight(), imageTwo.GetWidth()
    # Get the descriptors of keypoints of two images
    print("register descriptors")
    briefOne = BRIEF(width, nbits, detOne, pattern, imageOne.GetHeight(), imageOne.GetWidth())
    briefTwo = BRIEF(width, nbits, detTwo, pattern, imageTwo.GetHeight(), imageTwo.GetWidth())

    # Get the matching points and draw it
    match = briefOne.matching(briefTwo.getDescriptor(), errorTh)
    matchTwo = match
    #else:
    #    match = briefTwo.matching(briefOne.getDescriptor(), errorTh)

    # Initialize image for display matching
    if isDraw:
        print("draw Key points of images")
        #testOne = PIL.Image.open(imageOnePath)
        #testTwo = PIL.Image.open(imageTwoPath)
        #detOne.drawKeyPoints(testOne)
        #detTwo.drawKeyPoints(testTwo)

        # Draw initial matches
        print("draw matching")
        matchImage = np.zeros((max(imageOneHeight, imageTwoHeight), (imageOneWidth + imageTwoWidth), 3))
        matchImage[:imageOneHeight, :imageOneWidth, 0] = imageOne.GetGrey()
        matchImage[:imageOneHeight, :imageOneWidth, 1] = imageOne.GetGrey()
        matchImage[:imageOneHeight, :imageOneWidth, 2] = imageOne.GetGrey()
        matchImage[:imageTwoHeight, imageOneWidth:, 0] = imageTwo.GetGrey()
        matchImage[:imageTwoHeight, imageOneWidth:, 1] = imageTwo.GetGrey()
        matchImage[:imageTwoHeight, imageOneWidth:, 2] = imageTwo.GetGrey()
        result = PIL.Image.fromarray(np.uint8(matchImage))
        draw = PIL.ImageDraw.Draw(result)
        for pointOne in match:
            # Draw
            (x1, y1) = pointOne[1], pointOne[0]
            (x2, y2) = match[pointOne][1] + imageOneWidth, match[pointOne][0]
            color = []
            for i in range(3):
                color.append(random.randint(0, 255))
            print("current match point", pointOne, match[pointOne])
            draw.line((x1, y1, x2, y2), fill=tuple(color))
            draw.ellipse((x1 - 1, y1 - 1, x1 + 1, y1 + 1), fill=tuple(color))
            draw.ellipse((x2 - 1, y2 - 1, x2 + 1, y2 + 1), fill=tuple(color))
        result.show()
        #result.save("match.jpg")
        del draw

    # Draw inliners
    match = parseMatchData(match)
    ransc = RANSAC(match)
    inliner, reca, H = ransc.computeHRANSAC()
    if isDraw:
        result = PIL.Image.fromarray(np.uint8(matchImage))
        draw = PIL.ImageDraw.Draw(result)
        for point in match:
            (x1, y1) = point[0][1], point[0][0]
            (x2, y2) = point[1][1] + imageOneWidth, point[1][0]
            color = [255, 0, 0]
            draw.line((x1, y1, x2, y2), fill=tuple(color))
            draw.ellipse((x1 - 1, y1 - 1, x1 + 1, y1 + 1), fill=tuple(color))
            draw.ellipse((x2 - 1, y2 - 1, x2 + 1, y2 + 1), fill=tuple(color))
        for point in inliner:
            (x1, y1) = point[0][1], point[0][0]
            (x2, y2) = point[1][1] + imageOneWidth, point[1][0]
            color = [0, 255, 0]
            draw.line((x1, y1, x2, y2), fill=tuple(color))
            draw.ellipse((x1 - 1, y1 - 1, x1 + 1, y1 + 1), fill=tuple(color))
            draw.ellipse((x2 - 1, y2 - 1, x2 + 1, y2 + 1), fill=tuple(color))

        result.show()
        result.save("ransac.jpg")
        del draw
    return H, reca

# Create a panorama using homography matrix calculated above
def panorama(imageOneName, imageTwoName):
    # Calculate the homography matrix
    pattern = Pattern(9, 256)
    H, inliner = getMatch(imageOneName, imageTwoName, 0.03, 12, pattern, 9, 256, 0.07, True)

    # Wrap image
    img1 = cv2.imread(imageOneName, 0)
    output = cv2.warpPerspective(img1, np.linalg.inv(H), (500, 500))
    arrayOutput = np.array(output)

    # Calculate the transformation of keypoints
    afterTransMatch = dict()
    for point in inliner:
        pt = np.array([[point[0][1]], [point[0][0]], [1]])
        newPt = np.matmul(np.linalg.inv(H), pt)
        newPt = tuple([int(round(newPt[1][0] / newPt[2][0])), int(round(newPt[0][0] / newPt[2][0]))])
        afterTransMatch[newPt] = tuple([point[1][0], point[1][1]])

    # Intialize the final panorama image
    imageOne = image(imageTwoName)
    imageTwoHeight, imageTwoWidth = arrayOutput.shape[0], arrayOutput.shape[1]
    imageOneHeight, imageOneWidth = imageOne.GetHeight(), imageOne.GetWidth()
    matchImage = np.zeros((max(imageOneHeight, imageTwoHeight), (imageOneWidth + imageTwoWidth), 3))
    matchImage[:imageOneHeight, :imageOneWidth, 0] = imageOne.GetGrey()
    matchImage[:imageOneHeight, :imageOneWidth, 1] = imageOne.GetGrey()
    matchImage[:imageOneHeight, :imageOneWidth, 2] = imageOne.GetGrey()
    matchImage[:imageTwoHeight, imageOneWidth:, 0] = arrayOutput
    matchImage[:imageTwoHeight, imageOneWidth:, 1] = arrayOutput
    matchImage[:imageTwoHeight, imageOneWidth:, 2] = arrayOutput

    # Get the move parameter
    move = []
    for point in afterTransMatch:
        (x1, y1) = afterTransMatch[point][1] + imageOneWidth, afterTransMatch[point][0]
        (x2, y2) = point[1], point[0]
        move = [x1 - x2, y1 - y2]
        break

    imageOneGrey = imageOne.GetGrey()
    for i in range(imageOneHeight):
        for j in range(imageOneWidth):
            matchImage[i + move[1]][j + move[0]] = imageOneGrey[i][j]

    final = matchImage[:imageOneHeight, move[0]:]
    final = PIL.Image.fromarray(np.uint8(final))
    final.show()
    final.save("final.jpg")










