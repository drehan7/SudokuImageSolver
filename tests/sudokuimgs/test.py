import cv2
import numpy as np
import operator
import os
import sys

MIN_CONTOUR_AREA = 500
MAX_CONTOUR_AREA = 7000

RZW, RZH = 20, 30

# Organize points into 
# topleft, topright, bottomright, bottomleft
def order_points(pts):

    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))


    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0,0], # topleft
        [maxWidth - 1, 0], # topright
        [maxWidth - 1, maxHeight - 1], # bottomright
        [0, maxHeight -1]], # bottomleft
        dtype="float32")


    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def find_board(imagepath):
    img = cv2.imread(imagepath, 0)
    gaus = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)

    cv2.imshow("Original image", img)
    cv2.waitKey(0)

    thresh = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    thresh = cv2.bitwise_not(thresh)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # biggest contour
    maxapprox = None
    maxarea = 0
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4:
            maxapprox = approx
            maxarea = max(maxarea, area)
            
    contour = max(contours, key = cv2.contourArea)

    out = img.copy()
    cv2.drawContours(out, [contour], -1, (0,255,0), 3)

    ap = img.copy()

    # finds polygon on closed contour
    approx = cv2.approxPolyDP(contour, 0.010 * cv2.arcLength(contour, True), True)

    n = approx.ravel()

    [intx, inty, intw, inth] = cv2.boundingRect(contour)
    pts1 = np.float32([
        [164, 507],
        [989, 484],
        [151, 1392],
        [1051, 1372]
        ])

    pts2 = np.float32([
        [0,0],
        [intx+intw, 0],
        [0, inty+inth],
        [intx+intw, inty+inth]
        ])

    M = cv2.getPerspectiveTransform( pts1, pts2 )

    warp = img.copy()

    dst = cv2.warpPerspective(img, M, (intx+intw, inty+inth))

    return dst


class ContourWithData():

    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0

    fltArea = 0.0

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intx, inty, intw, inth] = self.boundingRect
        self.intRectX = intx
        self.intRectY = inty
        self.intRectWidth = intw
        self.intRectHeight = inth

    def checkIfContourValid(self):
        if self.fltArea < MIN_CONTOUR_AREA or self.fltArea > MAX_CONTOUR_AREA: return False
        return True

class ContoursWithData():
    contours = []
    avg_area = -1
    avg_height = -1
    avg_width = -1

    def __init__(self, _contours):
        self.contours = _contours
        self.calc_avg_area()

    def calc_avg_area(self):
        if len(self.contours) == 0:
            print("contours empty")
            return

        area_sum = 0
        height_sum = 0
        width_sum = 0
        for contour in self.contours:
            _area = contour.intRectWidth * contour.intRectHeight
            area_sum += _area
            height_sum += contour.intRectHeight
            width_sum += contour.intRectWidth
        
        self.avg_area = (area_sum / len(self.contours))
        self.avg_height = (height_sum / len(self.contours))
        self.avg_width = (width_sum / len(self.contours))


    def within_area(self, contour):
        pass

    def set_contours(self, contours_arr):
        contours = contours_arr

def main():
    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("../../src/models/classifications.txt", np.float32)
        print("Successfully loaded classifications")
    except:
        print("Error")
        sys.exit(1)

    try:
        npaFlattenedImages = np.loadtxt("../../src/models/flattened_images.txt", np.float32)
        print("Successfully loaded flattened images values")
    except:
        print("Error")
        sys.exit(1)

    npaClassifications = npaClassifications.reshape((npaClassifications.size), 1)

    kNearest = cv2.ml.KNearest_create()

    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    # kNearest.save("KNN.bin")

    imgTestingNumbers = find_board("../../assets/sudoku2.jpeg")

    if imgTestingNumbers is None:
        print("ERROR")
        sys.exit(1)

    imgGray = imgTestingNumbers.copy()

    imgBlurred = cv2.bilateralFilter(imgGray, 10, 100, 75)

    _, imgThresh = cv2.threshold(imgBlurred, 135, 255, cv2.THRESH_BINARY)

    imgThreshCopy = imgThresh.copy()

    cv2.imshow("IMTHRES", imgThresh)
    intKey = cv2.waitKey(0)
    if intKey == 27: exit(0)

    npaContours, npaHier = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for npaContour in npaContours:
        contourWithData = ContourWithData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourValid():
            validContoursWithData.append(contourWithData)

    contours_with_data_calculated = ContoursWithData(validContoursWithData)

    strFinalString = ""

    for contourWithData in contours_with_data_calculated.contours:
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 0, 0),
                      2)                        # thickness


        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RZW, RZH))
        npaROIResized = imgROIResized.reshape((1, RZW * RZH))
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)
        
        print("NPA RESULTS", npaResults)
        print("NPA retval", neigh_resp)

        strCurrChar = str(chr(int(npaResults[0][0])))
        print("CURR CHAR: ", strCurrChar)
        print("RETVAL: ", retval)

        strFinalString = strFinalString + strCurrChar
    
    print(strFinalString)

    cv2.imshow("imgtest", imgTestingNumbers)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
