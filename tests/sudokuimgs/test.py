import cv2
import numpy as np
import operator
import os
import sys

MIN_CONTOUR_AREA = 265 
MAX_CONTOUR_AREA = 7500

RZW, RZH = 20, 30

# Organize points
def order_points(pts):

    rect = np.zeros((4,2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]

    return rect


def preprocess_img(img):
    imgBlurred = cv2.GaussianBlur(img, (1,1), cv2.BORDER_DEFAULT)
    _, imgThresh = cv2.threshold(imgBlurred, 115, 255, cv2.THRESH_TOZERO)
    imgThreshCopy = imgThresh.copy()

    return img, imgThresh, imgThreshCopy


def find_board(imagepath):
    img = cv2.imread(imagepath, 0)
    cv2.imshow("orig", img)
    cv2.waitKey(0)
    gaus = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)

    thresh = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    thresh = cv2.bitwise_not(thresh)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    per = cv2.arcLength(contour, True)
    ap = cv2.approxPolyDP(contour, 0.05 * per, True)

    out = img.copy()
    cv2.drawContours(out, [contour], -1, (0,255,0), 3)

    pts1 = []
    for point in ap:
        x, y = point[0]
        pts1.append([x, y])
        cv2.circle(out, (x,y), 7, (0,0,0), -1)

    ap = img.copy()

    # finds polygon on closed contour
    approx = cv2.approxPolyDP(contour, 0.010 * cv2.arcLength(contour, True), True)

    n = approx.ravel()

    pts1 = np.float32(pts1)
    pts1 = order_points(pts1)

    [intx, inty, intw, inth] = cv2.boundingRect(contour)
    pts2 = np.float32([
        [0,0],
        [intx+intw, 0],
        [0, inty+inth],
        [intx+intw, inty+inth]
        ])

    M = cv2.getPerspectiveTransform( pts1, pts2 )

    warp = img.copy()
    dst = cv2.warpPerspective(img, M, (intx+intw, inty+inth))
    rsdst = cv2.resize(dst, (900, 900))

    return rsdst


class ContourWithData():

    npaContour = None
    boundingRect = None
    intRectX = 0
    intRectY = 0
    intRectWidth = 0
    intRectHeight = 0

    ch_digit = ''

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

    def contourGetDims(self):
        return self.intRectX, self.intRectY, self.intRectWidth, self.intRectHeight


# Rename this, redundant af
class DigitContours():
    contours: ContourWithData = []
    avg_area = -1
    avg_height = -1
    avg_width = -1

    def __init__(self):
        pass
    
    def unload(self):
        self.contours = []
        self.avg_area = -1
        self.avg_height = -1
        self.avg_width = -1

    def calc_avg_area(self, tmp_contours):
        if len(tmp_contours) == 0:
            print("contours empty")
            return

        area_sum = 0
        height_sum = 0
        width_sum = 0
        for contour in tmp_contours:
            _area = contour.intRectWidth * contour.intRectHeight
            area_sum += _area
            height_sum += contour.intRectHeight
            width_sum += contour.intRectWidth
        
        self.avg_area = (area_sum / len(tmp_contours))
        self.avg_height = (height_sum / len(tmp_contours))
        self.avg_width = (width_sum / len(tmp_contours))
        print("Avg height: ", self.avg_height)


    def append_contours(self, contours, img):
        imgDims = cv2.boundingRect(img)
        tmp_contours = []
        for c in contours:
            tmp_contours.append(c)

        self.calc_avg_area(tmp_contours)
        self.contours = [ c for c in tmp_contours if self.within_area(c, imgDims) ]


    # Reject contours that are way too close to the edges of image
    def within_area(self, contour, imgDims):
        x, y, w, h = contour.contourGetDims()
        a = w*h
        if a > self.avg_area * 2: return False

        edge_margin = 20
        imgX, imgY, imgW, imgH = imgDims

        # Check if contour is on edges of image
        if x < imgX + edge_margin or y < imgY + edge_margin:
            return False
        elif h > self.avg_height + (1/3 * self.avg_height): return False

        return True

    # Organize contours from topleft to bottomright
    # Idea 1:
        # Draw a horizontal line across image.
        # Go top to bottom
        # Once contour is intersected, calculate distance from top, use as stride
        # Get digit in contour
    def organize(self):
        pass


def load_knn():
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

    return kNearest


def get_images():
    cropped_imgs = []
    path = "../../assets"
    for im in os.listdir("../../assets"):
        if "sudoku" in im:
            cropped_imgs.append(find_board(os.path.join(path, im)))

    return cropped_imgs


def main():
    allContoursWithData = []
    validContoursWithData = []
    digit_contours = DigitContours()

    images = get_images()
    kNearest = load_knn()

    for imgTestingNumbers in images:

        if imgTestingNumbers is None:
            print("ERROR")
            sys.exit(1)

        cpImg = imgTestingNumbers.copy()
        imgGray, imgThresh, imgThreshCopy = preprocess_img(cpImg)

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

        digit_contours.append_contours(validContoursWithData, imgGray)

        # sort contours from topleft to bottomright
        digit_contours.organize()

        strFinalString = ""

        for i, contourWithData in enumerate(digit_contours.contours):
            cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                          (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                          (0, 0, 0),
                          2)                        # thickness

            # crop char out of threshold image
            imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,
                               contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

            imgROIResized = cv2.resize(imgROI, (RZW, RZH))
            npaROIResized = imgROIResized.reshape((1, RZW * RZH))
            npaROIResized = np.float32(npaROIResized)

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)
            
            strCurrChar = str(chr(int(npaResults[0][0])))
            contourWithData.ch_digit = strCurrChar
            strFinalString = strFinalString + strCurrChar
        

        # cv2.namedWindow("Out", cv2.WINDOW_NORMAL)
        # cv2.imshow("Out", imgTestingNumbers)
        # intKey = cv2.waitKey(0)
        # if intKey == 27: exit(0)

        digit_contours.unload()
        allContoursWithData = []
        validContoursWithData = []
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
