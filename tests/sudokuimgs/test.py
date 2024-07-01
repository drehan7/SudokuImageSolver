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
    # edges = cv2.Canny(gaus, 70, 150)

    thresh = cv2.adaptiveThreshold(gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    thresh = cv2.bitwise_not(thresh)

    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    # return img

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
            
    # i = img.copy()
    # pts = maxapprox.ravel()

    # cv2.imshow("I", i)
    # cv2.waitKey(0)

    contour = max(contours, key = cv2.contourArea)

    #contImg = cv2.drawContours(img, contour, -1, (0,255,0), 3)
    # print("CONTOUR: ", cv2.boundingRect(contour))
    # print("ARCLENGTH: ", cv2.arcLength(contour, True))

    out = img.copy()
    cv2.drawContours(out, [contour], -1, (0,255,0), 3)
    # cv2.imshow("Contour of sudokuboard", out)
    # cv2.waitKey(0)

    ap = img.copy()

    # finds polygon on closed contour
    approx = cv2.approxPolyDP(contour, 0.010 * cv2.arcLength(contour, True), True)

    # cv2.drawContours(ap, [approx], 0, (0,0,0), 3)
    # cv2.imshow("Approx", ap)
    # cv2.waitKey(0)

    print("APPROX: ", approx.ravel())

    # circles = img.copy()
    n = approx.ravel()
    #for i in range(0, len(n)-1, 2):
    #    x = n[i]
    #    y = n[i+1]

    #    print("x: %s  y: %s" % ( x, y ))
    #    #cv2.circle(circles, (x, y), 5, 255, -1)
        
    # cv2.imshow("Circles", circles)
    # cv2.waitKey(0)

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
    cv2.imshow("Warp", dst)
    cv2.waitKey(0)



    # minAreaRect = cv2.minAreaRect(contour)
    # angle = minAreaRect[-1]
    # if angle < -45:
    #     angle = 90 + angle

    # print("ANGLE: ", -1.0 * angle)

    # print("MINAREARECT: ", minAreaRect)

    
    # for c in contour:
    #     print("C: ", c)

    # DESKEW
    # newImage = img.copy()
    # (h,w) = newImage.shape[:2]
    # center = (w // 2, h //2)
    # M = cv2.getRotationMatrix2D(center, (-1.0*angle), 1.0)
    # newImage = cv2.warpAffine(newImage, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # cv2.imshow("DESKEW IMAGE", newImage)
    # cv2.waitKey(0)



    # roi = img[inty:inty+inth, intx:intx+intw]

    # Draw rect over orig img
    # cv2.rectangle(img, (intx, inty), (intx + intw, inty + inth), (0,255,0), 2)

    # cv2.imshow("Orig image", img)
    # cv2.imshow("Get Sudoku BOard: ", roi)
    # intKey = cv2.waitKey(0)

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

def main():
    allContoursWithData = []
    validContoursWithData = []

    try:
        npaClassifications = np.loadtxt("../../src/models/classifications.txt", np.float32)
        print("LOADED")
    except:
        print("Error")
        sys.exit(1)

    try:
        npaFlattenedImages = np.loadtxt("../../src/models/flattened_images.txt", np.float32)
    except:
        print("Error")
        sys.exit(1)

    npaClassifications = npaClassifications.reshape((npaClassifications.size), 1)

    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = find_board("../../assets/sudoku2.jpeg")
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    intKey = cv2.waitKey(0)
    if intKey == 27: exit(0)

    # exit(0)

    if imgTestingNumbers is None:
        print("ERROR")
        sys.exit(1)

    imgGray = imgTestingNumbers.copy()

    # imgBlurred = cv2.GaussianBlur(imgGray, (7,7),0)
    imgBlurred = cv2.bilateralFilter(imgGray, 7, 75, 75)

    _, imgThresh = cv2.threshold(imgBlurred, 110, 255, cv2.THRESH_BINARY)

    imgThreshCopy = imgThresh.copy()

    cv2.imshow("IMTHRES", imgThresh)
    intKey = cv2.waitKey(0)
    if intKey == 27: exit(0)

    npaContours, npaHier = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    print("CONTOURS: ", len(npaContours))
    for npaContour in npaContours:
        # if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA and cv2.contourArea(npaContour) < MAX_CONTOUR_AREA:
        #     [ intx, inty, intw, inth ] = cv2.boundingRect(npaContour)

        #     cv2.rectangle(imgTestingNumbers, (intx, inty), (intx+intw, inty+inth), (0,0,255), 2)

        #     imgROI = imgThresh[inty:inty+inth, intx:intx+intw]
        #     imgROIResize = cv2.resize(imgROI, (RZW, RZH))


        contourWithData = ContourWithData()
        contourWithData.npaContour = npaContour
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
        allContoursWithData.append(contourWithData)

    # cv2.imshow("imgroi",imgROI )
    # cv2.imshow("imgroiresize", imgROIResize)
    # cv2.imshow("testimg", imgTestingNumbers)
    # cv2.waitKey(0)
    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourValid():
            validContoursWithData.append(contourWithData)

    #validContoursWithData.sort(key=operator.attrgetter("intRectX"))

    strFinalString = ""

    for contourWithData in validContoursWithData:
        cv2.rectangle(imgTestingNumbers,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness


        cv2.imshow("imgtest", imgTestingNumbers)
        intKey = cv2.waitKey(0)
        if intKey == 27:
            exit(0)
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        # cv2.imshow("IMGROI", imgROI)
        # cv2.waitKey(0)

        imgROIResized = cv2.resize(imgROI, (RZW, RZH))
        npaROIResized = imgROIResized.reshape((1, RZW * RZH))
        npaROIResized = np.float32(npaROIResized)

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)

        strCurrChar = str(chr(int(npaResults[0][0])))

        strFinalString = strFinalString + strCurrChar
    
    print(strFinalString)

    cv2.imshow("imgtest", imgTestingNumbers)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
