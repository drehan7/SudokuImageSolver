import cv2
import numpy as np
import operator
import os
import sys



MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 7000

RZW, RZH = 20, 30

def find_board(imagepath):
    img = cv2.imread(imagepath, 0)
    gaus = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
    edges = cv2.Canny(gaus, 100, 200)

    contours, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key = len)

    #contImg = cv2.drawContours(img, contour, -1, (0,255,0), 3)

    [intx, inty, intw, inth] = cv2.boundingRect(contour)


    roi = img[inty:inty+inth, intx:intx+intw]

    return roi


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
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
        print("LOADED")
    except:
        print("Error")
        sys.exit(1)

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("Error")
        sys.exit(1)

    npaClassifications = npaClassifications.reshape((npaClassifications.size), 1)

    kNearest = cv2.ml.KNearest_create()
    kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    imgTestingNumbers = find_board("sudoku2.jpeg")

    if imgTestingNumbers is None:
        print("ERROR")
        sys.exit(1)

    imgGray = imgTestingNumbers.copy()

    imgBlurred = cv2.GaussianBlur(imgGray, (5,5),0)

    _, imgThresh = cv2.threshold(imgBlurred, 80, 255, cv2.THRESH_BINARY)

    imgThreshCopy = imgThresh.copy()

    cv2.imshow("IMTHRES", imgThresh)
    cv2.waitKey(0)

    npaContours, npaHier = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
        cv2.waitKey(0)
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
