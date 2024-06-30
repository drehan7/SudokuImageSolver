import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, sys

def showimg(img, name="IMG"):
    cv2.imshow(name, img)
    cv2.waitKey(0)


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


images = []
imgDir = "../assets"
for im in os.listdir("../assets"):
    if "sudoku" in im:
        images.append(os.path.join(imgDir, im))


cropped_imgs = []

for im in images:
    cropped_imgs.append(find_board(im))


MIN_CONTOUR_AREA = 300
MAX_CONTOUR_AREA = 7000

RZW, RZH = 20, 30

for img in cropped_imgs:

    imgBlurred = cv2.GaussianBlur(img, (5,5), 0)

    _, imgThresh = cv2.threshold(imgBlurred, 80, 255, cv2.THRESH_BINARY)
    #imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)

    showimg(imgThresh, "imgThresh")

    imgThreshCopy = imgThresh.copy()

    npaContours, npaHier = cv2.findContours(imgThreshCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    intClassifications = []
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')]
    npaFlattenedImages =  np.empty((0, RZW * RZH))

    for npaContour in npaContours:
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA and cv2.contourArea(npaContour) < MAX_CONTOUR_AREA:
            [ intx, inty, intw, inth ] = cv2.boundingRect(npaContour)

            cv2.rectangle(img, (intx, inty), (intx+intw, inty+inth), (0,0,255), 2)

            imgROI = imgThresh[inty:inty+inth, intx:intx+intw]
            imgROIResize = cv2.resize(imgROI, (RZW, RZH))

            cv2.imshow("imgroi",imgROI )
            cv2.imshow("imgroiresize", imgROIResize)
            cv2.imshow("img", img)

            intChar = cv2.waitKey(0)

            if intChar == 27:
                sys.exit()
            elif intChar in intValidChars:
                intClassifications.append(intChar)
                npaFlattenedImage = imgROIResize.reshape((1, RZW * RZH))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)

    fltClassifications = np.array(intClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))

print("TRAINING COMPLETE")

np.savetxt("classifications.txt", npaClassifications)
np.savetxt("flattened_images.txt", npaFlattenedImages)

cv2.destroyAllWindows()

####################

#thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#kernel = np.ones((4,4), np.uint8)
#morph = cv2.morphologyEx(thresh[1], cv2.MORPH_GRADIENT, kernel)

##i, cont = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#contours, hier = cv2.findContours(morph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

#ext = np.zeros(morph.shape)

#for i in range(len(contours)):
#    if hier[0][i][3] == -1:
#        cv2.drawContours(ext, contours, i, 255, -1)

#showimg(ext)
#print(contours)

# for 
#     print(c)
#     r = cv2.boundingRect(c)
#     cv2.rectangle(img, r, (0,0,255), 2)

#     showimg(img)


#############################
