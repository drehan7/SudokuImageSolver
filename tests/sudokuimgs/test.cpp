#include "test.h"

std::vector<cv::Point> getMaxContour( std::vector<std::vector<cv::Point>>& contours )
{
    auto ret = contours.front();
    double maxArea = 1;
    for ( const auto& cont : contours ) {
        double newArea = cv::contourArea(cont);
        if ( newArea > maxArea ) {
            ret = cont;
            maxArea = newArea;
        }
    }

    return ret;
}

cv::Mat findBoard( const std::string& fp )
{
    cv::Mat gauss;
    cv::Mat edges;

    std::vector<std::vector<cv::Point>> pContours;
    std::vector<cv::Vec4i> v4iHier;

    cv::Mat im = cv::imread( fp, 0 );
    if ( im.empty() ) {
        printf("Error with image\n");
        return cv::Mat();
    }
    cv::GaussianBlur(im, gauss, cv::Size(3,3), cv::BORDER_DEFAULT);
    cv::Canny(gauss, edges, 100, 200);
    cv::findContours(edges, pContours, v4iHier, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    auto maxCont = getMaxContour( pContours );

    cv::Rect br = cv::boundingRect(maxCont);

    cv::Mat roi = im(br);

    /* cv::imshow(fp, roi); */
    /* cv::waitKey(0); */
    /* cv::destroyAllWindows(); */

    return roi;
}

int main()
{
    std::string dir = "../../assets/sudoku*.jpeg";
    std::vector<cv::String> filenames;
    cv::glob(dir, filenames, false);

    train( filenames );

    /* for ( const auto& fn : filenames ) { */
    /*     findBoard(fn); */
    /* } */
}

int train(std::vector<cv::String>& filenames)
{

    std::vector<cv::Mat> imgTrainingNumbers;
    cv::Mat imgTrainingNumber;
    cv::Mat imgBlurred;
    cv::Mat imgThresh;
    cv::Mat imgThreshCopy;


    std::vector<std::vector<cv::Point>> pContours;
    std::vector<cv::Vec4i> v4iHierarchy;

    cv::Mat matClassificationInts;

    std::vector<int> intValidChars = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};


    // LOAD IMAGE
    for ( const auto& f : filenames ) {
        cv::Mat grayImg;

        auto i = findBoard( f );
        if ( i.empty() ) continue;

        imgTrainingNumbers.push_back( i );
    }

    // DEBUG
    imgTrainingNumber = imgTrainingNumbers.front().clone();
    cv::Mat imgGrayScale = imgTrainingNumbers.front().clone();

#ifdef DEBUG_VIZ
    cv::imshow("GRAY", imgGrayScale);
    cv::waitKey(0);
#endif

    cv::GaussianBlur(imgGrayScale, imgBlurred, cv::Size(5,5), 0);

#ifdef DEBUG_VIZ
    cv::imshow("Guass blue", imgBlurred);
    cv::waitKey(0);
#endif


    cv::adaptiveThreshold(imgBlurred,
            imgThresh,
            155,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            11,
            2);

#if 1 
    cv::imshow("Thres", imgThresh);
    cv::waitKey(0);
#endif

    /* cv::Mat sobelx; */
    /* cv::Sobel(imgThresh, sobelx, CV_32F, 1, 0, 5); */

/* #ifdef DEBUG_VIZ */
    /* cv::imshow("Soble", sobelx); */
    /* cv::waitKey(0); */
/* #endif */

    imgThreshCopy = imgThresh.clone();

    cv::findContours(imgThreshCopy,
            pContours,
            v4iHierarchy,
            cv::RETR_EXTERNAL,
            cv::CHAIN_APPROX_NONE);

    printf("AFTER CONTOURS Len: %ld\n", pContours.size());

    /* cv::imshow("Contoures", imgThreshCopy); */
    /* cv::waitKey(0); */


    for ( int i = 0; i < pContours.size(); ++i ) {
        /* if ( cv::contourArea(pContours.at(i)) > MIN_CONTOUR_AREA ) */
        {
            cv::Rect boundingRect = cv::boundingRect(pContours.at(i));

            cv::rectangle(imgTrainingNumber, boundingRect, cv::Scalar(0,0,255), 2); // red rect around contour

            cv::Mat matROI = imgThresh(boundingRect); // Get ROI image of bounding rect
            cv::Mat matROIResized;
            cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

            cv::imshow("matROI", matROI);
            cv::imshow("matROIResized", matROIResized);
            cv::imshow("imgTrainingNumber", imgTrainingNumber);

            int ch = cv::waitKey(0);

            if ( ch == 27 ) {
                return 0;
            }
        }
    }



    return 0;

}
