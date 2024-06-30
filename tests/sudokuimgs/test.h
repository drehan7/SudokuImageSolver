#ifndef IMAGE_READER_H
#define IMAGE_READER_H

#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <vector>

constexpr int MIN_CONTOUR_AREA = 300;
constexpr int MAX_CONTOUR_AREA = 300;

constexpr int RESIZED_IMAGE_WIDTH = 20;
constexpr int RESIZED_IMAGE_HEIGHT = 30;

cv::Mat findBoard( const std::string& fp );
int train(std::vector<cv::String>& filenames);
std::vector<cv::Point> getMaxContour( std::vector<std::vector<cv::Point>>& contours );


#endif // IMAGE_READER_H
