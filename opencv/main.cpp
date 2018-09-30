#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat detectPlate(std::string image) {
    cv::Mat src;
    cv::Mat src_gray;
    cv::Mat mask;
    src = cv::imread(image, 1 );
    if( !src.data ) {
        throw "Error loading the image";
    }

    cvtColor( src, src_gray, cv::COLOR_BGR2GRAY);
    GaussianBlur( src_gray, src_gray, cv::Size(9, 9), 2, 2 );
    std::vector<cv::Vec3f> circles;

    // Find circles
    HoughCircles( src_gray, circles, cv::HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );

    // Find the biggest circle
    cv::Point circleCenter;
    int circleRadius = 0;
    for( size_t i = 0; i < circles.size(); i++ ) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        if(i == 0 || radius > circleRadius) {
            circleCenter = center;
            circleRadius = radius;
        }
    }

    // Create mask
    mask = cv::Mat::zeros(src.size(), src.type());

    // Process biggest circle
    circle(mask, circleCenter, circleRadius, cv::Scalar(255, 255, 255), cv::FILLED, 8, 0 );
    cv::Mat dst;
    src.copyTo(dst, mask);
    return dst;
}

cv::Mat reduceColorKmeans(const cv::Mat src, int K) {
    int n = src.rows * src.cols;
    cv::Mat data = src.reshape(1, n);
    cv::Mat dst;
    data.convertTo(data, CV_32F);

    std::vector<int> labels;
    cv::Mat1f colors;
    kmeans(data, K, labels, cv::TermCriteria(), 1, cv::KMEANS_PP_CENTERS, colors);

    for (int i = 0; i < n; ++i) {
        data.at<float>(i, 0) = colors(labels[i], 0);
        data.at<float>(i, 1) = colors(labels[i], 1);
        data.at<float>(i, 2) = colors(labels[i], 2);
    }

    cv::Mat reduced = data.reshape(3, src.rows);
    reduced.convertTo(dst, CV_8U);
    return dst;
}

int main() {
    cv::Mat plate = detectPlate("/tmp/food-small.jpg");
    cv::Mat plateKM = reduceColorKmeans(plate, 6);

    /// Show your results
    namedWindow("Plate", cv::WINDOW_AUTOSIZE );
    imshow("Plate", plateKM);
    cv::waitKey(0);
    return 0;
}