#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat detectPlateMask(cv::Mat src) {
    cv::Mat src_gray;
    cv::Mat mask;

    cvtColor( src, src_gray, cv::COLOR_BGR2GRAY);
    GaussianBlur( src_gray, src_gray, cv::Size(9, 9), 2, 2 );
    std::vector<cv::Vec3f> circles;

    // Find circles
    HoughCircles( src_gray, circles, cv::HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );

    // If no circle found
    if(circles.empty()) {
        cv::Mat emptyMask = cv::Mat(src.size(), src.type(), cv::Scalar(255, 255, 255));
        cvtColor(emptyMask, emptyMask, cv::COLOR_BGR2GRAY, 1);
        return emptyMask;
    }

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
    cvtColor(mask, mask, cv::COLOR_BGR2GRAY, 1);
    return mask;
}

cv::Mat reduceColorKmeans(cv::Mat src, cv::Mat mask, int K) {
    cv::Mat maskedImg;
    src.copyTo(maskedImg, mask);

    int n = maskedImg.rows * maskedImg.cols;
    cv::Mat data = maskedImg.reshape(1, n);
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

    cv::Mat reduced = data.reshape(3, maskedImg.rows);
    reduced.convertTo(dst, CV_8U);
    return dst;
}

cv::Mat detectFoodMask(cv::Mat mat) {
    cv::Vec3b brightestColor;
    // Find brightest color
    int maxSum = 0;
    for( int row = 0; row < mat.rows; row++ ) {
        for (int col = 0; col < mat.cols; col++) {
            int sum = 0;
            for( int c = 0; c < mat.channels(); c++ ) {
                sum += mat.at<cv::Vec3b>(row,col)[c];
            }
            if(sum > maxSum) {
                brightestColor = mat.at<cv::Vec3b>(row, col);
                maxSum = sum;
            }
        }
    }

    // Replace the brightest color
    cv::Mat mask = cv::Mat::zeros(mat.size(), mat.type());
    cv::inRange(mat, cv::Vec3b(0, 0, 0), brightestColor - cv::Vec3b(1, 1, 1), mask);
    return mask;
}


std::vector<cv::Mat> groupMasks(cv::Mat mat, int k) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mat, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // Get all non-zero pixels
    std::vector<cv::Point2f> points;
    for(int row=0; row < mat.rows; row++) {
        for(int col=0; col < mat.cols; col++) {
            if(mat.at<uchar>(row, col) != 0) {
                points.push_back(cv::Point2f(col, row));
            }
        }
    }

    // Prepare the contour vector
    std::vector<std::vector<cv::Point>> kContours(k);

    // Apply k means
    cv::Mat labels;
    cv::Mat centers;
    cv::kmeans(points, k, labels,
               cv::TermCriteria( cv::TermCriteria::EPS+cv::TermCriteria::COUNT, 50, 1.0), 3,
               cv::KMEANS_PP_CENTERS, centers);
    for(int i=0; i < points.size(); i++) {
        kContours[labels.at<uchar>(i, 0)].push_back(points[i]);
    }

    std::vector<cv::Mat> mats;
    for(int i=0; i < kContours.size(); i++) {
        // Smooth contours
        std::vector<cv::Point> convexHullPoints;
        cv::convexHull(kContours[i], convexHullPoints);

        cv::Mat partMat = cv::Mat::zeros(mat.size(), mat.type());
//        cv::drawContours(partMat, kContours, i, cv::Scalar(255), cv::FILLED);
        cv::fillConvexPoly(partMat, convexHullPoints, cv::Scalar(255));

        mats.push_back(partMat);
    }
    return mats;
}

cv::Mat cleanMask(cv::Mat mat) {
    cv::Mat dst;
    // Erode
    int erodeVal = 6;
    cv::Mat eKernel(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erodeVal, erodeVal)));
    cv::erode(mat, dst, eKernel, cv::Point(-1, -1), 1);

    // Dilate
    int dilateVal = 4;
    cv::Mat dKernel(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilateVal, dilateVal)));
    cv::dilate(dst, dst, dKernel, cv::Point(-1, -1), 1);
    return dst;
}

int main(int argc, char *argv[]) {
    cv::Mat src;
    src = cv::imread(argv[1], 1 );
    if( !src.data || argc != 3) {
        throw "Error loading the image";
    }

    int NUMBER_OF_ITEMS = std::stoi(argv[2]);
    cv::Mat plateMask = detectPlateMask(src);
    cv::Mat plateKM = reduceColorKmeans(src, plateMask, NUMBER_OF_ITEMS + 1);
    cv::Mat foodMask = detectFoodMask(plateKM);
    cv::Mat finalMask;
    cv::bitwise_and(foodMask, plateMask, finalMask);
    cv::Mat finalCleanMask = cleanMask(finalMask);
    std::vector<cv::Mat> masks = groupMasks(finalCleanMask, NUMBER_OF_ITEMS);
    std::vector<cv::Mat> plateFood;

    for(int i=0; i < masks.size(); i++) {
        cv::Mat foodComp;
        src.copyTo(foodComp, masks[i]);
        plateFood.push_back(foodComp);
    }
//    namedWindow("test", cv::WINDOW_AUTOSIZE );
//    imshow("test", finalCleanMask);

    // Save images
    std::string path = "/home/amir/github/implementAI2018/backend/myapp/public/images/ocv/";
    cv::imwrite(path + "ocv-image1.jpg", src);
    cv::imwrite(path + "ocv-image2.jpg", plateMask);

    cv::Mat plateImage;
    src.copyTo(plateImage, plateMask);
    cv::imwrite(path + "ocv-image3.jpg", plateImage);
    cv::imwrite(path + "ocv-image4.jpg", plateKM);
    cv::imwrite(path + "ocv-image5.jpg", finalMask);
    cv::imwrite(path + "ocv-image6.jpg", finalCleanMask);
    for(size_t i=0; i < plateFood.size(); i++) {
        cv::imwrite(path + "ocv-mask" + std::to_string(i) + ".jpg", masks[i]);
        cv::imwrite(path + "ocv-part" + std::to_string(i) + ".jpg", plateFood[i]);
    }

    cv::waitKey(0);
    return 0;
}