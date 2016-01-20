#include "camera_calibration.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

bool CameraCalibration::initialize() {
    image = readChannel<lms::imaging::Image>("IMAGE");

    if(!setPattern()) {
        return false;
    }

    patternSize = cv::Size(config().get<int>("points_per_row"),
                           config().get<int>("points_per_col"));
    computePatternPoints();

    cv::namedWindow("camera_calibration");

    return true;
}

bool CameraCalibration::deinitialize() {
    cv::destroyWindow("camera_calibration");
    return true;
}

bool CameraCalibration::cycle() {
    cv::Mat img, imgColor;
    cv::equalizeHist(image->convertToOpenCVMat(), img);
    cv::cvtColor(img, imgColor, CV_GRAY2BGR);

    std::vector<cv::Point2f> centers;
    auto found = findPoints(img, centers);
    drawChessboardCorners(imgColor, patternSize, cv::Mat(centers), found);

    if(found)
    {
        detectedPoints.push_back(centers);
        logger.info("detect") << "Found " << detectedPoints.size() << " patterns";
    }

    if( detectedPoints.size() > config().get<size_t>("min_detections", 10) )
    {
        logger.info("calibrate") << "Computing camera matrix";
        calibrate();
    }

    cv::imshow("camera_calibration", imgColor);
    cv::waitKey(0);

    int i = 0;

    return true;
}

bool CameraCalibration::setPattern()
{
    auto pt = config().get<std::string>("pattern", "chessboard");

    if(pt == "chessboard") {
        pattern = Pattern::CHESSBOARD;
    } else if(pt == "circles") {
        pattern = Pattern::CIRCLES;
    } else if(pt == "circles") {
        pattern = Pattern::CIRCLES_ASYMMETRIC;
    } else {
        logger.error("pattern") << "Invalid calibration pattern";
        return false;
    }
    return true;
}

void CameraCalibration::computePatternPoints()
{
    patternPoints.resize(0);

    float length = config().get<float>("length", 1);

    switch(pattern)
    {
        case Pattern::CHESSBOARD:
        case Pattern::CIRCLES:
            for(int i = 0; i < patternSize.height; i++) {
                for(int j = 0; j < patternSize.width; j++) {
                    patternPoints.emplace_back(j * length,
                                               i*length,
                                               0);
                }
            }
            break;
        case Pattern::CIRCLES_ASYMMETRIC:
            for(int i = 0; i < patternSize.height; i++) {
                for(int j = 0; j < patternSize.width; j++) {
                    patternPoints.emplace_back((2 * j + i % 2) * length,
                                               i*length,
                                               0);
                }
            }
            break;
    }
}

bool CameraCalibration::findPoints(const cv::Mat& img, std::vector<cv::Point2f>& points)
{
    bool success = false;
    switch(pattern) {
        case Pattern::CHESSBOARD:
            success = cv::findChessboardCorners(img, patternSize, points);
            if(success) {
                //cv::cornerSubPix(img, points, cv::Size(11,11), cv::Size(-1,-1),
                //             cv::TermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1));
            }
            break;
        case Pattern::CIRCLES:
            success = cv::findCirclesGrid(img, patternSize, points, cv::CALIB_CB_SYMMETRIC_GRID);
            break;
        case Pattern::CIRCLES_ASYMMETRIC:
            success = cv::findCirclesGrid(img, patternSize, points, cv::CALIB_CB_ASYMMETRIC_GRID);
            break;
    }
    return success;
}

bool CameraCalibration::calibrate()
{
    // Calibrate from detected pattersn
    std::vector<std::vector<cv::Point3f> > worldCoordinates(detectedPoints.size(), patternPoints);
    cv::Size imgSize(image->width(), image->height());

    cv::Mat K;
    cv::Mat coeff;
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    auto ret = cv::calibrateCamera(worldCoordinates, detectedPoints, imgSize,
                                   K, coeff, rvecs, tvecs, CV_CALIB_FIX_K4|CV_CALIB_FIX_K4);

    std::cout << "k" << std::endl;
    for(int i = 0; i < K.rows; i++) {
        for(int j = 0; j < K.cols; j++) {
            std::cout << K.at<double>(i,j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "dist" << std::endl;
    for(int i = 0; i < coeff.rows; i++) {
        for(int j = 0; j < coeff.cols; j++) {
            std::cout << coeff.at<double>(i,j) << " ";
        }
        std::cout << std::endl;
    }

    return false;
}
