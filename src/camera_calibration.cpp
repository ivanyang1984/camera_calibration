#include "camera_calibration.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

bool CameraCalibration::initialize()
{
    image = readChannel<lms::imaging::Image>("IMAGE");

    if(!initPattern()) {
        return false;
    }

    cv::namedWindow("camera_calibration");

    hasValidCalibration = false;
    lastCapture = lms::Time::ZERO;

    return true;
}

bool CameraCalibration::deinitialize()
{
    cv::destroyWindow("camera_calibration");
    return true;
}

bool CameraCalibration::cycle()
{
    cv::Mat img, imgColor;
    cv::equalizeHist(image->convertToOpenCVMat(), img);
    cv::cvtColor(img, imgColor, CV_GRAY2BGR);

    if (hasValidCalibration) {
        // Show undistored image
        cv::Mat tmp;
        undistort(imgColor, tmp);
        imgColor = tmp;
    } else {
        // Detect pattern
        detect(img, imgColor);
    }

    cv::imshow("camera_calibration", imgColor);
    cv::waitKey(config().get<int>("wait", 10));

    return true;
}

void CameraCalibration::configsChanged() {
    logger.info("configs") << "Configs changed, recomputing calibration pattern";
    initPattern();
}

void CameraCalibration::detect(cv::Mat& img, cv::Mat& visualization)
{
    std::vector<cv::Point2f> centers;
    auto found = findPoints(img, centers);
    drawChessboardCorners(visualization, patternSize, cv::Mat(centers), found);

    if (found &&
        lastCapture.since().toFloat<std::milli>() > config().get<float>("delay", 1)) {
        // Enough delay has passed between images to generate a new sample
        detectedPoints.push_back(centers);
        logger.info("detect") << "Found " << detectedPoints.size() << " patterns";
    }

    if (detectedPoints.size() > config().get<size_t>("min_detections", 10)) {
        logger.info("calibrate") << "Computing camera matrix";
        calibrate();
    }
}

bool CameraCalibration::initPattern() {
    if (!setPattern()) {
        return false;
    }

    patternSize = cv::Size(config().get<int>("points_per_row"),
                           config().get<int>("points_per_col"));
    computePatternPoints();

    return true;
}

bool CameraCalibration::setPattern()
{
    auto pt = config().get<std::string>("pattern", "chessboard");

    if (pt == "chessboard") {
        pattern = Pattern::CHESSBOARD;
    } else if (pt == "circles") {
        pattern = Pattern::CIRCLES;
    } else if (pt == "circles_asymmetric") {
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

    switch (pattern) {
        case Pattern::CHESSBOARD:
        case Pattern::CIRCLES:
            for (int i = 0; i < patternSize.height; i++) {
                for (int j = 0; j < patternSize.width; j++) {
                    patternPoints.emplace_back(j * length,
                                               i * length,
                                               0);
                }
            }
            break;
        case Pattern::CIRCLES_ASYMMETRIC:
            for (int i = 0; i < patternSize.height; i++) {
                for (int j = 0; j < patternSize.width; j++) {
                    patternPoints.emplace_back((2 * j + i % 2) * length,
                                               i * length,
                                               0);
                }
            }
            break;
    }
}

bool CameraCalibration::findPoints(const cv::Mat& img, std::vector<cv::Point2f>& points)
{
    bool success = false;
    switch (pattern) {
        case Pattern::CHESSBOARD:
            success = cv::findChessboardCorners(img, patternSize, points);
            if (success) {
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

    double calibrationResult = 0.0;

    if (config().get<std::string>("model") == "fisheye") {
        calibrationResult = cv::fisheye::calibrate(worldCoordinates, detectedPoints, imgSize,
                                                   intrinsics, coeff, cv::noArray(), cv::noArray());
    } else {
        calibrationResult = cv::calibrateCamera(worldCoordinates, detectedPoints, imgSize,
                                                intrinsics, coeff, cv::noArray(), cv::noArray());
    }

    hasValidCalibration = true;

    saveCalibration();

    return true;
}

bool CameraCalibration::saveCalibration()
{
    // Save calibration data
    if(!isEnableSave())
    {
        logger.error() << "Command line option --enable-save was not specified";
        return false;
    }

    if(intrinsics.rows != 3 || intrinsics.cols != 3)
    {
        logger.error("intrinsics") << "Invalid intrinsic matrix dimensions: " << intrinsics.rows << " x " << intrinsics.cols;
    }

    std::ofstream output(saveLogDir("camera_calibration") + "calibration.lconf");

    output << "model = " << config().get<std::string>("model", "default") << std::endl;
    output << "col = " << image->width() << std::endl;
    output << "row = " << image->height() << std::endl;
    output << "Fx = " << intrinsics.at<double>(0, 0) << std::endl;
    output << "Fy = " << intrinsics.at<double>(1, 1) << std::endl;
    output << "Cx = " << intrinsics.at<double>(0, 2) << std::endl;
    output << "Cy = " << intrinsics.at<double>(1, 2) << std::endl;

    for(int i = 0; i < coeff.cols; i++)
    {
        output << "K" + std::to_string(i+1) << " = " << coeff.at<double>(i) << std::endl;
    }

    return true;
}

bool CameraCalibration::undistort(const cv::Mat& img, cv::Mat& undist)
{
    if (config().get<std::string>("model") == "fisheye") {
        cv::fisheye::undistortImage(img, undist, intrinsics, coeff);
    }
    else {
        cv::undistort(img, undist, intrinsics, coeff);
    }
    return false;
}
