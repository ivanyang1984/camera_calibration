#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <lms/module.h>

#include <opencv2/core/core.hpp>
#include <lms/imaging/image.h>

#include "calibration.h"

/**
 * @brief LMS module camera_calibration
 **/
class CameraCalibration : public lms::Module {
public:
    bool initialize() override;
    bool deinitialize() override;
    bool cycle() override;
    void configsChanged() override;

protected:
    lms::ReadDataChannel<lms::imaging::Image> image;

    Model model;
    Pattern pattern;
    cv::Size patternSize;
    std::vector<cv::Point3f> patternPoints;
    std::vector< std::vector<cv::Point2f> > detectedPoints;

    lms::Time lastCapture;

    // Calibration
    bool hasValidCalibration;
    double reprojectionError;
    cv::Mat intrinsics; //!< Computed intrinsic camera matrix
    cv::Mat coeff;      //!< Computed distortion model coefficients

    bool initParameters();
    bool setModel();
    bool setPattern();
    void computePatternPoints();

    cv::Size getSize();
    cv::Mat getNewCameraMatrix();

    bool findPoints(const cv::Mat& img, std::vector<cv::Point2f>& points);
    void detect(cv::Mat& img, cv::Mat& visualization);
    bool calibrate();
    bool undistort(const cv::Mat& img, cv::Mat& undist);
    bool undistort(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& undist);
    bool distort(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& dist);

    void interpolate(cv::Mat& mat, size_t iterations);

    bool saveCalibration();
};

#endif // CAMERA_CALIBRATION_H
