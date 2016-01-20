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

protected:
    lms::ReadDataChannel<lms::imaging::Image> image;
    Pattern pattern;
    cv::Size patternSize;
    std::vector<cv::Point3f> patternPoints;

    std::vector< std::vector<cv::Point2f> > detectedPoints;

    bool setPattern();
    void computePatternPoints();
    bool findPoints(const cv::Mat& img, std::vector<cv::Point2f>& points);
    bool calibrate();
};

#endif // CAMERA_CALIBRATION_H
