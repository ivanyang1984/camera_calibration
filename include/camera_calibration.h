#ifndef CAMERA_CALIBRATION_H
#define CAMERA_CALIBRATION_H

#include <lms/module.h>

/**
 * @brief LMS module camera_calibration
 **/
class CameraCalibration : public lms::Module {
public:
    bool initialize() override;
    bool deinitialize() override;
    bool cycle() override;
};

#endif // CAMERA_CALIBRATION_H
