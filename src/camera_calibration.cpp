#include "camera_calibration.h"

#include <boost/format.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/calib3d/calib3d.hpp>
bool CameraCalibration::initialize()
{
    image = readChannel<lms::imaging::Image>("IMAGE");

    if(!initParameters()) {
        return false;
    }

    cv::namedWindow("camera_calibration");

    hasValidCalibration = false;
    lastCapture = lms::Time::ZERO;
    nframesIntr = config().get<size_t>("min_detections", 10);

	  patternOverlay = cv::Mat::zeros(1,1, 6);
    return true;
}

bool CameraCalibration::deinitialize()
{
    cv::destroyWindow("camera_calibration");
    return true;
}

bool CameraCalibration::cycle()
{
  std::string msg;
    cv::Mat img, imgColor;
    cv::equalizeHist(image->convertToOpenCVMat(), img);
    cv::cvtColor(img, imgColor, CV_GRAY2BGR);

  
    if (patternOverlay.cols == 1) {
			patternOverlay = cv::Mat::zeros(imgColor.size(), imgColor.type());
		}

    if (hasValidCalibration) {
        // Show undistored image
        cv::Mat tmp;
        undistort(imgColor, tmp);
        imgColor = tmp;
    } else {
        // Detect pattern
        detect(img, imgColor);
        
    }
    cv::add(imgColor,patternOverlay,imgColor);
    msg = str(boost::format("%d %d %d %d %d %d/%d")  % topLeft % topMid % topRight %  bottomLeft % bottomMid % bottomRight % nframesIntr);
 
		int b = 0;
    cv::Size textSize = cv::getTextSize(msg, 1, 2, 1, &b);
		int x_val = imgColor.cols - 2 * textSize.width - 10;

    cv::Point textOrigin(x_val > 0 ? x_val : 10, imgColor.rows - 2 * b - 10);
    cv::putText(imgColor, msg, textOrigin, 1, 1, cv::Scalar(0, 0, 255));
    cv::imshow("camera_calibration", imgColor);
    cv::waitKey(config().get<int>("wait", 10));

    return true;
}

void CameraCalibration::configsChanged() {
    logger.info("configs") << "Configs changed, recomputing calibration pattern";
    initParameters();
}

void CameraCalibration::detect(cv::Mat& img, cv::Mat& visualization)
{
    std::vector<cv::Point2f> centers;
    auto found = findPoints(img, centers);
    drawChessboardCorners(visualization, patternSize, cv::Mat(centers), found);

    if (found &&
        lastCapture.since().toFloat<std::milli>() > config().get<float>("delay", 1)) {
        // Enough delay has passed between images to generate a new sample
     
      cv::Mat_<cv::Point> corners = cv::Mat(centers);  
      if(checkFoundPattern(corners, visualization)){
       
        detectedPoints.push_back(centers);
        lastCapture = lms::Time::now();
        logger.info("detect") << "Found " << detectedPoints.size() << " patterns";
      }   
      
    }

    if (detectedPoints.size() >= config().get<size_t>("min_detections", 10) * 6) { //Warum mal 6?
        logger.info("calibrate") << "Computing camera matrix";
        calibrate();
        logger.info("calibrate") << "Camera calibration finished with reprojection error " << reprojectionError;
    }
}

bool CameraCalibration::initParameters() {
    if (!setPattern()) {
        return false;
    }

    if(!setModel()) {
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
        logger.error("pattern") << "Invalid calibration pattern '" << pt << "'";
        return false;
    }
    return true;
}

bool CameraCalibration::setModel()
{
    auto md = config().get<std::string>("model", "default");
    if(md == "default") {
        model = Model::DEFAULT;
    } else if(md == "fisheye") {
        model = Model::FISHEYE;
    } else {
        logger.error("model") << "Invalid distortion model '" << md << "'";
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

    if (model == Model::FISHEYE) {
        calibrationResult = cv::fisheye::calibrate(worldCoordinates, detectedPoints, imgSize,
                                                   intrinsics, coeff, cv::noArray(), cv::noArray());
    } else {
        int flags = 0;
        if(config().get<bool>("fix_principal_point", false)) {
            flags |= CV_CALIB_FIX_PRINCIPAL_POINT;
        }
        if(config().get<bool>("fix_aspect_ratio", false)) {
            flags |= CV_CALIB_FIX_ASPECT_RATIO;
        }
        if(config().get<bool>("zero_tangent_dist", false)) {
            flags |= CV_CALIB_ZERO_TANGENT_DIST;
        }
        if(config().get<bool>("rational_model", false)) {
            flags |= CV_CALIB_RATIONAL_MODEL;
        }

        calibrationResult = cv::calibrateCamera(worldCoordinates, detectedPoints, imgSize,
                                                intrinsics, coeff, cv::noArray(), cv::noArray(), flags);
    }

    hasValidCalibration = true;
    reprojectionError = calibrationResult;

    saveCalibration();

    return true;
}

bool CameraCalibration::saveCalibration()
{
    // Save calibration data
    if(!isEnableSave()) {
        logger.error() << "Command line option --enable-save was not specified";
        return false;
    }

    if(intrinsics.rows != 3 || intrinsics.cols != 3) {
        logger.error("intrinsics") << "Invalid intrinsic matrix dimensions: " << intrinsics.rows << " x " << intrinsics.cols;
    }

    std::ofstream params_out(saveLogDir("camera_calibration") + "params.lconf");

    params_out << "model = ";
    if(Model::FISHEYE == model) {
        params_out << "fisheye";
    } else {
        params_out << "default";
    }
    params_out << std::endl;

    params_out << "col = " << image->width() << std::endl;
    params_out << "row = " << image->height() << std::endl;
    params_out << "Fx = " << intrinsics.at<double>(0, 0) << std::endl;
    params_out << "Fy = " << intrinsics.at<double>(1, 1) << std::endl;
    params_out << "Cx = " << intrinsics.at<double>(0, 2) << std::endl;
    params_out << "Cy = " << intrinsics.at<double>(1, 2) << std::endl;

    for(int i = 0; i < coeff.cols; i++) {
        params_out << "K" + std::to_string(i + 1) << " = " << coeff.at<double>(i) << std::endl;
    }

    // Compute mapping from undistored pixel coordinates to pixel coordinates in original image (and back)
    std::ofstream lut_dist_out(saveLogDir("camera_calibration") + "lut_dist.lconf");
    std::ofstream lut_dist_mat(saveLogDir("camera_calibration") + "lut_dist.m");
    std::ofstream lut_undist_out(saveLogDir("camera_calibration") + "lut_undist.lconf");
    std::ofstream lut_undist_mat(saveLogDir("camera_calibration") + "lut_undist.m");

    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> pointsDistortedCoordinates;
    std::vector<cv::Point2f> pointsUndistortedCoordinates;
    for( int c = 0; c < image->width(); c++ ) {
        for( int r = 0; r < image->height(); r++ ) {
            points.emplace_back(r, c);
        }
    }

    // Compute undistortion
    undistort(points, pointsDistortedCoordinates);
    {
        std::stringstream x, y, xMat, yMat;
        auto p = pointsDistortedCoordinates.begin();
        for(int i = 0; i < image->width()*image->height(); i++) {
            if(i % image->height() == 0) {
                x << "\\" << std::endl;
                y << "\\" << std::endl;
            }
            x << p->x;
            y << p->y;
            xMat << p->x;
            yMat << p->y;
            p++;
            if( i < image->width()*image->height()-1 ) {
                x << ",";
                y << ",";
                xMat << ",";
                yMat << ",";
            }
        }
        lut_dist_out << "n2dX = " << x.str() << std::endl;
        lut_dist_out << "n2dY = " << y.str() << std::endl;
        lut_dist_mat << "x = [" << xMat.str() << "];" << std::endl;
        lut_dist_mat << "y = [" << yMat.str() << "];" << std::endl;
    }

    // Compute distortion
    distort(points, pointsUndistortedCoordinates);
    {
        std::stringstream x, y, xMat, yMat;
        auto p = pointsUndistortedCoordinates.begin();
        for(int i = 0; i < image->width()*image->height(); i++) {
            if(i % image->height() == 0) {
                x << "\\" << std::endl;
                y << "\\" << std::endl;
            }
            x << p->x;
            y << p->y;
            xMat << p->x;
            yMat << p->y;
            p++;
            if( i < image->width()*image->height()-1 ) {
                x << ",";
                y << ",";
                xMat << ",";
                yMat << ",";
            }
        }
        lut_undist_out << "d2nX = " << x.str() << std::endl;
        lut_undist_out << "d2nY = " << y.str() << std::endl;
        lut_undist_mat << "x = [" << xMat.str() << "];" << std::endl;
        lut_undist_mat << "y = [" << yMat.str() << "];" << std::endl;
    }
    return true;
}

bool CameraCalibration::undistort(const cv::Mat& img, cv::Mat& undist)
{
    if (model == Model::FISHEYE) {
        cv::fisheye::undistortImage(img, undist, intrinsics, coeff);
    } else {
        cv::undistort(img, undist, intrinsics, coeff, getNewCameraMatrix());
    }
    return true;
}

bool CameraCalibration::undistort(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& undist)
{
    undist.clear();

    cv::Mat mapX, mapY;
    if (model == Model::FISHEYE)
    {
        //cv::fisheye::undistortPoints(points, undist, intrinsics, coeff);
        return false;
    }
    else
    {
        initUndistortRectifyMap(intrinsics, coeff, cv::noArray(),
                                getNewCameraMatrix(),
                                getSize(), CV_32FC1, mapX, mapY);
    }

    // Remap points
    for(const auto& p : points)
    {
        undist.emplace_back(mapX.at<float>(p.x, p.y), mapY.at<float>(p.x, p.y));
    }

    return true;
}

bool CameraCalibration::distort(const std::vector<cv::Point2f>& points, std::vector<cv::Point2f>& dist)
{
    dist.clear();

    cv::Mat mapX, mapY;
    if (model == Model::FISHEYE)
    {
        //cv::fisheye::undistortPoints(points, undist, intrinsics, coeff);
        return false;
    }
    else
    {
        initUndistortRectifyMap(intrinsics, coeff, cv::noArray(),
                                getNewCameraMatrix(),
                                getSize(), CV_32FC1, mapX, mapY);
    }

    // Compute reverse mapping using interpolation
    cv::Mat revX(mapX.rows, mapX.cols, CV_32FC1, std::numeric_limits<float>::quiet_NaN());
    cv::Mat revY(mapX.rows, mapX.cols, CV_32FC1, std::numeric_limits<float>::quiet_NaN());

    // "Project" original mapping
    for( int c = 0; c < mapX.cols; c++ ) {
        for( int r = 0; r < mapX.rows; r++ ) {
            auto x = mapX.at<float>(r, c);
            auto y = mapY.at<float>(r, c);
            revX.at<float>(y, x) = static_cast<float>(c);
            revY.at<float>(y, x) = static_cast<float>(r);
        }
    }

    // Interpolate missing projections
    interpolate(revX, 10);
    interpolate(revY, 10);

    // Remap points
    for(const auto& p : points)
    {
        auto x = revX.at<float>(p.x, p.y);
        if(std::isnan(x)) {
            x = -1;
        }
        auto y = revY.at<float>(p.x, p.y);
        if(std::isnan(x)) {
            y = -1;
        }
        dist.emplace_back(x, y);
    }

    return true;
}

void CameraCalibration::interpolate(cv::Mat& mat, size_t iterations)
{
    for(size_t i = 0; i < iterations; i++) {
        for( int c = 1; c < mat.cols-1; c++ ) {
            for( int r = 0; r < mat.rows; r++ ) {
                // Interpolate horizontally
                if(std::isnan(mat.at<float>(r, c))) {
                    // needs interpolation
                    auto left = mat.at<float>(r, c-1);
                    auto right = mat.at<float>(r, c+1);
                    if(!std::isnan(left) && !std::isnan(right)) {
                        mat.at<float>(r, c) = (left+right)/2;
                    }
                }
            }
        }
        for( int c = 0; c < mat.cols; c++ ) {
            for( int r = 1; r < mat.rows-1; r++ ) {
                // Interpolate horizontally
                if(std::isnan(mat.at<float>(r, c))) {
                    // needs interpolation
                    auto top = mat.at<float>(r-1, c);
                    auto bottom = mat.at<float>(r+1, c);
                    if(!std::isnan(top) && !std::isnan(bottom)) {
                        mat.at<float>(r, c) = (top+bottom)/2;
                    }
                }
            }
        }
    }
}


cv::Size CameraCalibration::getSize()
{
    return cv::Size(image->width(), image->height());
}

cv::Mat CameraCalibration::getNewCameraMatrix()
{
    if(model == Model::FISHEYE) {
        cv::Mat newIntrinsics;
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(intrinsics, coeff, getSize(), cv::noArray(), newIntrinsics);
        return newIntrinsics;
    } else {
        return cv::getOptimalNewCameraMatrix(intrinsics, coeff, getSize(), config().get<float>("scale_factor", 1));
    }
}


bool CameraCalibration::handlePatternTop(int midX, int cols) {
	if (midX < cols * 2 / 5) {
		if (topLeft < nframesIntr) {
			topLeft++;
			return true;
		}
	}
	else if (midX < cols * 3 / 5) {
		if (topMid < nframesIntr) {
			topMid++;
			return true;
		}
	}
	else {
		if (topRight < nframesIntr) {
			topRight++;
			return true;
		}
	}

	return false;
}

bool CameraCalibration::handlePatternBottom(int midX, int cols) {
	if (midX < cols * 2 / 5) {
		if (bottomLeft < nframesIntr) {
			bottomLeft++;
			return true;
		}
	}
	else if (midX < cols * 3 / 5) {
		if (bottomMid < nframesIntr) {
			bottomMid++;
			return true;
		}
	}
	else {
		if (bottomRight < nframesIntr) {
			bottomRight++;
			return true;
		}
	}
  return false;
}

bool CameraCalibration::checkFoundPattern(cv::Mat corners, cv::Mat view) {
  cv::Point patTop = corners.at<cv::Point>(0, 0);
  cv::Point patBot = corners.at<cv::Point>(patternSize.height * patternSize.width - 1, 0);

	int midX = (patBot.x + patTop.x) / 2, midY = (patTop.y + patBot.y) / 2;
	bool validNew = false;

	if (midY < view.rows / 2) {
		validNew = handlePatternTop(midX, view.cols);
	}
	else {
		validNew = handlePatternBottom(midX, view.cols);
	}

	if (validNew) {
    cv::Mat poly = cv::Mat::zeros(patternOverlay.size(), patternOverlay.type());

    cv::Point outline[] = { patTop,
			corners.at<cv::Point>(patternSize.width - 1, 0),
			patBot,
			corners.at<cv::Point>((patternSize.height - 1) * (patternSize.width), 0) };
    cv::fillConvexPoly(poly, outline, 4, cv::Scalar(0, 255, 0));

    cv::add(patternOverlay, 0.4 * poly, patternOverlay);
	}

	return validNew;
}


