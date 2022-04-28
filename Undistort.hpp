#ifndef LOCALIZATION_UNDISTORT_HPP
#define LOCALIZATION_UNDISTORT_HPP

#include <opencv2/opencv.hpp>
#include <fstream>

using namespace std;
using namespace cv;

constexpr int DEFAULT_IMAGE_WIDTH = 1280;
constexpr int DEFAULT_IMAGE_HEIGHT = 720;

/**
 * @brief Use the params storaged in the files to undistort the raw image.
 * @input intrinsics.txt, dis_coeff.txt
 * @author Xi wang
 * @usage Undistort::GetInstance(correctedSize).ExecuteUndistort(frame, corrected),
where correctedSize is a cv::Size variable with 2 dimensions that indicates the size of undistort image,
frame and corrected are the input and output image with cv::Mat type.
 * @note Singleton design pattern, which means the class can only be constructed once.
 */
class Undistort{
public:
    static Undistort& GetInstance(const cv::Size& _correctedSize);
    bool ExecuteUndistort(Mat& rawImage, Mat& corrected);

protected:
    explicit Undistort(const cv::Size& correctedSize);

private:
    static Undistort* undistort; //Instance
    Mat mapx, mapy;
    cv::Size correctedSize = cv::Size(DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT);

    static bool LoadCalibrationParams(Matx33d& intrinsics, Vec4d& distortion_coeff);
};

Undistort* Undistort::undistort = nullptr;

Undistort::Undistort(const cv::Size& _correctedSize) {
    /**
     * Load calibration parameter from file
     */
    Matx33d intrinsics;//相机内参
    Vec4d distortion_coeff;//相机畸变系数
    LoadCalibrationParams(intrinsics, distortion_coeff);
    fisheye::initUndistortRectifyMap(intrinsics, distortion_coeff, cv::Matx33d::eye(), intrinsics, _correctedSize, CV_16SC2, mapx, mapy);
}

Undistort &Undistort::GetInstance(const cv::Size &_correctedSize) {
    /**
     * Check the size
     */
    if((_correctedSize.height > 100) && (_correctedSize.width > 100)) {
        if (nullptr == undistort){
            undistort = new Undistort(_correctedSize);
            undistort->correctedSize = _correctedSize;
        }
        else if (_correctedSize != undistort->correctedSize) {
            delete undistort;
            undistort = new Undistort(_correctedSize);
            undistort->correctedSize = _correctedSize;
        }
    }
    else {
        if (nullptr == undistort) {
            undistort = new Undistort(undistort->correctedSize);
        }
    }

    return *undistort;
}

bool Undistort::LoadCalibrationParams(Matx33d& intrinsics, Vec4d& distortion_coeff){

    ifstream intrinsicFile("intrinsics.txt");
    ifstream disFile("dis_coeff.txt");

    for(auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            intrinsicFile >> intrinsics(i, j);
        }
    }
    cout << "Intrinsics:\r\n" << intrinsics << endl;
    for(auto i = 0; i < 4; i++){
        disFile >> distortion_coeff(i);
    }
    cout << "Distortion:\r\n" << distortion_coeff << endl;

    return true;
}

bool Undistort::ExecuteUndistort(Mat& rawImage, Mat& corrected) {
    remap(rawImage, corrected, mapx, mapy, INTER_LINEAR, BORDER_TRANSPARENT);
    return true;
}

#endif //LOCALIZATION_UNDISTORT_HPP
