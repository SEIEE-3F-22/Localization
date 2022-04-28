#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace boost::gregorian;
using namespace boost::posix_time;
namespace fs = boost::filesystem;

constexpr auto savePath = "./data/";

int main(int argc, char **argv) {

    // Create directory for image storage
    date today = day_clock::local_day();
    const String dirString = savePath + to_iso_string(today);

    fs::path dir(dirString);
    if (!fs::exists(dir)) {
        create_directories(dir);
        std::cout << "Create directory \"" << dirString << '\"' << std::endl;
    }

    ptime op = second_clock::local_time();
    std::cout << op << std::endl;
    String timeString;

    VideoCapture cap(atoi(argv[0]));
    if (!cap.isOpened()) {
        std::cerr << "Cap open failure!" << std::endl;
        return 0;
    }

    Mat frame;
    namedWindow("img");

    while (true) {
        cap >> frame;
        if (!frame.empty()) {
            imshow("img", frame);
        } else {
            std::cerr << "Image acquisition error!" << std::endl;
        }

        auto input = waitKey(1);
        if ('q' == input) break;
        if ('s' == input) {
            String frameName = dirString;
            op = second_clock::local_time();
            timeString = to_iso_string(op);
            frameName.append("/").append(timeString).append(".jpg");
            imwrite(frameName, frame);
            std::cout << "Save frame in " << frameName << std::endl;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}