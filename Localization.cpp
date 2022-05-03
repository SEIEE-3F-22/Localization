#include <iostream>
#include "opencv2/opencv.hpp"
#include "Undistort.hpp"
#include <cmath>

extern "C" {
#include "apriltag.h"
#include "apriltag_pose.h"
#include "tag16h5.h"
#include "common/getopt.h"
}

#define AREA_THRESHOLD 2000

using namespace std;
using namespace cv;

double getDist(Point p1, Point p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

double getTriangleArea(Point p1, Point p2, Point p3) {
    double a = getDist(p1, p2);
    double b = getDist(p2, p3);
    double c = getDist(p1, p3);
    double p = (a + b + c) / 2;
    return sqrt(p * (p - a) * (p - b) * (p - c));
}

double getQuadrangleArea(Point p1, Point p2, Point p3, Point p4) {
    double s1 = getTriangleArea(p1, p2, p4);
    double s2 = getTriangleArea(p2, p3, p4);
    return (s1 + s2);
}

double convertAngle(double angle) {
    if (angle >= 0.) {
        angle = fmod(angle + M_PI, M_PI * 2) - M_PI;
    } else {
        angle = fmod(angle - M_PI, -M_PI * 2) + M_PI;
    }
    return angle;
}

int main(int argc, char **argv) {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cap open failure!" << std::endl;
        return 0;
    }

    // Load image size
    int inputWidth, inputHeight;
    static Size correctedSize(0, 0);
    if (argc > 2) {
        inputWidth = static_cast<int>(strtol(argv[1], nullptr, 10));
        inputHeight = static_cast<int>(strtol(argv[2], nullptr, 10));
        cout << "Resolution: " << inputWidth << " * " << inputHeight << endl;
        correctedSize = Size(inputWidth, inputHeight);
    }


    // AprilTag initialization
    getopt_t *getopt = getopt_create();

    getopt_add_bool(getopt, 'h', "help", 0, "Show this help");
    getopt_add_bool(getopt, 'd', "debug", 0, "Enable debugging output (slow)");
    getopt_add_bool(getopt, 'q', "quiet", 0, "Reduce output");
    getopt_add_string(getopt, 'f', "family", "tag16h5", "Tag family to use");
    getopt_add_int(getopt, 't', "threads", "1", "Use this many CPU threads");
    getopt_add_double(getopt, 'x', "decimate", "2.0", "Decimate input image by this factor");
    getopt_add_double(getopt, 'b', "blur", "0.0", "Apply low-pass blur to input");
    getopt_add_bool(getopt, '0', "refine-edges", 1, "Spend more time trying to align edges of tags");

    if (!getopt_parse(getopt, argc, argv, 1) ||
        getopt_get_bool(getopt, "help")) {
        printf("Usage: %s [options]\n", argv[0]);
        getopt_do_usage(getopt);
        exit(0);
    }

    cout << "Enabling AprilTag" << endl;

    // Initialize tag detector with options
    apriltag_family_t *tf = nullptr;
    const char *famname = getopt_get_string(getopt, "family");
    if (!strcmp(famname, "tag16h5")) {
        tf = tag16h5_create();
    } else {
        printf("Unrecognized tag family name. Use e.g. \"tag16h5\".\n");
        exit(-1);
    }

    apriltag_detector_t *td = apriltag_detector_create();
    apriltag_detector_add_family(td, tf);
    td->quad_decimate = getopt_get_double(getopt, "decimate");
    td->quad_sigma = getopt_get_double(getopt, "blur");
    td->nthreads = getopt_get_int(getopt, "threads");
    td->debug = getopt_get_bool(getopt, "debug");
    td->refine_edges = getopt_get_bool(getopt, "refine-edges");

    Mat frame, gray, corrected;

    Matx33d intrinsics;
    ifstream intrinsicFile("intrinsics.txt");
    for(auto i = 0; i < 3; i++) {
        for (auto j = 0; j < 3; j++) {
            intrinsicFile >> intrinsics(i, j);
        }
    }

    apriltag_detection_info_t info;
    info.tagsize = 0.075;  // The size of the tag in meters
    info.fx = intrinsics(0, 0);  // The camera's focal length (in pixels)
    info.fy = intrinsics(1, 1);
    info.cx = intrinsics(0, 2);  // The camera's focal center (in pixels)
    info.cy = intrinsics(1, 2);

    while (true) {
        cap >> frame;
        if (!frame.empty()) {
            Undistort::GetInstance(correctedSize).ExecuteUndistort(frame, corrected);
        } else {
            std::cerr << "Image acquisition error!" << std::endl;
            continue;
        }

        cvtColor(corrected, gray, COLOR_BGR2GRAY);

        // Make an image_u8_t header for the Mat data
        image_u8_t im = {.width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
        };
        zarray_t *detections = apriltag_detector_detect(td, &im);

        // Draw detection outlines
        int index = 0;
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);

            double area = getQuadrangleArea(
                    Point(det->p[0][0], det->p[0][1]),
                    Point(det->p[1][0], det->p[1][1]),
                    Point(det->p[2][0], det->p[2][1]),
                    Point(det->p[3][0], det->p[3][1]));
            if (area > AREA_THRESHOLD) {
                index++;
                info.det = det;
                apriltag_pose_t pose;
                estimate_pose_for_tag_homography(&info, &pose);

                double x = pose.t->data[0];
                double y = pose.t->data[1];
                double z = pose.t->data[2];
                double yaw = 180 * convertAngle(atan2(pose.R->data[3], pose.R->data[0])) / M_PI;
                double pitch = 180 * convertAngle(asin(-pose.R->data[6])) / M_PI;
                double roll = 180 * convertAngle(atan2(pose.R->data[7], pose.R->data[8])) / M_PI;

                cout << "Tag " << index << endl;
                cout << "\tx: " << x << "\ty: " << y << "\tz: " << z << endl;
                cout << "\tyaw: " << yaw << "\tpitch: " << pitch << "\troll: " << roll << endl;

                line(corrected, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[1][0], det->p[1][1]),
                     Scalar(0, 0xff, 0), 2);
                line(corrected, Point(det->p[0][0], det->p[0][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0, 0, 0xff), 2);
                line(corrected, Point(det->p[1][0], det->p[1][1]),
                     Point(det->p[2][0], det->p[2][1]),
                     Scalar(0xff, 0, 0), 2);
                line(corrected, Point(det->p[2][0], det->p[2][1]),
                     Point(det->p[3][0], det->p[3][1]),
                     Scalar(0xff, 0, 0), 2);

                stringstream ss;
                ss << det->id;
                String text = ss.str();
                int fontface = FONT_HERSHEY_SCRIPT_SIMPLEX;
                double fontscale = 1.0;
                int baseline;
                Size textsize = getTextSize(text, fontface, fontscale, 2,
                                            &baseline);
                putText(corrected, text, Point(det->c[0] - textsize.width / 2,
                                               det->c[1] + textsize.height / 2),
                        fontface, fontscale, Scalar(0xff, 0x99, 0), 2);
            }
        }
        apriltag_detections_destroy(detections);
        imshow("Tag Detections", corrected);

        auto input = waitKey(1);
        if ('q' == input)
            break;
    }
    cap.release();
    destroyAllWindows();

    apriltag_detector_destroy(td);

    if (!strcmp(famname, "tag16h5")) {
        tag16h5_destroy(tf);
    }

    getopt_destroy(getopt);
    return 0;
}
