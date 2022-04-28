#include <iostream>
#include "opencv2/opencv.hpp"
#include "Undistort.hpp"

extern "C" {
#include "apriltag.h"
#include "tag16h5.h"
#include "common/getopt.h"
}

using namespace std;
using namespace cv;

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

    while (true) {
        Mat frame, corrected;
        cap >> frame;
        if (!frame.empty()) {
            Undistort::GetInstance(correctedSize).ExecuteUndistort(frame, corrected);
        } else {
            std::cerr << "Image acquisition error!" << std::endl;
            continue;
        }

        Mat gray;
        cvtColor(corrected, gray, COLOR_BGR2GRAY);

        // Make an image_u8_t header for the Mat data
        image_u8_t im = {.width = gray.cols,
                .height = gray.rows,
                .stride = gray.cols,
                .buf = gray.data
        };
        zarray_t *detections = apriltag_detector_detect(td, &im);

        // Draw detection outlines
        for (int i = 0; i < zarray_size(detections); i++) {
            apriltag_detection_t *det;
            zarray_get(detections, i, &det);
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
        apriltag_detections_destroy(detections);
        imshow("Tag Detections", corrected);

        auto input = waitKey(1);
        if ('q' == input) break;
    }
    cap.release();
    destroyAllWindows();
    return 0;
}
