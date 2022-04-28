#ifndef CALLIBRATION_FISHEYE_H
#define CALLIBRATION_FISHEYE_H

#include <iostream>
#include <fstream>
#include <dirent.h>
#include <vector>

#include "opencv2/opencv.hpp"

#include "Undistort.hpp"

constexpr int board_w = 9;
constexpr int board_h = 6;
constexpr int NPoints = board_w * board_h;//棋盘格内角点总数
constexpr int squareSize = 23; //mm

#endif //CALLIBRATION_FISHEYE_H
