#pragma once
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <array>

void create_anchor_retinaface(std::vector<float> &anchor, int w, int h);
void nms(std::vector<std::vector<double>> &all_result_tensors_vect, double NMS_THRESH,std::vector<double> &vArea);
torch::Tensor decode(torch::Tensor bbox, torch::Tensor anchor);
torch::Tensor decode_landm(torch::Tensor pre, torch::Tensor anchor);
torch::Tensor retinaface_correct_boxes(torch::Tensor boxes, torch::Tensor landmarks, const double input_shape[],const double image_shape[]);