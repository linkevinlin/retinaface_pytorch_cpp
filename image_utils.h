#pragma once
#include <string>
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui.hpp>


struct LetterboxInfo {
    double scale, xscale, yscale;
    bool vertical;
    cv::Mat resizedImage;

    
} typedef LetterboxInfo;

std::string get_image_type(const cv::Mat& img, bool more_info); 
void show_image(cv::Mat& img, std::string title);
at::Tensor transpose(at::Tensor tensor, c10::IntArrayRef dims);
at::Tensor ToTensor(cv::Mat img, bool show_output, bool unsqueeze, int unsqueeze_dim);
std::vector<torch::jit::IValue>  ToInput(at::Tensor tensor_image);
cv::Mat ToCvImage(at::Tensor tensor);
LetterboxInfo letterboxResize(const cv::Mat& img);
cv::Mat histogram(cv::Mat image);
cv::Mat adopt_histogram(cv::Mat image);
cv::Mat sharpen(cv::Mat image);


