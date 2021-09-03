// #include <torch/script.h> // One-stop header.
// #include <torch/torch.h>
// #include "opencv2/opencv.hpp"
// #include <opencv2/highgui.hpp>
#include "image_utils.h"

using namespace std;

std::string get_image_type(const cv::Mat& img, bool more_info=true) 
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans + '0');
   
    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}

void show_image(cv::Mat& img, std::string title)
{
    std::string image_type = get_image_type(img);
    cv::namedWindow(title + " type:" + image_type, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title + " type:" + image_type, img);
    cv::waitKey(0);
}

at::Tensor transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 })
{
    std::cout << "############### transpose ############" << std::endl;
    std::cout << "shape before : " << tensor.sizes() << std::endl;
    tensor = tensor.permute(dims);
    std::cout << "shape after : " << tensor.sizes() << std::endl;
    std::cout << "######################################" << std::endl;
    return tensor;
}

at::Tensor ToTensor(cv::Mat img, bool show_output = false, bool unsqueeze=false, int unsqueeze_dim = 0)
{
    std::cout << "image shape: " << img.size() << std::endl;
    at::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kByte);

    if (unsqueeze)
    {
        tensor_image.unsqueeze_(unsqueeze_dim);
        std::cout << "tensors new shape: " << tensor_image.sizes() << std::endl;
    }
    
    if (show_output)
    {
        std::cout << tensor_image.slice(2, 0, 1) << std::endl;
    }
    std::cout << "tenor shape: " << tensor_image.sizes() << std::endl;
    return tensor_image;
}

std::vector<torch::jit::IValue> ToInput(at::Tensor tensor_image)
{
    // Create a vector of inputs.
    return std::vector<torch::jit::IValue>{tensor_image};
}

cv::Mat ToCvImage(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    try
    {
        cv::Mat output_mat(cv::Size{ height, width }, CV_8UC3, tensor.data_ptr<uchar>());
        
        // show_image(output_mat, "converted image from tensor");
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    return cv::Mat(height, width, CV_8UC3);
}


LetterboxInfo letterboxResize(const cv::Mat& img)
{
    LetterboxInfo letterboxInfo;
    int width = img.cols,
       height = img.rows;
    cout<<"image size("<<img.cols<<","<<img.rows<<")"<<endl;
    letterboxInfo.resizedImage = cv::Mat::zeros( 480, 640, img.type());

    // int max_dim = ( width >= height ) ? width : height;
    letterboxInfo.xscale = ( ( float ) 640 ) / width;
    letterboxInfo.yscale = ( ( float ) 480 ) / height;
    cout<<"image scale("<<letterboxInfo.xscale<<","<<letterboxInfo.yscale<<")"<<endl;
    cout<<"w/h:"<< double(width)/double(height) <<endl;
    cv::Rect roi;
    if (double(width)/double(height) > double(4./3.))
    {
        roi.width = 640;
        roi.x = 0;
        roi.height = height * letterboxInfo.xscale;
        roi.y = 0;
        letterboxInfo.vertical = false;
        letterboxInfo.scale = letterboxInfo.xscale;
    }
    else
    {
        roi.y = 0;
        roi.height = 480;
        roi.width = width * letterboxInfo.yscale;
        roi.x = 0;
        letterboxInfo.vertical = true;
        letterboxInfo.scale = letterboxInfo.yscale;
    }
    cout<<"roi scale("<<roi.width<<","<<roi.height<<")"<<endl;
    cv::resize(img, letterboxInfo.resizedImage( roi ), roi.size());

    return letterboxInfo;
}


cv::Mat histogram(cv::Mat image){
	cv::Mat ycrcb;
	cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
	vector<cv::Mat>channels;
	cv::split(ycrcb, channels);
	cv::equalizeHist(channels[0], channels[0]);
	cv::Mat result;
	cv::merge(channels, ycrcb);
	cv::cvtColor(ycrcb, result, cv::COLOR_YCrCb2BGR);
    return result;


}

cv::Mat adopt_histogram(cv::Mat image){
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);

    cv::Mat dst,result;
    clahe->apply(gray,dst);
    cv::cvtColor(dst, result, cv::COLOR_GRAY2BGR);
    cv::waitKey();
    
    return result;
}


cv::Mat sharpen(cv::Mat image){
    cv::Mat sharpen_op = (cv::Mat_<char>(3, 3) << 0, -1, 0,
		-1, 5, -1,
		0, -1, 0);

	cv::Mat result;
	cv::filter2D(image, result, CV_32F, sharpen_op);
	cv::convertScaleAbs(result, result);

	imshow("sharpen image", result);
    return result;
}
