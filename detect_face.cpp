#pragma once
#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <nlohmann/json.hpp>
#include "opencv2/opencv.hpp"
#include <iostream>
#include <array>
#include <fstream>
#include <string.h>
// #include <ctime>
#include "image_utils.h"
#include "utils.h"


using namespace std;
using namespace torch::indexing;


// g++ detect_face.cpp utils.cpp image_utils.cpp -I/libtorch/include -I/libtorch/include/torch/csrc/api/include -I//home/kebin/Downloads/json-develop/single_include -std=c++17 -L/libtorch/lib/ -ltorch_cpu -lc10  -o output `pkg-config --cflags --libs opencv`



std::vector<vector<double>>detect_face(torch::jit::script::Module retina, cv::Mat img_origin){
    cv::Mat img;
    img_origin.copyTo(img) ;
    double image_arr[] = { float(img.rows), float(img.cols) };
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor image_shape = torch::from_blob(image_arr, {img.rows, img.cols}, options);

    cv::Size scale(640, 480);
    auto letterboxInfo = letterboxResize(img);
    double rows = double(img.rows);
    double cols = double(img.cols);
    if (letterboxInfo.vertical){
        cols = double(img.cols+((640-img.cols*letterboxInfo.scale)/letterboxInfo.scale));
    }else{
        rows = double(img.rows+((480-img.rows*letterboxInfo.scale)/letterboxInfo.scale));
    }
    // cout<<"rows*scale:"<< img.rows*letterboxInfo.scale <<endl;
    show_image(letterboxInfo.resizedImage, "msg");
    double image_scale[] = {cols, rows, cols, rows};
    double image_scale_for_landmarks[] = { cols, rows, cols, rows, cols, rows, cols, rows, cols, rows};
   
    torch::Tensor scale_t = torch::from_blob(image_scale, {4}, options);
    torch::Tensor scale_for_landmarks_t = torch::from_blob(image_scale_for_landmarks, {10}, options);
    // cout<< "scale_t:"<< scale_t <<endl;
    // cout<< "scale_for_landmarks_t:"<< scale_for_landmarks_t <<endl;

    double input_arr[] = { 640, 480 };
    torch::Tensor input_shape = torch::from_blob(input_arr, {2}, options);

    // convert the cvimage into tensor
    auto tensor = ToTensor(letterboxInfo.resizedImage,false,false,0);
    tensor = tensor.toType(c10::kFloat);
    
    // swap axis 
    tensor = transpose(tensor, { (2),(0),(1) });
    // //add batch dim (an inplace operation just like in pytorch)
    tensor.unsqueeze_(0);
    tensor.to(c10::DeviceType::CPU);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    // tend = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    auto output = retina.forward(inputs);

    torch::Tensor bbox = output.toTuple()->elements()[0].toTensor();
    torch::Tensor score = output.toTuple()->elements()[1].toTensor();
    torch::Tensor landmark = output.toTuple()->elements()[2].toTensor();
    // bbox = retinaface_correct_boxes(bbox, landmark, image_arr, input_arr);
    
    std::cout << "ok\n";

    std::vector<float> anchor;
    cout<< "clos:"<< img.cols<<endl;
    cout<< "rows:"<< img.rows<<endl;
    // create_anchor_retinaface(anchor, img.cols, img.rows);
    create_anchor_retinaface(anchor, 640, 480);
    // cout << anchor.size()/4 << endl;
    
    torch::Tensor ancor_t = torch::from_blob(anchor.data(), {int(anchor.size()/4),4});
    
    auto boxes_result = decode(bbox[0], ancor_t);
    // cout << boxes_result << endl;.
    auto score_result =  score[0].index({"...", Slice(1, 2)});
    // cout << score_result << endl;
    auto landm_result = decode_landm(landmark[0], ancor_t);
    // cout << landm_result << endl;
    std::vector<std::vector<double>> all_result_tensors_vect;
    for (int i=0; i < int(anchor.size()/4); ++i){
        if (score_result[i].item<float>() > 0.7){
            // cout <<"boxes result:"<< boxes_result[i]<< endl;
            auto final = boxes_result[i]*scale_t;
            auto final_landm = landm_result[i]*scale_for_landmarks_t;
            for (int j=0;j < torch::size(final,0); j++){
                if (j==0 || j ==2){
                    if (final[j].item<float>()> img.rows){
                        final[j] = float(img.rows);
                    }
                }
                else{
                    if (final_landm[j].item<float>()> img.cols){
                        final_landm[j] = float(img.cols);
                    }
                }
                if (final[j].item<float>() < 0){
                    final[j] = float(0.);
                }
                if (final_landm[j].item<float>() < 0){
                    final_landm[j] = float(0.);
                }



            }
            auto all_result = torch::cat({final, score_result[i],final_landm},0);
            // cout<< all_result <<endl;
            auto r_ptr = all_result.data_ptr<double>();
            std::vector<double> result{r_ptr, r_ptr + all_result.size(0)};
            // cout << result <<endl;
            all_result_tensors_vect.push_back(result);
           
        }

    }
    return all_result_tensors_vect;
    

}
std::vector<vector<double>>get_faces(std::vector<vector<double>> all_result_tensors_vect){
    std::vector<double>vArea(all_result_tensors_vect.size());
    nms(all_result_tensors_vect, 0.4, vArea);
    
    int maxElementIndex = std::max_element(vArea.begin(),vArea.end()) - vArea.begin();

    return all_result_tensors_vect;

}

std::vector<double> get_biggest_face(std::vector<vector<double>> all_result_tensors_vect){
    std::vector<double>vArea(all_result_tensors_vect.size());
    nms(all_result_tensors_vect, 0.4, vArea);
    int maxElementIndex = std::max_element(vArea.begin(),vArea.end()) - vArea.begin();
    vector<double> biggest_face = all_result_tensors_vect[maxElementIndex];
    return biggest_face;
}

void draw_box(cv::Mat img_origin,std::vector<double> &result_tensors_vect){
    cv::rectangle(img_origin, cv::Point(int(result_tensors_vect[0]),int(result_tensors_vect[1])), cv::Point(int(result_tensors_vect[2]),int(result_tensors_vect[3])), cv::Scalar(0, 255, 0));
    cv::circle(img_origin, cv::Point(int(result_tensors_vect[5]),int(result_tensors_vect[6])), 2, cv::Scalar(0, 255, 255), -1);
    cv::circle(img_origin, cv::Point(int(result_tensors_vect[7]),int(result_tensors_vect[8])), 2, cv::Scalar(0, 255, 255), -1);
    cv::circle(img_origin, cv::Point(int(result_tensors_vect[9]),int(result_tensors_vect[10])), 2, cv::Scalar(0, 255, 255), -1);
    cv::circle(img_origin, cv::Point(int(result_tensors_vect[11]),int(result_tensors_vect[12])), 2, cv::Scalar(0, 255, 255), -1);
    cv::circle(img_origin, cv::Point(int(result_tensors_vect[13]),int(result_tensors_vect[14])), 2, cv::Scalar(0, 255, 255), -1);
}


int main(int argc, const char* argv[]) 
{
    std::ifstream config_file("config.json", std::ifstream::binary);
    nlohmann::json cfg;
    config_file >> cfg;
    torch::jit::script::Module retina;
    retina = torch::jit::load(cfg["model_path"]);
    std::string msg = "sample image";
    auto currentPath = cfg["input_image"];
    auto img_origin = cv::imread(currentPath);
    if (img_origin.empty()) {
		printf("could not load image...\n");
		return -1;
	}

    img_origin = histogram(img_origin);
    // img_origin = adopt_histogram(img_origin);

    std::vector<vector<double>> result = detect_face(retina, img_origin);
    if (cfg["detect_mode"] == "single") {
        auto result_tensors_vect = get_biggest_face(result);
        draw_box(img_origin, result_tensors_vect);
       
    }
    else{
        auto all_result_tensors_vect = get_faces(result);
        for(int i=0; i < int(all_result_tensors_vect.size()); ++i){
            cout<<int(all_result_tensors_vect[i][0])<<","<<int(all_result_tensors_vect[i][1])<<","<< int(all_result_tensors_vect[i][2])<<","<<int(all_result_tensors_vect[i][3])<<endl;
            draw_box(img_origin, all_result_tensors_vect[i]);
            }
        }

    show_image(img_origin, msg);

    return 0;
}







