
#include "utils.h"

using namespace std;
using namespace torch::indexing;


void create_anchor_retinaface(std::vector<float> &anchor, int w, int h)
{
//    anchor.reserve(num_boxes);
    anchor.clear();
    std::vector<std::vector<int> > feature_map(3), min_sizes(3);
    float steps[] = {8, 16, 32};
    for (int i = 0; i < feature_map.size(); ++i) {
        feature_map[i].push_back(ceil(h/steps[i]));
        feature_map[i].push_back(ceil(w/steps[i]));
    }
    std::vector<int> minsize1 = {16, 32};
    min_sizes[0] = minsize1;
    std::vector<int> minsize2 = {64, 128};
    min_sizes[1] = minsize2;
    std::vector<int> minsize3 = {256, 512};
    min_sizes[2] = minsize3;

    for (int k = 0; k < feature_map.size(); ++k)
    {
        std::vector<int> min_size = min_sizes[k];
        for (int i = 0; i < feature_map[k][0]; ++i)
        {
            for (int j = 0; j < feature_map[k][1]; ++j)
            {
                for (int l = 0; l < min_size.size(); ++l)
                {
                    float s_kx = min_size[l]*1.0/w;
                    float s_ky = min_size[l]*1.0/h;
                    float cx = (j + 0.5) * steps[k]/w;
                    float cy = (i + 0.5) * steps[k]/h;
                    // box axil = {cx, cy, s_kx, s_ky};
                    // anchor.push_back(axil);
                    anchor.push_back(cx);
                    anchor.push_back(cy);
                    anchor.push_back(s_kx);
                    anchor.push_back(s_ky);
                    // torch::Tensor axil = torch::tensor({cx, cy, s_kx, s_ky}, {torch::kFloat64});
                    // cout << axil << endl;
                    // anchor = torch::stack([anchor, axil], dim=0)
                }
            }
        }

    }

}





void nms(std::vector<std::vector<double>> &all_result_tensors_vect, double NMS_THRESH,std::vector<double> &vArea)
{
    // std::vector<float>vArea(all_result_tensors_vect.size());
    for (int i = 0; i < int(all_result_tensors_vect.size()); ++i)
    {
        vArea[i] = (all_result_tensors_vect.at(i)[2] - all_result_tensors_vect.at(i)[0] + 1)
                   * (all_result_tensors_vect.at(i)[3] - all_result_tensors_vect.at(i)[1] + 1);
    }
    for (int i = 0; i < int(all_result_tensors_vect.size()); ++i)
    {
        for (int j = i + 1; j < int(all_result_tensors_vect.size());)
        {
            double xx1 = std::max(all_result_tensors_vect[i][0], all_result_tensors_vect[j][0]);
            double yy1 = std::max(all_result_tensors_vect[i][1], all_result_tensors_vect[j][1]);
            double xx2 = std::min(all_result_tensors_vect[i][2], all_result_tensors_vect[j][2]);
            double yy2 = std::min(all_result_tensors_vect[i][3], all_result_tensors_vect[j][3]);
            double w = std::max(double(0), xx2 - xx1 + 1);
            double h = std::max(double(0), yy2 - yy1 + 1);
            double   inter = w * h;
            double ovr = inter / (vArea[i] + vArea[j] - inter);
            double socre_i = all_result_tensors_vect[i][4];
            double socre_j = all_result_tensors_vect[j][4];
            if (ovr >= NMS_THRESH)
            {
                if (socre_i > socre_j){
                // cout<<" stay:"<<all_result_tensors_vect[0+i][4].item<float>()<<endl;
                // cout<<" erase:"<<all_result_tensors_vect[0+j][4].item<float>()<<endl;
                all_result_tensors_vect.erase(all_result_tensors_vect.begin() + j);
                vArea.erase(vArea.begin() + j);
                }
                else{
                // cout<<"erase:"<<all_result_tensors_vect[all_result_tensors_vect.begin() + i][4]].item<float>();
                // cout<<"stay:"<<all_result_tensors_vect[all_result_tensors_vect.begin() + i][4]].item<float>();
                all_result_tensors_vect.erase(all_result_tensors_vect.begin() + i);
                vArea.erase(vArea.begin() + i);
                i = i - 1 ;
                break;

                }
            }
            else
            {
                j++;
            }
        }
    }
}


torch::Tensor decode(torch::Tensor bbox, torch::Tensor anchor)
{
    auto bbox_slice = bbox.index({"...", Slice(None, 2)});
    auto bbox_2slice = bbox.index({"...", Slice(2, None)});
    auto anchor_slice = anchor.index({"...", Slice(None, 2)});
    auto anchor_2lice = anchor.index({"...", Slice(2, None)});


    torch::Tensor boxes = torch::cat({anchor_slice + bbox_slice * 0.1 * anchor_2lice,
                    anchor_2lice * torch::exp(bbox_2slice * 0.2)}, 1);

    auto boxes_slice = boxes.index({"...", Slice(None, 2)});
    // cout << boxes_slice << endl;
    auto boxes_2slice = boxes.index({"...", Slice(2, None)});
    // cout << boxes_2slice << endl;
   
    boxes_slice = boxes_slice - (boxes_2slice / 2.0);
    // cout << boxes_slice << endl;
    boxes_2slice += boxes_slice;
    // cout << boxes_2slice << endl;

    torch::Tensor result = torch::cat({boxes_slice, boxes_2slice}, 1);
    // cout << result << endl;
    return result;
}


torch::Tensor decode_landm(torch::Tensor pre, torch::Tensor anchor)
{
    auto pre_slice =  pre.index({"...", Slice(None, 2)});
    auto pre_slice24 =  pre.index({"...", Slice(2, 4)});
    auto pre_slice46 = pre.index({"...", Slice(4, 6)});
    auto pre_slice68 = pre.index({"...", Slice(6, 8)});
    auto pre_slice810 = pre.index({"...", Slice(8,10)});
    auto anchor_slice = anchor.index({"...", Slice(None, 2)});
    auto anchor_2lice = anchor.index({"...", Slice(2, None)});

    auto landms = torch::cat({anchor_slice + pre_slice* 0.1 * anchor_2lice,
                        anchor_slice + pre_slice24 *  0.1 * anchor_2lice,
                        anchor_slice + pre_slice46 *  0.1 * anchor_2lice,
                        anchor_slice + pre_slice68 *  0.1 * anchor_2lice,
                        anchor_slice + pre_slice810 *  0.1 * anchor_2lice
                        }, 1);
    return landms;
}


torch::Tensor retinaface_correct_boxes(torch::Tensor boxes, torch::Tensor landmarks, const double input_shape[],const double image_shape[]){
    // cout<< "result boxes:" << result << endl;
    cout<< "input_shape:" <<  *input_shape << endl;
    cout<<  "image_shape:" << *image_shape << endl;
    auto w = input_shape[0]/image_shape[0];
    auto h = input_shape[1]/image_shape[1];
    cout<< w  << endl;
    cout<< h << endl;
    auto m = std::min(w , h);
    // auto min = torch::min(input_shape/image_shape);
    cout<< "min" << m << endl;
    double new_shape[] = {image_shape[0]*m, image_shape[1]*m};
    cout<< "new_shape" << *new_shape << endl;
    double offset[] = {(input_shape[0] - new_shape[0])/float(2)/input_shape[0], (input_shape[1] - new_shape[1])/float(2)/input_shape[1]};

    cout<<  "offset" << offset[0] <<  endl;
    cout<<  "offset" << offset[1] <<  endl;
    double scale[] = {input_shape[0]/new_shape[0], input_shape[1]/new_shape[1]};
    
    double scale_for_boxs[] = {scale[1], scale[0], scale[1], scale[0]};
    double scale_for_landmarks[] = {scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0], scale[1], scale[0]};

    double offset_for_boxs[] = {offset[1], offset[0], offset[1],offset[0]};
    double offset_for_landmarks[] = {offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0], offset[1], offset[0]};


    auto optt = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor scale_for_boxs_t = torch::from_blob(scale_for_boxs, {4}, optt);
    torch::Tensor scale_for_landmarks_t = torch::from_blob(scale_for_landmarks, {10}, optt);

    torch::Tensor offset_for_boxs_t = torch::from_blob(offset_for_boxs, {4}, optt);
    torch::Tensor offset_for_landmarks_t = torch::from_blob(offset_for_landmarks, {10}, optt);

    auto result_boxes = (boxes-offset_for_boxs_t)*scale_for_boxs_t;
    auto result_landmarks = (landmarks - offset_for_landmarks_t)*scale_for_landmarks_t;

    // cout << result_boxes << endl;
    // cout << landmarks <<endl;
    // result[:,:4] = (result[:,:4] - np.array(offset_for_boxs)) * np.array(scale_for_boxs)
    // result[:,5:] = (result[:,5:] - np.array(offset_for_landmarks)) * np.array(scale_for_landmarks)

    return result_boxes;
}

