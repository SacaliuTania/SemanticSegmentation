#include <opencv2/opencv.hpp>
#include <iostream>

#include<torch/torch.h>
#include<torch/script.h>


//e modelul in python rulat si tot
    //trebuie exportat in c++
    //decodificarea tensorului
    //ceva ceva jtrace
    //pe o singura imagine, nu trebuie pe tot folder ul
    //trebuie sa imi antrenez modelul in python
    //codul din python trebuie transformat in c++ pentru optimizare
    //fara onnx, fara cuda


int main() {

    //PREPROCESARE
    //citire imagine
    //cv::Mat img = cv::imread("D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png");
    //cv::Mat img = cv::imread("D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/bonn/bonn_000045_000019_leftImg8bit.png");
    //cv::Mat img = cv::imread("D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/munich/munich_000038_000019_leftImg8bit.png");
    cv::Mat img = cv::imread("D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/munich/munich_000069_000019_leftImg8bit.png");
    //cv::Mat img = cv::imread("D:/facultate/An3/An3_sem2/segmentare_semantica/leftImg8bit_trainvaltest/leftImg8bit/test/bonn/bonn_000015_000019_leftImg8bit.png");

    std::cout << "Img loaded" << std::endl;

    if (img.empty()) {
        std::cout << "Could not load image" << std::endl;
        return -1;
    }

    cv::resize(img, img, cv::Size(1024, 512));

    //conversie RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    //normalizare si conversie NCHW
    const int H = 512;
    const int W = 1024;
    const float mean[] = {0.485f, 0.456f, 0.406f};
    const float stdv[] = {0.229f, 0.224f, 0.225f};

    //open cv stocheaza imaginea rgb
    //modelul vrea rrr ggg bbb
    //blob pentru stocarea datelor asa cum le vrea modelul

    std::vector<float> blob(3 * H * W);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            auto px = img.at<cv::Vec3b>(r, c);
            for (int ch = 0; ch < 3; ++ch)
                blob[ch * H * W + r * W + c] = (px[ch] / 255.0f - mean[ch]) / stdv[ch];
        }

    //LibTorch
    //incarcarea modelului
    torch::jit::script::Module model;
    std::cout << "Loading Model" << std::endl;
 try {
        //model = torch::jit::load("D:/facultate/An3/An3_sem2/segmentare_semantica/segformer_cityscapes.pt");
     model = torch::jit::load("D:/facultate/An3/An3_sem2/segmentare_semantica/deeplabv3_antrenat.pt");
        model.eval();
        std::cout<< "Model incarcat";
    }
    catch (std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    //INFERENTA

    torch::Tensor input_tensor =
          torch::from_blob(blob.data(), {1, 3, H, W}, torch::kFloat32).clone();

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    torch::Tensor output;
    {
        torch::NoGradGuard no_grad;
        std::cout << "Rulare forward" << std::endl;

        output = model.forward(inputs).toTensor();
        std::cout << "Output" << std::endl;
    }


    //POSTPROCESARE

    torch::Tensor pred = output.argmax(1).squeeze(0).to(torch::kU8);
    auto pred_cpu = pred.contiguous().cpu();
    uint8_t* pred_data = pred_cpu.data_ptr<uint8_t>();

    const uint8_t COLORMAP[19][3] = {
        {128,  64, 128},  //  0 road
        {244,  35, 232},  //  1 sidewalk
        { 70,  70,  70},  //  2 building
        {102, 102, 156},  //  3 wall
        {190, 153, 153},  //  4 fence
        {153, 153, 153},  //  5 pole
        {250, 170,  30},  //  6 traffic light
        {220, 220,   0},  //  7 traffic sign
        {107, 142,  35},  //  8 vegetation
        {152, 251, 152},  //  9 terrain
        { 70, 130, 180},  // 10 sky
        {220,  20,  60},  // 11 person
        {255,   0,   0},  // 12 rider
        {  0,   0, 142},  // 13 car
        {  0,   0,  70},  // 14 truck
        {  0,  60, 100},  // 15 bus
        {  0,  80, 100},  // 16 train
        {  0,   0, 230},  // 17 motorcycle
        {119,  11,  32},  // 18 bicycle
    };

    cv::Mat seg_color(H, W, CV_8UC3);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            uint8_t cls = pred_data[r * W + c];
            seg_color.at<cv::Vec3b>(r, c) = {
                COLORMAP[cls][2],
                COLORMAP[cls][1],
                COLORMAP[cls][0]
            };
        }

    //overlay
    const float ALPHA = 0.55f;

    //imaginea rgb trebuie folosita ca bgr
    cv::Mat img_bgr;
    cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
    cv::Mat overlay;
    cv::addWeighted(seg_color, ALPHA, img_bgr, 1.0f - ALPHA, 0.0, overlay);


    //show
    //cv::imshow("imagine", img);

    // cv::Mat combined;
    // cv::hconcat(std::vector<cv::Mat>{img_bgr, seg_color, overlay}, combined);

    cv::Mat img_small, seg_small, overlay_small;
    cv::resize(img_bgr, img_small, cv::Size(640, 320));
    cv::resize(seg_color, seg_small, cv::Size(640, 320));
    cv::resize(overlay, overlay_small, cv::Size(640, 320));

    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{img_small, seg_small, overlay_small}, combined);

    cv::imwrite("segmentare_cityscapes_cpp.png", combined);
    std::cout << "[Salvat] segmentare_cityscapes_cpp.png\n";

    cv::imshow("Original | Segmentare | Overlay", combined);
    cv::waitKey(0);
    return 0;
}
