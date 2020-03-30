#include <torch/torch.h>
#include <iostream>
#include "../include/conditional_dcgan.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>


using namespace torch;
using namespace std;

namespace dcgan{

    int noise_size = 100;
    DiscriminatorImpl::DiscriminatorImpl()
    {

        register_module("layer1_1", layer1_1);
        register_module("layer1_2", layer1_2);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        name = "SavedModels/discriminator.pt";
    }
    Tensor DiscriminatorImpl::forward(Tensor x, Tensor y)
    {
        y = y.view({y.size(0), y.size(1), 1, 1});
        y = y.expand({y.size(0), y.size(1), 28, 28});
        std::cout << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << std::endl;
        x = layer1_1->forward(x);
        y = layer1_2->forward(y);
        x = torch::cat({x,y}, 1);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        return x;
    }

    GeneratorImpl::GeneratorImpl()
    {
        register_module("layer1_1", layer1_1);
        register_module("layer1_2", layer1_2);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        name = "SavedModels/generator.pt";
    }

    Tensor GeneratorImpl::forward(Tensor x, Tensor y)
    {
        x = layer1_1->forward(x);
        y = layer1_2->forward(y);
        x = torch::cat({x,y}, 1);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        return x;
    }

  
}