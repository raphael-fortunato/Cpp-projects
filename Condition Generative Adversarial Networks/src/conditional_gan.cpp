#include <torch/torch.h>
#include "../include/conditional_gan.h"


using namespace torch;
using namespace std;


namespace gan{

    const int noise_size = 100;
    DiscriminatorImpl::DiscriminatorImpl()
    {
        //model registration is required to be able to call the model parameters
        register_module("layer1_1", layer1_1);
        register_module("layer1_2", layer1_2);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        name = "SavedModels/discriminator.pt";
    }
    Tensor DiscriminatorImpl::forward(Tensor x, Tensor y)
    {
        x = x.view({x.size(0), -1});
        y = y.view({y.size(0), -1});
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
        //model registration is required to be able to call the model parameters
        register_module("layer1_1", layer1_1);
        register_module("layer1_2", layer1_2);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        name = "SavedModels/generator.pt";
    }
    Tensor GeneratorImpl::forward(Tensor x, Tensor y)
    {
        x = x.view({x.size(0), -1});
        y = y.view({y.size(0), -1});
        x = layer1_1->forward(x);
        y = layer1_2->forward(y);
        x = torch::cat({x,y}, 1);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        return x;
    }
}
