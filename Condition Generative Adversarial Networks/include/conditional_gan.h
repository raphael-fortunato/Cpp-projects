#ifndef GAN_H
#define GAN_H
#include <torch/torch.h>
#include <iostream>

namespace gan{

class GeneratorImpl : public torch::nn::Module
{
    public:
        GeneratorImpl();
        torch::Tensor forward(torch::Tensor, torch::Tensor);
        std::string name;
    private:
        torch::nn::Sequential layer1_1 {
        torch::nn::Linear(torch::nn::LinearOptions(100,256)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256)),
        torch::nn::ReLU()
        };
        torch::nn::Sequential layer1_2 {
        torch::nn::Linear(torch::nn::LinearOptions(10,256)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256)),
        torch::nn::ReLU()
        };
        torch::nn::Sequential layer2 {
        torch::nn::Linear(torch::nn::LinearOptions(512,512)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)),
        torch::nn::ReLU()
        };
        torch::nn::Sequential layer3 {
        torch::nn::Linear(torch::nn::LinearOptions(512,1024)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(1024)),
        torch::nn::ReLU()
        };
        torch::nn::Sequential layer4 {
        torch::nn::Linear(torch::nn::LinearOptions(1024,784)),
        torch::nn::Sigmoid()
        };

};
TORCH_MODULE(Generator);

class DiscriminatorImpl : public torch::nn::Module
{
    public:
        DiscriminatorImpl();
        torch::Tensor forward(torch::Tensor, torch::Tensor);
        std::string name;
    private:
        torch::nn::Sequential layer1_1 {
        torch::nn::Linear(torch::nn::LinearOptions(784,512)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
        };
        torch::nn::Sequential layer1_2 {
        torch::nn::Linear(torch::nn::LinearOptions(10,512)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
        };
        torch::nn::Sequential layer2 {
        torch::nn::Linear(torch::nn::LinearOptions(1024,512)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(512)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
        };
        torch::nn::Sequential layer3 {
        torch::nn::Linear(torch::nn::LinearOptions(512,256)),
        torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(256)),
        torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
        };
        torch::nn::Sequential layer4 {
        torch::nn::Linear(torch::nn::LinearOptions(256,1)),
        torch::nn::Sigmoid()
        };
};       
TORCH_MODULE(Discriminator);
}

#endif // !GAN_H
