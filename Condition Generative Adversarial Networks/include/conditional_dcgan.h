#ifndef DCGAN_H
#define DCGAN_H
#include <torch/torch.h>

namespace dcgan{


class GeneratorImpl : public torch::nn::Module
{
    public:
        GeneratorImpl();
        torch::Tensor forward(torch::Tensor, torch::Tensor);
        std::string name;
    private:
        torch::nn::Sequential layer1_1 {
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(100, 256, 4).bias(false)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        };
        torch::nn::Sequential layer1_2 {
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(10, 256, 4).bias(false)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        };
        torch::nn::Sequential layer2 {
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(512, 128, 3)
            .stride(2)
            .padding(1)
            .bias(false)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        };
        torch::nn::Sequential layer3 {
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(128, 64, 4)
            .stride(2)
            .padding(1)            
            .bias(false)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        };
        torch::nn::Sequential layer4 {
            torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(64, 1, 4)
            .stride(2)
            .padding(1)            
            .bias(false)),
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
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1).bias(false)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2)),
        };
        torch::nn::Sequential layer1_2 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 64, 4).stride(2).padding(1).bias(false)),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
        };
        torch::nn::Sequential layer2 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1).bias(false)),
            torch::nn::BatchNorm2d(256),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
        };
        torch::nn::Sequential layer3 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 4).stride(2).padding(1).bias(false)),
            torch::nn::BatchNorm2d(128),
            torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.2))
        };
        torch::nn::Sequential layer4 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 1, 3).stride(1).padding(0).bias(false)),
            torch::nn::Tanh()
        };
};       
TORCH_MODULE(Discriminator);

}

#endif // !DCGAN_H