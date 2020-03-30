#ifndef UTILS_H
#define UTILS_H 
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>


template<typename Generator>
void CreateImageGrid(Generator& generator, torch::Device device,int epoch)
{
        const std::string labels[10] = {"T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandel", " Shirt",
                                    "Sneaker", " Bag", "Ankle Boot"};
        cv::Mat dst(250, 500, CV_32F, cv::Scalar(255));
        const int noise_size = 100;
        const int rows = 4;
        const int cols = 10;
        const int image_size = 28;
        const int grid_size = 40;
        for (size_t r = 0; r < rows; r++)
        {
            torch::Tensor noise = torch::rand({cols,noise_size, 1,1 }, device);
            torch::Tensor targets = torch::arange(10, device);
            torch::Tensor one_hot = torch::zeros({cols, 10}, device);
            one_hot = one_hot.scatter_(1, targets.view({cols,1}), 1);
            torch::Tensor fake_images = generator->forward(noise,one_hot);
            
            for (size_t c = 0; c < cols; c++)
            {   
                if(r == 0)
                {
                    cv::putText(dst, labels[c], cv::Point(50 +c*grid_size, 40+ r*grid_size),.2 ,.3, cv::Scalar(0), 1);
                }
                torch::Tensor tensor_image = fake_images[c].squeeze();
                tensor_image = tensor_image.reshape({image_size,image_size});
                tensor_image *= 250;
                cv::Mat image(tensor_image.size(0), tensor_image.size(1), CV_32F, tensor_image.cpu().data_ptr<float>());
                image.copyTo(dst(cv::Rect(50 + c*grid_size, 50+ r*grid_size, image_size, image_size)));
            }
        }
        cv::resize(dst, dst, cv::Size(1000, 500));
        cv::imwrite("Visualisation/Epoch-" + std::to_string(epoch) + ".png", dst);
}

#endif // !UTILS_H

