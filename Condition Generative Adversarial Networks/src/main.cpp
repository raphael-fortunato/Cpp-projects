#include <iostream>
#include <limits>
#include <torch/torch.h>
#include <experimental/filesystem>
#include "../include/conditional_gan.h"
#include "../include/conditional_dcgan.h"
#include "../include/utils.h"

namespace fs = std::experimental::filesystem;
using namespace torch;


template<typename Generator,typename G_optimizer, typename Discriminator, typename D_optimizer >
void Train(Generator& generator, G_optimizer& g_optimizer, 
        Discriminator& discriminator, D_optimizer& d_optimizer, Device device, 
        const int n_epoch, const int batch_size, bool load)
        {

        auto dataset = torch::data::datasets::MNIST("./fashion-mnist/")
            .map(torch::data::transforms::Normalize<>(0.5, 0.5))
            .map(torch::data::transforms::Stack<>());

        const int dataset_size = dataset.size().value();

        const int batch_per_epoch =
            std::ceil(dataset_size / (double)batch_size);   

        auto dataloader = torch::data::make_data_loader(std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));

        const int noise_size =100;

        //loading the models en the optimizers    
        if(load)
        {
            try
            {
                torch::load(generator, generator->name);
                torch::load(discriminator, discriminator->name);
                torch::load(d_optimizer, "SavedModels/d_optimizer.pt");
                torch::load(g_optimizer, "SavedModels/g_optimizer.pt");
            }
            catch(const std::exception& e)
            {
                std::cerr << "Unable to locate saved files!" << std::endl;
            }
        }
        
     
        float lowest_loss = std::numeric_limits<float>::max();

        for (size_t epoch = 0; epoch < n_epoch; epoch++)
        {
            float discriminator_loss = 0;
            float generator_loss = 0;
            int likelihood = 0;
            for (torch::data::Example<>& batch : *dataloader)
            {
                //train discriminator with images
                discriminator->zero_grad();
                torch::Tensor real_images = batch.data.to(device);

                torch::Tensor real_targets = batch.target.to(device);
                torch::Tensor fake_labels = torch::empty(batch.data.size(0), device).uniform_(.8,1.);
                torch::Tensor valid_labels = torch::rand(batch.data.size(0), device) * .2;

                //noise for the generator to create images from
                torch::Tensor noise = torch::rand({batch.data.size(0),noise_size, 1,1 }, device);

                ///////////////////////////////////////
                //training discriminator on real images
                ///////////////////////////////////////
                torch::Tensor one_hot = torch::zeros({batch.data.size(0), 10}).to(device);
                one_hot = one_hot.scatter_(1, real_targets.view({batch.data.size(0),1}), 1);
                torch::Tensor real_output = discriminator->forward(real_images, one_hot);
                torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, valid_labels);


                ///////////////////////////////////////
                //training discriminator on fake images
                ///////////////////////////////////////
                torch::Tensor fake_output = discriminator->forward(generator->forward(noise, one_hot).detach(), one_hot);
                torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);


                torch::Tensor d_loss =  d_loss_fake + d_loss_real;
                d_loss.backward();
                d_optimizer.step();



                //////////////////////
                //training generator
                //////////////////////
                generator->zero_grad();
                valid_labels = torch::zeros(batch.data.size(0), device);
                fake_output = discriminator->forward(generator->forward(noise, one_hot), one_hot);
                torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, valid_labels);
                g_loss.backward();
                g_optimizer.step();
                
                likelihood += torch::round(fake_output).sum().item<float>();
                discriminator_loss = d_loss.item<float>();
                generator_loss = g_loss.item<float>();
            } 
            std::cout << "Epoch: "<< epoch << ", generator loss: "<< generator_loss/batch_per_epoch << ", discriminator loss: "<< discriminator_loss/batch_per_epoch  <<
            ", Likelihood: "<< 1-((float) likelihood / dataset_size )<< std::endl;
            if(epoch % 5 == 0)
            {
                CreateImageGrid(generator, device, epoch);
            }
            if(epoch % 50 == 0)
            {
                torch::save(discriminator, discriminator->name);
                torch::save(generator, generator->name);
                torch::save(d_optimizer, "SavedModels/d_optimizer.pt");
                torch::save(g_optimizer, "SavedModels/g_optimizer.pt");
                std::cout << "saving" << std::endl;
                lowest_loss = generator_loss;
            }
        }
        std::cout <<"Finished Training" << std::endl;
}



int main()
{
    //create the necessary directories
    if (!fs::is_directory("SavedModels") || !fs::exists("SavedModels")) 
    { 
        fs::create_directory("SavedModels"); // create src folder
    }   
    if (!fs::is_directory("Visualisation") || !fs::exists("Visualisation")) 
    { 
        fs::create_directory("Visualisation"); // create src folder
    }
    torch::manual_seed(1);
    const int batch_size = 256;
    const int n_epoch = 500;

    bool cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

    dcgan::Discriminator discriminator;
    discriminator->to(device);
    dcgan::Generator generator;
    generator->to(device);

    torch::optim::Adam generator_optimizer(
        generator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
    torch::optim::Adam discriminator_optimizer(
        discriminator->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

    Train(generator,  generator_optimizer, discriminator, discriminator_optimizer, device, n_epoch, batch_size, true);
}
