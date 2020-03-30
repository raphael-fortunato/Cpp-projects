# Conditional-Gan in C++

Two implementations of a Conditonal Generative Adverserial Network, 
one with convolution and one with fully connected layers.

# Dependencies 
- pytorch C++ frontend
- opencv

To build the code, run the following commands from your terminal:

```shell
$ cd dcgan
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
$ make
```

where `/path/to/libtorch` should be the path to the unzipped *LibTorch*

# Results
  Conditional Gan after 300 epochs:
  
![alt text](https://github.com/raphael-fortunato/Cpp-projects/blob/master/Condition%20Generative%20Adversarial%20Networks/Visualisation/Conditional_gan-epoch300.png)


