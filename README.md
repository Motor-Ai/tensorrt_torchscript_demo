# Deployment Tools 
In this repository, we have summarized several deployment pathways with small-scale examples. Deployment here, strictly refers to converting a Pytorch Model, into an executable (or a series of executables) that can be integrated in an already existing C++ based project, on a specific hardware. In practice, even though PyTorch based development is done in python, one should not simply deploy such models in python. Instead, these models should be converted to C++ executables, after adding further inference centric optimizations. 


## A. System Setup
Before we dive further into specifics, we must first get our system ready. 
1. First install required libraries using ``` requirements.txt```.


    ```
    pip3 install -r requirements.txt
    ```
2. Then, clone the repository locally :

    ```
    git clone https://github.com/sanket-perception/tensorrt_torchscript_demo.git
    ```



## B. Optimizing Classification Model in ONNX + TensorRT

There are several pathways one can choose from, in order to take a model from development to production. A clear winner amongst all existing methods is TensorRT, NVIDIA's flagship model optimization framework and inference engine generator. In simple words, TensorRT takes a Torch/TensorFlow model and converts it into an "engine" such that it makes most of the existing hardware resource. We will first look at a simple example of converting a classification model into TensorRT engine. Refer to the notebook inside [this notebook](/notebooks/Simple%20Classification%20Model%20-%20TensorRT%20%2B%20ONNX.ipynb).

## C. Optimizing Classification Model in TorchScript

TensorRT is an unparalleled solution for model deployment,however, for Pytorch or ONNX-based models it has incomplete support and suffers from poor portability. It is difficult to integrate TensorRT with custom torch operations, which is the case more often than not. There is a plugin system to add custom layers and postprocessing, but this low-level work is out of reach for groups without specialized deployment teams. TensorRT also doesn’t support cross-compilation so models must be optimized directly on the target hardware — not great for embedded platforms or highly diverse compute ecosystems. Thus, in conlusion, TensorRT is not the ultimate answer to the deployment puzzle. Another candidate for solving this problem is TorchScript. 

TorchScript is a way to create serializable and optimizable models from PyTorch code. Any TorchScript program can be saved from a Python process and loaded in a process where there is no Python dependency.

It is a tool to incrementally transition a model from a pure Python program to a TorchScript program that can be run independently from Python, such as in a standalone C++ program. This makes it possible to train models in PyTorch using familiar tools in Python and then export the model via TorchScript to a production environment where Python programs may be disadvantageous for performance and multi-threading reasons.

Let us look at the same model as we explored with TensorRT, using TorchScript to understand how this framework works. Please refer to [this notebook](/notebooks/Simple%20Classification%20Model%20%20TorchScript.ipynb) for further details.


## D. Loading TorchScript in C++ 
The main objective of converting a torch model to Torchscript is to allow for portability from python to C++ environment. To that end, let us look at how our  torchscript engine, we generated in previous step. We will deserialize the torch script module and give dummy inputs to the network in C++ to perform inference.
The C++ file [inference.cpp](cpp_files/inference.cpp) contains the script for loading Torchscript in C++ and performing inference on dummy inputs.
To build this project, run the following commands in the terminal.

```
cd cpp_files/
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
cmake --build . --config Release

```

where `/path/to/libtorch` should be the full path to the unzipped LibTorch distribution. 

If all goes well, there will be a `inference` executable inside build directory. Finally to run inference, execute the following command in terminal from build directory :

```
./inference <path_to_repo>/torchscript_engines/simple_classifier.pt
```

This will print the first 5 values of the predicted 1000 probabilities by the network.

In conclusion, we deserialized a Torchscript module in c++originally created with Python and passed dummy inputs to perform inference with the network.

## E. Using Custom Operations with TorchScript
Torchscript allows for registering custom operations written in CUDA/C++ as torch extensions. In other words, the way we use native torch functions like add() with `torch.add()`, after this process, one can use custom_op with `torch.my_op.custom_op()` given `custom_op` is registered witing `my_op' namespace. Let us look at an example to understand how this is achieved. 

We leverage our custom OP called `bevpool_forward()` written in C++/CUDA.
In the C++ files we add the following lines of code to register it with TorchScript.


```

TORCH_LIBRARY(my_ops, m) {
  m.def("bev_pool_forward", &bev_pool_forward);
}
```

where `my_ops` is the namespace, `bev_pool_forward` is the name of the operation. Then we enter the following command to build this extenstion from within the `bev_pool` directory.

```
python setup.py install 
```

This will create a `build` directory inside `bev_pool` directory and store the corresponding library within it's `lib` directory.

The [notebook](notebooks/Custom_Op.ipynb) contains the script to first demonstrate how to load this library and use this custom ops with torch. 
Furthermore, it also shows how to build a torcscript, this time with our CustomOp. 





