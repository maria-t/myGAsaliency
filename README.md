# myGAsaliency
This project is about optimizing the weights of a computational framework that detects salient objects in camera frames.

## My framework
- The algorithm used for optimization is a Genetic Algorithm.
- The weights [w_1, w_2, w_3] are represented in binary strings.
- The objective function used is the Euclidean Distance between the color images and their equivalent groundtruth in the SED1 and SED2 databases.
- The parents are selected using Tournament Selection.
- A 2-point Crossover is implemented.
- Flip-bit Mutation is implemented.

## Implementation Details
The framework is implemented in C++. The following libraries are used:
- `C++ Standard` 
- `xtensor`
- `OpenCV` 

You can compile the code with:

``g++ -I "/home/usr/ProjectDirectory/xtl/include" -o myGAsaliency my_GA_saliency_framework.cpp `pkg-config opencv --cflags --libs` `` 

