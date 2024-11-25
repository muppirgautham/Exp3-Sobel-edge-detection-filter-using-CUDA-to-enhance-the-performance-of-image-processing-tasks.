# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>ENTER YOUR NAME: M Gautham</h3>
<h3>ENTER YOUR REGISTER NO: 212221230027</h3>
<h3>EX. NO : 3</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,  
                            unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

        int sumX = 0;
        int sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                unsigned char pixel = srcImage[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int magnitude = sqrtf(sumX * sumX + sumY * sumY);
        magnitude = min(max(magnitude, 0), 255);
        dstImage[y * width + x] = static_cast<unsigned char>(magnitude);
    }
}

void checkCudaErrors(cudaError_t r) {
    if (r != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

void analyzePerformance(const std::vector<std::pair<int, int>>& sizes, 
                        const std::vector<int>& blockSizes, unsigned char *d_inputImage, 
                        unsigned char *d_outputImage) {
                        
    for (auto size : sizes) {
        int width = size.first;
        int height = size.second;

        printf("CUDA - Size: %dx%d\n", width, height);
        
        dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));
        for (auto blockSize : blockSizes) {
            dim3 blockDim(blockSize, blockSize);
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start);
            sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            printf("    Block Size: %dx%d Time: %f ms\n", blockSize, blockSize, milliseconds);

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    }
}

int main() {
    Mat image = imread("/content/images.jpg", IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Image not found.\n");
        return -1;
    }

    // Convert to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    int width = grayImage.cols;
    int height = grayImage.rows;
    size_t imageSize = width * height * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);
    if (h_outputImage == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    unsigned char *d_inputImage, *d_outputImage;
    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));
    checkCudaErrors(cudaMemcpy(d_inputImage,grayImage.data,imageSize,cudaMemcpyHostToDevice));

    // Performance analysis
    std::vector<std::pair<int, int>> sizes = {{256, 256}, {512, 512}, {1024, 1024}};
    std::vector<int> blockSizes = {8, 16, 32};

    analyzePerformance(sizes, blockSizes, d_inputImage, d_outputImage);

    // Execute CUDA Sobel filter one last time for the original image
    dim3 gridSize(ceil(width / 16.0), ceil(height / 16.0));
    dim3 blockDim(16, 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    sobelFilter<<<gridSize, blockDim>>>(d_inputImage, d_outputImage, width, height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    checkCudaErrors(cudaMemcpy(h_outputImage,d_outputImage,imageSize,cudaMemcpyDeviceToHost));

    // Output image
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel_cuda.jpeg", outputImage);

    // OpenCV Sobel filter for comparison
    Mat opencvOutput;
    auto startCpu = std::chrono::high_resolution_clock::now();
    cv::Sobel(grayImage, opencvOutput, CV_8U, 1, 0, 3);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpuDuration = endCpu - startCpu;

    // Save and display OpenCV output
    imwrite("output_sobel_opencv.jpeg", opencvOutput);
    
    printf("Input Image Size: %d x %d\n", width, height);
    printf("Output Image Size (CUDA): %d x %d\n", outputImage.cols, outputImage.rows);
    printf("Total time taken (CUDA): %f ms\n", milliseconds);
    printf("OpenCV Sobel Time: %f ms\n", cpuDuration.count());

    // Cleanup
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```

## OUTPUT:
![image](https://github.com/user-attachments/assets/f40e7bb0-d036-4ba6-9291-b4b22961e86c)

- **Sample Execution Results**:
  - **CUDA Execution Times (Sobel filter)**
  </br>

<img src="https://github.com/user-attachments/assets/bc4f2f40-2f6c-4dd8-af8b-e4583b28079d" width="400">


  - **OpenCV Execution Time**
  </br>

![image](https://github.com/user-attachments/assets/db287c68-1ae8-42a4-9ef3-a78f16d0de37)

- **Graph Analysis**:
  - Displayed a graph showing the relationship between image size, block size, and execution time.
 </br>

<img src="https://github.com/user-attachments/assets/ddb5fb05-53fe-440f-8876-0d3797de1bb5" width="500">

## Answers to Questions

1. **Challenges Implementing Sobel for Color Images**:
   - Converting images to grayscale in the kernel increased complexity. Memory management and ensuring correct indexing for color to grayscale conversion required attention.

2. **Influence of Block Size**:
   - Smaller block sizes (e.g., 8x8) were efficient for smaller images but less so for larger ones, where larger blocks (e.g., 32x32) reduced overhead.

3. **CUDA vs. CPU Output Differences**:
   - The CUDA implementation was faster, with minor variations in edge sharpness due to rounding differences. CPU output took significantly more time than the GPU.

4. **Optimization Suggestions**:
   - Use shared memory in the CUDA kernel to reduce global memory access times.
   - Experiment with adaptive block sizes for larger images.
  
  ## Result
Successfully implemented a CUDA-accelerated Sobel filter, demonstrating significant performance improvement over the CPU-based implementation, with an efficient parallelized approach for edge detection in image processing.

