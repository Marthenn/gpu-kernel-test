#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

void printGPUDashboard() {
  int deviceCount;
  checkCudaError(cudaGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cout << "No CUDA devices found." << std::endl;
    return;
  }

  int deviceId = 0;
  checkCudaError(cudaSetDevice(deviceId));

  cudaDeviceProp deviceProp;
  checkCudaError(cudaGetDeviceProperties(&deviceProp, deviceId));

  std::cout << "--- GPU Monitoring Dashboard ---" << std::endl;
  std::cout << "Device ID: " << deviceId << std::endl;
  std::cout << "Device Name: " << deviceProp.name << std::endl;

  double totalMemoryGB = deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
  std::cout << "Total Memory: " << totalMemoryGB << " GB" << std::endl;

  std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
  std::cout << "Clock Rate: " << deviceProp.clockRate / 1000 << " MHz" << std::endl;
  std::cout << "Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;

  std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
  std::cout << "Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;

  std::cout << "\n--- Dynamic Usage (Updated every 2s) ---" << std::endl;
  std::cout << "Press Ctrl+C to stop monitoring." << std::endl;

  while (true) {
    size_t freeMem, totalMem;
    checkCudaError(cudaMemGetInfo(&freeMem, &totalMem));

    double freeMemoryGB = static_cast<double>(freeMem) / (1024.0 * 1024.0 * 1024.0);
    double usedMemoryGB = static_cast<double>(totalMem - freeMem) / (1024.0 * 1024.0 * 1024.0);

    double usagePercentage = (usedMemoryGB / totalMemoryGB) * 100.0;

    int barWidth = 25;
    int pos = static_cast<int>(barWidth * usagePercentage / 100.0);

    std::string bar = "[";
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos) bar += "=";
      else bar += " ";
    }
    bar += "]";

    std::cout << "\r" << "Memory Usage: " << bar << " "
              << std::fixed << std::setprecision(2) << usagePercentage << "% ("
              << usedMemoryGB << " GB / " << totalMemoryGB << " GB)"
              << " Free Memory: " << freeMemoryGB << " GB" << std::flush;

    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
}

int main() {
  try {
    printGPUDashboard();
  } catch (const std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
