import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/ys3860/')
from CudaModule_2 import CudaModule
from time import perf_counter as time

# Initialize CUDA module
cuda_module = CudaModule()

# Vector sizes for testing
vector_sizes = [10**i for i in range(1, 9)]
iterations = 50

# Initialize lists to store execution times
device_mem_times = []
host_mem_times = []
gpuarray_kernel_times = []
gpuarray_no_kernel_times = []
cpu_numpy_times = []

for size in vector_sizes:
    a = np.random.randn(size).astype(np.float32)

    b = np.random.randn(size).astype(np.float32)  # 'b' is a vector
    is_b_a_vector = True  # Set b as vector

    # b = np.float32(np.random.randn())  # 'b' is a scalar
    # is_b_a_vector = False  # Set b as scalar

    # Initialize accumulators for each method
    device_mem_time = 0
    host_mem_time = 0
    gpuarray_kernel_time = 0
    gpuarray_no_kernel_time = 0
    cpu_numpy_time = 0

    for _ in range(iterations):
        # PyCUDA device memory method
        total_time = cuda_module.add_device_mem_gpu(a, b, is_b_a_vector=is_b_a_vector)[1]
        device_mem_time += total_time

        # PyCUDA host memory method
        total_time = cuda_module.add_host_mem_gpu(a, b, is_b_a_vector=is_b_a_vector)[1]
        host_mem_time += total_time

        # PyCUDA gpuarray + kernel
        total_time = cuda_module.add_gpuarray_kernel(a, b, is_b_a_vector=is_b_a_vector)[1]
        gpuarray_kernel_time += total_time

        # PyCUDA gpuarray without kernel
        total_time = cuda_module.add_gpuarray_no_kernel(a, b, is_b_a_vector=is_b_a_vector)[1]
        gpuarray_no_kernel_time += total_time

        # CPU NumPy addition
        cpu_start = time()
        result = a + b
        cpu_end = time()
        cpu_numpy_time += cpu_end - cpu_start

    # Print the times for the current vector size
    print(f"\nVector Size: {size}")
    print(f"Device Mem (incl transfer): {device_mem_time / iterations:.6f} s")
    print(f"Host Mem (incl transfer): {host_mem_time / iterations:.6f} s")
    print(f"GPUArray + Kernel (incl transfer): {gpuarray_kernel_time / iterations:.6f} s")
    print(f"GPUArray no Kernel (incl transfer): {gpuarray_no_kernel_time / iterations:.6f} s")
    print(f"CPU NumPy: {cpu_numpy_time / iterations:.6f} s")

    # Record average times
    device_mem_times.append(device_mem_time / iterations)
    host_mem_times.append(host_mem_time / iterations)
    gpuarray_kernel_times.append(gpuarray_kernel_time / iterations)
    gpuarray_no_kernel_times.append(gpuarray_no_kernel_time / iterations)
    cpu_numpy_times.append(cpu_numpy_time / iterations)

# Plotting results
plt.figure(figsize=(12, 6))

# Plot for including memory transfer
plt.subplot(1, 2, 1)
plt.plot(vector_sizes, device_mem_times, label='Device Mem')
plt.plot(vector_sizes, host_mem_times, label='Host Mem')
plt.plot(vector_sizes, gpuarray_kernel_times, label='GPUArray + Kernel')
plt.plot(vector_sizes, gpuarray_no_kernel_times, label='GPUArray no Kernel')
plt.plot(vector_sizes, cpu_numpy_times, label='CPU NumPy')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Vector Size')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Including Memory Transfer')
plt.legend()

# Plot for excluding memory transfer
plt.subplot(1, 2, 2)
plt.plot(vector_sizes, device_mem_times, label='Device Mem')
plt.plot(vector_sizes, gpuarray_kernel_times, label='GPUArray + Kernel')
plt.plot(vector_sizes, gpuarray_no_kernel_times, label='GPUArray no Kernel')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Vector Size')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Excluding Memory Transfer')
plt.legend()

plt.tight_layout()
plt.show()
