import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter as time
import sys
sys.path.append('/home/ys3860/')
from OpenCLmodule_2 import OpenCLModule

# Initialize OpenCL module
opencl_module = OpenCLModule()

# Vector sizes for testing
vector_sizes = [10**i for i in range(1, 9)]  
iterations = 50

# Accumulate kernel times in a list for each vector size
device_mem_times = []
buffer_mem_times = []
cpu_numpy_times = []

# Manually define whether 'b' is a vector or a scalar
for size in vector_sizes:
    a = np.random.randn(size).astype(np.float32)

    b = np.float32(np.random.randn())  # 'b' is a scalar
    is_b_a_vector = False  

    # Initialize accumulators for each method
    device_mem_time = 0
    buffer_mem_time = 0
    cpu_numpy_time = 0

    # Run multiple iterations for each size
    for _ in range(iterations):
        # OpenCL device memory method (pyopencl.array)
        _, total_time = opencl_module.device_add(a, b, is_b_a_vector=is_b_a_vector)
        device_mem_time += total_time

        # OpenCL buffer memory method
        _, total_time = opencl_module.buffer_add(a, b, is_b_a_vector=is_b_a_vector)
        buffer_mem_time += total_time

        # CPU NumPy addition (works for both scalar and vector addition)
        cpu_start = time()
        if is_b_a_vector:
            result = a + b
        else:
            result = a + np.float32(b)
        cpu_end = time()
        cpu_numpy_time += cpu_end - cpu_start

    # After all iterations, calculate the average times for each method
    avg_device_mem_time = device_mem_time / iterations
    avg_buffer_mem_time = buffer_mem_time / iterations
    avg_cpu_numpy_time = cpu_numpy_time / iterations

    # Print the times for the current vector size (average over iterations)
    print(f"\nVector Size: {size}")
    print(f"Device Mem (OpenCL array): {avg_device_mem_time:.6f} s")
    print(f"Buffer Mem (OpenCL buffer): {avg_buffer_mem_time:.6f} s")
    print(f"CPU NumPy: {avg_cpu_numpy_time:.6f} s")

    # Record average times for plotting
    device_mem_times.append(avg_device_mem_time)
    buffer_mem_times.append(avg_buffer_mem_time)
    cpu_numpy_times.append(avg_cpu_numpy_time)

# Plotting results
plt.figure(figsize=(12, 6))

# Plot for memory transfer
plt.subplot(1, 2, 1)
plt.plot(vector_sizes, device_mem_times, label='OpenCL Device Mem')
plt.plot(vector_sizes, buffer_mem_times, label='OpenCL Buffer Mem')
plt.plot(vector_sizes, cpu_numpy_times, label='CPU NumPy')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Vector Size')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time Including Memory Transfer')
plt.legend()

plt.tight_layout()
plt.show()
