import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import numpy as np

class CudaModule:
    def __init__(self):
        """
        Compile CUDA kernels and initialize CUDA device.
        """
        self.mod = self.getSourceModule()

    def getSourceModule(self):
        """
        Compile the kernel functions for vector addition operations.
        """
        kernel_code = """
        __global__ void Add_two_vectors(float *a, float *b, float *result, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                result[idx] = a[idx] + b[idx];
            }
        }

        __global__ void Add_to_each_element(float *a, float b, float *result, int n) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx < n) {
                result[idx] = a[idx] + b;
            }
        }
        """
        return SourceModule(kernel_code)

    def add_device_mem_gpu(self, a, b, is_b_a_vector=True):
        """
        Perform vector addition using explicit device memory allocation.
        """
        n = len(a)
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        # Allocate device memory for a, b, and result
        a_gpu = cuda.mem_alloc(a.nbytes)
        result_gpu = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(a_gpu, a)

        if is_b_a_vector:
            b_gpu = cuda.mem_alloc(b.nbytes)
            cuda.memcpy_htod(b_gpu, b)
            func = self.mod.get_function("Add_two_vectors")
        else:
            func = self.mod.get_function("Add_to_each_element")

        # Create CUDA Events for timing
        start_event = cuda.Event()
        end_event = cuda.Event()

        # Record the start event
        start_event.record()

        # Execute the kernel
        if is_b_a_vector:
            func(a_gpu, b_gpu, result_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))
        else:
            func(a_gpu, np.float32(b), result_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

        # Record the end event and synchronize
        end_event.record()
        end_event.synchronize()

        # Compute the total execution time using CUDA events
        total_time = start_event.time_till(end_event) * 1e-3  # Convert to seconds

        # Copy result from device to host
        result = np.empty_like(a)
        cuda.memcpy_dtoh(result, result_gpu)

        return result, total_time

    def add_host_mem_gpu(self, a, b, is_b_a_vector=True):
        """
        Perform vector addition without explicit device memory allocation using host memory.
        """
        n = len(a)
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        result = np.empty_like(a)

        if is_b_a_vector:
            func = self.mod.get_function("Add_two_vectors")
        else:
            func = self.mod.get_function("Add_to_each_element")

        # Create CUDA Events for timing
        start_event = cuda.Event()
        end_event = cuda.Event()

        # Record the start event
        start_event.record()

        # Execute the kernel
        if is_b_a_vector:
            func(cuda.InOut(a), cuda.InOut(b), cuda.Out(result), np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))
        else:
            func(cuda.InOut(a), np.float32(b), cuda.Out(result), np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

        # Record the end event and synchronize
        end_event.record()
        end_event.synchronize()

        # Compute the total execution time using CUDA events
        total_time = start_event.time_till(end_event) * 1e-3  # Convert to seconds

        return result, total_time

    def add_gpuarray_kernel(self, a, b, is_b_a_vector=True):
        """
        Perform vector addition using gpuarray and kernel.
        """
        n = len(a)
        block_size = 256
        grid_size = (n + block_size - 1) // block_size

        a_gpu = gpuarray.to_gpu(a)
        result_gpu = gpuarray.empty_like(a_gpu)

        # Create CUDA Events for timing
        start_event = cuda.Event()
        end_event = cuda.Event()

        # Record the start event
        start_event.record()

        # Execute the kernel
        if is_b_a_vector:
            func = self.mod.get_function("Add_two_vectors")
            b_gpu = gpuarray.to_gpu(b)
            func(a_gpu, b_gpu, result_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))
        else:
            func = self.mod.get_function("Add_to_each_element")
            func(a_gpu, np.float32(b), result_gpu, np.int32(n), block=(block_size, 1, 1), grid=(grid_size, 1))

        # Record the end event and synchronize
        end_event.record()
        end_event.synchronize()

        # Compute the total execution time using CUDA events
        total_time = start_event.time_till(end_event) * 1e-3  # Convert to seconds

        result = result_gpu.get()
        return result, total_time

    def add_gpuarray_no_kernel(self, a, b, is_b_a_vector=True):
        """
        Perform vector addition using gpuarray without explicit kernel.
        """
        a_gpu = gpuarray.to_gpu(a)
        if is_b_a_vector:
            b_gpu = gpuarray.to_gpu(b)
            result_gpu = a_gpu + b_gpu
        else:
            result_gpu = a_gpu + np.float32(b)

        # Create CUDA Events for timing
        start_event = cuda.Event()
        end_event = cuda.Event()

        # Record the start event
        start_event.record()

        # Perform the addition
        result = result_gpu.get()

        # Record the end event and synchronize
        end_event.record()
        end_event.synchronize()

        # Compute the total execution time using CUDA events
        total_time = start_event.time_till(end_event) * 1e-3  # Convert to seconds

        return result, total_time
