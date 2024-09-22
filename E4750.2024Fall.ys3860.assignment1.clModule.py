import pyopencl as cl
import numpy as np
from time import perf_counter as time

class OpenCLModule:
    def __init__(self):
        """
        Initialize OpenCL context, queue, and program.
        """
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        # Define OpenCL kernels for vector addition
        kernel_code = """
        __kernel void Add_two_vectors(__global const float *a, __global const float *b, __global float *result, int n) {
            int idx = get_global_id(0);
            if (idx < n) {
                result[idx] = a[idx] + b[idx];
            }
        }

        __kernel void Add_to_each_element(__global const float *a, const float b, __global float *result, int n) {
            int idx = get_global_id(0);
            if (idx < n) {
                result[idx] = a[idx] + b;
            }
        }
        """
        self.program = cl.Program(self.context, kernel_code).build()

    def device_add(self, a, b, is_b_a_vector=True):
        """
        Perform vector addition using pyopencl.array (device memory).
       
        """
        mf = cl.mem_flags
        n = len(a)

        # Create buffers
        a_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, a.nbytes)

        if is_b_a_vector:
            b_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        # Time including memory transfer
        total_start = time()

        # Launch kernel
        if is_b_a_vector:
            event = self.program.Add_two_vectors(self.queue, (n,), None, a_buf, b_buf, result_buf, np.int32(n))
        else:
            event = self.program.Add_to_each_element(self.queue, (n,), None, a_buf, np.float32(b), result_buf, np.int32(n))

        # Wait for the kernel event to finish
        event.wait()  

        # Retrieve results from device
        result = np.empty_like(a)
        cl.enqueue_copy(self.queue, result, result_buf).wait()  

        total_end = time()
        total_time = total_end - total_start
        return result, total_time

    def buffer_add(self, a, b, is_b_a_vector=True):
        """
        Perform vector addition using pyopencl.Buffer.
       
        """
        mf = cl.mem_flags
        n = len(a)

        # Create buffers
        a_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
        result_buf = cl.Buffer(self.context, mf.WRITE_ONLY, a.nbytes)

        if is_b_a_vector:
            b_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

        # Time including memory transfer
        total_start = time()

        # Launch kernel
        if is_b_a_vector:
            event = self.program.Add_two_vectors(self.queue, (n,), None, a_buf, b_buf, result_buf, np.int32(n))
        else:
            event = self.program.Add_to_each_element(self.queue, (n,), None, a_buf, np.float32(b), result_buf, np.int32(n))

        # Wait for the kernel event to finish
        event.wait()  

        # Retrieve results from device
        result = np.empty_like(a)
        cl.enqueue_copy(self.queue, result, result_buf).wait()  

        total_end = time()
        total_time = total_end - total_start
        return result, total_time
