#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, int n) {
    // 计算全局线程索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 越界判断，超出n直接返回，不写c
    if (idx >= n) {
        return;
    }
    c[idx] = a[idx] + b[idx];
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void scale_array(float *a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= scalar;
    }
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void elementwise_exp(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = expf(a[i]);
    }
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void row_max(const float *matrix, float *out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float *row_ptr = matrix + r * cols;
    float max_val = row_ptr[0];
    for (int c = 1; c < cols; c++) {
        if (row_ptr[c] > max_val) {
            max_val = row_ptr[c];
        }
    }
    out[r] = max_val;
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__device__ float dot_product(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}
