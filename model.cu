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
