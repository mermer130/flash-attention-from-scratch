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

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void transpose(const float *in, float *out, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int i = idx / cols;
    int j = idx % cols;
    out[j * rows + i] = in[i * cols + j];
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void qk_scores(const float *Q, const float *K, float *S, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * seq_len) return;
    int i = idx / seq_len;
    int j = idx % seq_len;
    float s = 0.f;
    for (int t = 0; t < head_dim; ++t) s += Q[i * head_dim + t] * K[j * head_dim + t];
    S[i * seq_len + j] = s * rsqrtf((float)head_dim);
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void softmax_rows(float *M, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    if (threadIdx.x != 0) return;
    float *row = M + r * cols;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, row[c]);
    float s = 0.f;
    for (int c = 0; c < cols; ++c) {
        row[c] = expf(row[c] - m);
        s += row[c];
    }
    for (int c = 0; c < cols; ++c) row[c] /= s;
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void pv_matmul(const float *P, const float *V, float *O, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * head_dim) return;
    int i = idx / head_dim;
    int c = idx % head_dim;
    float s = 0.f;
    for (int j = 0; j < seq_len; ++j) s += P[i * seq_len + j] * V[j * head_dim + c];
    O[i * head_dim + c] = s;
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void naive_attention(const float *Q, const float *K, const float *V, float *O, int seq_len, int head_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;
    float scale = rsqrtf((float)head_dim);
    float s[32];
    float m = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
        float acc = 0.f;
        for (int t = 0; t < head_dim; ++t) acc += Q[i * head_dim + t] * K[j * head_dim + t];
        s[j] = acc * scale;
        m = fmaxf(m, s[j]);
    }
    float sum = 0.f;
    for (int j = 0; j < seq_len; ++j) {
        s[j] = expf(s[j] - m);
        sum += s[j];
    }
    for (int t = 0; t < head_dim; ++t) {
        float o = 0.f;
        for (int j = 0; j < seq_len; ++j) o += (s[j] / sum) * V[j * head_dim + t];
        O[i * head_dim + t] = o;
    }
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

void online_softmax_update(float &m_i, float &l_i, float *O_i,
    const float *S_ij, const float *V_j, int Br, int Bc, int d) {
    (void)Br;
    float m_tilde = -INFINITY;
    for (int j = 0; j < Bc; ++j) m_tilde = fmaxf(m_tilde, S_ij[j]);
    float l_tilde = 0.f;
    float P[32];
    for (int j = 0; j < Bc; ++j) {
        P[j] = expf(S_ij[j] - m_tilde);
        l_tilde += P[j];
    }
    float m_new = fmaxf(m_i, m_tilde);
    float l_new = expf(m_i - m_new) * l_i + expf(m_tilde - m_new) * l_tilde;
    float scale_old = expf(m_i - m_new) * l_i;
    float scale_p = expf(m_tilde - m_new);
    for (int t = 0; t < d; ++t) {
        float acc = scale_old * O_i[t];
        for (int j = 0; j < Bc; ++j) acc += scale_p * P[j] * V_j[j * d + t];
        O_i[t] = acc / l_new;
    }
    m_i = m_new;
    l_i = l_new;
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

void plan_tiling(int N, int d, int M, int *Br, int *Bc, int *Tr, int *Tc) {
    int t = 4 * d;
    int bc = (M + t - 1) / t;
    if (bc < 1) bc = 1;
    int br = bc < d ? bc : d;
    if (br < 1) br = 1;
    *Bc = bc;
    *Br = br;
    *Tr = (N + br - 1) / br;
    *Tc = (N + bc - 1) / bc;
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void flash_attention_kernel(
    const float *Q, const float *K, const float *V, float *O,
    int seq_len, int head_dim, int Br, int Bc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;
    (void)Br;
    if (Bc < 1) Bc = 1;
    float scale = rsqrtf((float)head_dim);
    float m = -INFINITY;
    float l = 0.f;
    float acc[16];
    for (int t = 0; t < head_dim; ++t) acc[t] = 0.f;
    for (int j0 = 0; j0 < seq_len; j0 += Bc) {
        int j1 = j0 + Bc;
        if (j1 > seq_len) j1 = seq_len;
        float m_blk = -INFINITY;
        for (int j = j0; j < j1; ++j) {
            float s = 0.f;
            for (int t = 0; t < head_dim; ++t) s += Q[i * head_dim + t] * K[j * head_dim + t];
            s *= scale;
            m_blk = fmaxf(m_blk, s);
        }
        float m_new = fmaxf(m, m_blk);
        float alpha = expf(m - m_new);
        l *= alpha;
        for (int t = 0; t < head_dim; ++t) acc[t] *= alpha;
        for (int j = j0; j < j1; ++j) {
            float s = 0.f;
            for (int t = 0; t < head_dim; ++t) s += Q[i * head_dim + t] * K[j * head_dim + t];
            s *= scale;
            float p = expf(s - m_new);
            l += p;
            for (int t = 0; t < head_dim; ++t) acc[t] += p * V[j * head_dim + t];
        }
        m = m_new;
    }
    float inv = l > 0.f ? 1.f / l : 0.f;
    for (int t = 0; t < head_dim; ++t) O[i * head_dim + t] = acc[t] * inv;
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

void flash_attention_launcher(const float *Q, const float *K, const float *V, float *O,
    int batch, int heads, int seq_len, int head_dim) {
    int stride = seq_len * head_dim;
    size_t bytes = (size_t)batch * heads * stride * sizeof(float);
    float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    cudaMalloc(&dQ, bytes);
    cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes);
    cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, bytes, cudaMemcpyHostToDevice);
    int Br = seq_len;
    int Bc = seq_len;
    int threads = 128;
    int blocks = (seq_len + threads - 1) / threads;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            int off = (b * heads + h) * stride;
            flash_attention_kernel<<<blocks, threads>>>(dQ + off, dK + off, dV + off, dO + off,
                seq_len, head_dim, Br, Bc);
        }
    }
    cudaDeviceSynchronize();
    cudaMemcpy(O, dO, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

__global__ void flash_attention_causal_kernel(
    const float *Q, const float *K, const float *V, float *O,
    int seq_len, int head_dim, int Br, int Bc, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;
    (void)Br;
    if (Bc < 1) Bc = 1;
    float m = -INFINITY;
    float l = 0.f;
    float acc[16];
    for (int t = 0; t < head_dim; ++t) acc[t] = 0.f;
    for (int j0 = 0; j0 < seq_len; j0 += Bc) {
        int j1 = j0 + Bc;
        if (j1 > seq_len) j1 = seq_len;
        float m_blk = -INFINITY;
        int any = 0;
        for (int j = j0; j < j1; ++j) {
            if (j > i) continue;
            float s = 0.f;
            for (int t = 0; t < head_dim; ++t) s += Q[i * head_dim + t] * K[j * head_dim + t];
            s *= scale;
            m_blk = fmaxf(m_blk, s);
            any = 1;
        }
        if (!any) continue;
        float m_new = fmaxf(m, m_blk);
        float alpha = expf(m - m_new);
        l *= alpha;
        for (int t = 0; t < head_dim; ++t) acc[t] *= alpha;
        for (int j = j0; j < j1; ++j) {
            if (j > i) continue;
            float s = 0.f;
            for (int t = 0; t < head_dim; ++t) s += Q[i * head_dim + t] * K[j * head_dim + t];
            s *= scale;
            float p = expf(s - m_new);
            l += p;
            for (int t = 0; t < head_dim; ++t) acc[t] += p * V[j * head_dim + t];
        }
        m = m_new;
    }
    float inv = l > 0.f ? 1.f / l : 0.f;
    for (int t = 0; t < head_dim; ++t) O[i * head_dim + t] = acc[t] * inv;
}

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

bool attention_close(const float *a, const float *b, int n, float tol) {
    for (int i = 0; i < n; ++i) {
        if (fabsf(a[i] - b[i]) > tol) return false;
    }
    return true;
}
