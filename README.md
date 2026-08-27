# Flash Attention 从零实现

只用 CUDA 从线程下标写到分块注意力：先做逐元素与归约，再拼出朴素 softmax 注意力，最后用在线 softmax 切块，避免物化整张分数矩阵。

## How to run

```bash
compile and run model.cu
```

## Steps

- [x] **1.** vector_add
- [x] **2.** scale_array
- [x] **3.** elementwise_exp
- [x] **4.** row_max
- [ ] **5.** row_sum
- [x] **6.** dot_product
- [ ] **7.** matmul
- [ ] **8.** transpose
- [ ] **9.** qk_scores
- [ ] **10.** softmax_rows
- [ ] **11.** pv_matmul
- [ ] **12.** naive_attention
- [ ] **13.** online_softmax_update
- [ ] **14.** plan_tiling
- [ ] **15.** flash_attention_kernel
- [ ] **16.** flash_attention_launcher
- [ ] **17.** flash_attention_causal_kernel
- [ ] **18.** attention_close
