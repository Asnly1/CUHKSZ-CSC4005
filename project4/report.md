# Project 4: Parallel Programming with FlashAttention Report

## 1\. How to Compile and Execute

The project is compiled and executed according to the `README.md`.

### Execution

All jobs are submitted to the Slurm cluster using the provided script.

```bash
cd /path/to/project4/
sbatch ./scripts/sbatch_Part1.sh
sbatch ./scripts/sbatch_Part2.sh
sbatch ./scripts/sbatch_Part3.sh
```

## 2\. Algorithm Design and Implementation

### Task 1.1: Softmax with CUDA

Similar to cpu_softmax below, but use shared memory to accelerate

1. Let each block take charge of one row and let each thread take charge of several elements in one row
2. Calculate local_max for each thread
3. Calculate the gloabl maximum of one row
4. Calculate unnormalized result for each element and local_sum for each thread
5. Calculate the gloabl sum of one row
6. Normailize elements and dropout

### Task 1.2: Softmax with Triton

1. Let each program get their own row
2. Calculate input address
3. If HAS_MASK, then load mask and take intersection with boundary mask
4. Load input value and do normal safe softmax calculations
5. If HAS_DROPOUT, generate dropout mask and mask it with softmax output
6. Calculate output address and store output

### Task 2: FlashAttention with Triton

Similar to what algorithm states, the main difference is that in algorithm, the outer loop is in K, V and the inner loop is in Q. My implementation lets each program take charge of BLOCK_M rows and loop on K, V

### Task 3: Sparse FlashAttention

Similar to Task 2 but add a single improvement: if the elements of K, V fall into the mask part, then return immediately.

### Extra Credit: FlashAttention Ver.2 in Triton

Similar to Task 2 but add a single improvement: no longer maintain complete output in the loop, but only maintain the numerator in the loop and do the division finnally, thus reduces non-matmul operations

## 3\. Experiment Results

## 4\. Performance Analysis
