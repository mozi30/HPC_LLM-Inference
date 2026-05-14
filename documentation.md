# LLM INFERENCE DOCUMENTATION
by Moritz Anton Zideck

## 1. Analytical Model Proposal
### TTFT Assumtion based on TPOT Assumtion

Since the available cache capacity is much smaller than the model weights, weight reuse from cache can be neglected
The test will also be done with a smaller model to see the impact of the cache.
To create the first token two phases have to be taken into account.
- Prefill Phase (Processing of input tokens)
- Generation Phase(Generating output)

In the prefill phase, the model weights must be read once from memory while the entire prompt is processed in parallel. Therefore, the prefill latency is the maximum of the memory-transfer time and the compute time for all prompt tokens:

T_prefill = max(
    Model_Size / Memory_Bandwidth,
    2 × Prompt_Length × Parameters / Effective_TFLOPs
)
The factor of 2 appears because each parameter participates in one multiply-accumulate (MAC) operation, which corresponds to two floating-point operations (one multiply and one add).

To generate the first output token, the model weights are read once again and a single forward pass is executed. The latency of this decode step is:

T_first_token = max(
    Model_Size / Memory_Bandwidth,
    2 × Parameters / Effective_TFLOPs
)

The Time to First Token (TTFT) is the sum of the prefill time and the first-token decode time:

TTFT = T_prefill + T_first_token

TTFT = max(
           Model_Size / Memory_Bandwidth,
           2 × Prompt_Length × Parameters / Effective_TFLOPs
       )
     + max(
           Model_Size / Memory_Bandwidth,
           2 × Parameters / Effective_TFLOPs
       )

### Models:
 - Meta-Llama-3.1-8B-Instruct-Q4_K_M
    - Model size: 4.5 GB
    - Paramters: 8 Billion
 - Gemma-2-2b-it-Q8_0
    - Model size: 2.6 GB
    - Paramters: 2.6 Billion
 - SmolLM2-1.7B-Instruct-Q4_K_M
    - Model size 1 GB
    - Paramters: 1.7 Billion

Deucalion ARM node:
- Measured TFLOPS: 1.9T
- Measured Memory Bandwidth: 450 GB/s

The inference engine llama.cpp with 48 threads was used for all tests.

### Calculated:

L... is the prompt length

Meta-Llama-3.1-8B-Instruct-Q4_K_M

Formula: TTFT=max(0.0100, 0.008421L)+0.0100
- L =   32 ->   280 ms
- L =  128 ->  1088 ms
- L =  512 ->  4321 ms
- L = 2048 -> 17257 ms
- L = 8192 -> 68986 ms


Gemma-2-2b-it-Q8_0
Formula: TTFT = max(0.00578, 0.002737L) + 0.00578
- L =   32 ->    93 ms
- L =  128 ->   356 ms
- L =  512 ->  1407 ms
- L = 2048 ->  5611 ms
- L = 8192 -> 22429 ms


SmolLM2-1.7B-Instruct-Q4_K_M
Formula: TTFT = max(0.00222, 0.001789L) + 0.00222
- L =  32  ->    59 ms
- L = 128  ->   231 ms
- L = 512  ->   918 ms
- L = 2048 ->  3665 ms
- L = 8192 -> 14657 ms

### Measured:

Meta-Llama-3.1-8B-Instruct-Q4_K_M
- L =   32 ->  9406 ms
- L =  128 -> 11760 ms
- L =  512 -> 16680 ms
- L = 2048 -> 75260 ms

Gemma-2-2b-it-Q8_0
- L =   32 ->  2873 ms
- L =  128 ->  3558 ms
- L =  512 ->  5153 ms
- L = 2048 -> 23890 ms

SmolLM2-1.7B-Instruct-Q4_K_M
- L =   32 ->  1328 ms
- L =  128 ->  1433 ms
- L =  512 ->  1775 ms
- L = 2048 ->  8654 ms

### Conclusion

The real implementation introduces significant overhead, which is most visible at low prompt lengths. In the analytical model, we use a measured performance of 1.9 TFLOPS, obtained from a C-based matrix multiplication benchmark using all cores. However, this value does not necessarily represent the effective TFLOPS achieved during LLM inference. For a more precise model, both the implementation overhead and the actual inference TFLOPS would need to be measured. Nevertheless, the model provides a useful general direction for understanding how TTFT behaves as prompt length, model size, and parameter count change.

## 2. LLM Inference Engine Analysis

Models as in 1.

Meta-Llama-3.1-8B-Instruct-Q4_K_M
TTFT(promt_length = 128) = 11760 ms
TPOT(mostly stable +- 1) =    80 ms
Throughput: Prefill = 30/s
Throughput: Generation = 12.5/s

Gemma-2-2b-it-Q8_0
TTFT(promt_length = 128) = 3558 ms
TPOT(stable until 512, breakdown at 2048) = 11.1 ms (at 2048: 17.2)
Throughput: Prefill = 290/s
Throughput: Generation = 99.71/s

SmolLM2-1.7B-Instruct-Q4_K_M
TTFT(promt_length = 128) = 1433 ms
TPOT(between) = 59ms - 68 ms
Throughput: Prefill = 100.88/s
Throughput: Generation = 17.18/s
