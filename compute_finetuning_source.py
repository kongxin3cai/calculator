from huggingface_hub.constants import default_home


# linear layer in which [S, M, K] -> [S, M, N] with LoRA matrix A, B whose ranks are r
def Linear_with_LoRA(S, M, K, N, r):
    # A * B = C, dA = dC * B^t, dB = A^t * dC
    # x * W + x * A * B = C -> x * W + D * B = C,
    # dD = dC * B^t, dA = x^t * dD ->
    # dA = x^t * dC * B^t
    # dB = (x * A)^t * dC
    # dx = dC * (W + A * B)^t

    forward_computation = 2 * S * M * K * N + 2 * S * M * K * r + 2 * S * M * r * N + S * M * N
    back_computation = S * M * (2 * N * K + 6 * N * r + 6 * r * K + K) + K * r + r * N
    # print(forward_compute_count, back_compute_count)
    return forward_computation + back_computation

def Score(B, T, H, d_h, d_hR):
    #forward
    temp = B * H * (T * T * (2 * d_h + 2 * d_hR - 1) + T * (3 * T - 1) + 2 * T * T * d_h - T * d_h)
    #backward
    temp += B * H * (4 * T * T * (d_h + d_hR) - 2 * T * (d_h + d_hR) + 4 * d_h * T * T - T * T - T * d_h)
    return temp

def deepseek_r1_computation(B, T, L, c_q, c_k, R, H, D_h, r = 16):
    t  = Linear_with_LoRA(B, T, L, c_q, r)
    t += Linear_with_LoRA(B, T, L, c_k, r)
    t += Linear_with_LoRA(B, T, L, R,   r)
    t += Linear_with_LoRA(B, T, c_q, H * D_h, r)
    t += Linear_with_LoRA(B, T, c_q, H * R  , r)
    t += Linear_with_LoRA(B, T, c_k, H * D_h, r)
    t += Linear_with_LoRA(B, T, c_k, H * R  , r)
    t += Score(B, T, H, D_h, R)
    t += Linear_with_LoRA(B, T, H * D_h, L, r)
    return t

def deepseek_r1_store(B, T, L, c_q, c_k, R, H, D_h, r = 16):
    M = 9
    t  = B * T * L
    t += c_q * L
    t += c_q * L
    t += B * T * c_q
    t += R * L
    t += B * T * c_k
    t += c_q * H * D_h
    t += c_q * H * R
    t += c_k + H * D_h
    t += c_k * H * D_h
    t += B * T * H * D_h
    t += B * T * H * R
    t += B * T * R
    t += B * T * H * D_h
    t += B * T * H * D_h
    t += B * T * T
    t += B * T * H * D_h
    t += H * D_h * L

    t += (M + 1) * r * (c_q + L)
    t += (M + 1) * r * (c_k + L)
    t += (M + 1) * r * (R + L)
    t += (M + 1) * r * (c_q + H * D_h)
    t += (M + 1) * r * (c_q + H * R)
    t += (M + 1) * r * (c_k + H * D_h)
    t += (M + 1) * r * (c_k + H * D_h)
    t += (M + 1) * r * (L + H * D_h)
    return t

if __name__ == "__main__":
    # Linear_with_LoRA(2, 3, 4, 5, 6)

    B = 8
    T = 4096
    L = 7168
    c_q = 1536
    c_k = 512
    R = 64
    H = 128
    D_h = 128
    M = 9

    temp = deepseek_r1_computation(B, T, L, c_q, c_k, R, H, D_h)
    temp /= 1024 * 1024 * 1024 * 1024
    temp *= 61
    print(f"cost {temp:.2f} TFLOPS")

    size_of_data_type = 2
    temp = size_of_data_type * deepseek_r1_store(B, T, L, c_q, c_k, R, H, D_h)
    temp /= 1024 * 1024 * 1024
    temp *= 61


    print(f"need {temp:.2f} GBytes")