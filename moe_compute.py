from huggingface_hub.constants import default_home



def moe_computation(B, T, L, MoE_latent_dim, Experts, Active_experts, MoE_layers, r = 16):
    # MoE computation uses total experts and active experts
    t = 8 * L * MoE_latent_dim * B * T * Active_experts * MoE_layers
    return t

def moe_store(B, T, L, MoE_latent_dim, Experts, Active_experts, MoE_layers, r = 16):
    t = 3 * L * MoE_latent_dim * Experts * MoE_layers
    t += 12 * r * (L + MoE_latent_dim) * Experts * MoE_layers
    t += (9 * MoE_latent_dim + 4 * L)  * B * T * Active_experts * MoE_layers
    return t
