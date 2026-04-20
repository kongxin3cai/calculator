const labels = [
  "Batch",
  "Sequence len",
  "Latent dim",
  "Q comp",
  "KV comp",
  "Head",
  "Head dim",
  "MoE latent dim",
  "Experts",
  "Active experts",
  "MoE layers"
];

const defaultValues = [8, 4096, 7168, 1536, 512, 128, 128, 2048, 257, 9, 59];

const inputList = document.getElementById("input-list");
const errorMessage = document.getElementById("error-message");
const resultAttentionCompute = document.getElementById("attention-compute");
const resultAttentionStore = document.getElementById("attention-store");
const resultMoeCompute = document.getElementById("moe-compute");
const resultMoeStore = document.getElementById("moe-store");

labels.forEach((name, idx) => {
  const row = document.createElement("div");
  row.className = "input-row";

  const label = document.createElement("label");
  label.textContent = name;
  label.htmlFor = `input-${idx}`;

  const input = document.createElement("input");
  input.id = `input-${idx}`;
  input.type = "text";
  input.placeholder = `不输入默认为 ${defaultValues[idx]}`;

  row.appendChild(label);
  row.appendChild(input);
  inputList.appendChild(row);
});

function linearWithLoRA(S, M, K, N, r) {
  const forwardComputation = 2 * S * M * K * N + 2 * S * M * K * r + 2 * S * M * r * N + S * M * N;
  const backComputation = S * M * (2 * N * K + 6 * N * r + 6 * r * K + K) + K * r + r * N;
  return forwardComputation + backComputation;
}

function score(B, T, H, d_h, d_hR) {
  let temp = B * H * (T * T * (2 * d_h + 2 * d_hR - 1) + T * (3 * T - 1) + 2 * T * T * d_h - T * d_h);
  temp += B * H * (4 * T * T * (d_h + d_hR) - 2 * T * (d_h + d_hR) + 4 * d_h * T * T - T * T - T * d_h);
  return temp;
}

function deepseekR1Computation(B, T, L, c_q, c_k, R, H, D_h, r = 16) {
  let t = linearWithLoRA(B, T, L, c_q, r);
  t += linearWithLoRA(B, T, L, c_k, r);
  t += linearWithLoRA(B, T, L, R, r);
  t += linearWithLoRA(B, T, c_q, H * D_h, r);
  t += linearWithLoRA(B, T, c_q, H * R, r);
  t += linearWithLoRA(B, T, c_k, H * D_h, r);
  t += linearWithLoRA(B, T, c_k, H * R, r);
  t += score(B, T, H, D_h, R);
  t += linearWithLoRA(B, T, H * D_h, L, r);
  return t;
}

function deepseekR1Store(B, T, L, c_q, c_k, R, H, D_h, r = 16) {
  const M = 9;
  let t = B * T * L;
  t += c_q * L;
  t += c_q * L;
  t += B * T * c_q;
  t += R * L;
  t += B * T * c_k;
  t += c_q * H * D_h;
  t += c_q * H * R;
  t += c_k + H * D_h;
  t += c_k * H * D_h;
  t += B * T * H * D_h;
  t += B * T * H * R;
  t += B * T * R;
  t += B * T * H * D_h;
  t += B * T * H * D_h;
  t += B * T * T;
  t += B * T * H * D_h;
  t += H * D_h * L;

  t += (M + 1) * r * (c_q + L);
  t += (M + 1) * r * (c_k + L);
  t += (M + 1) * r * (R + L);
  t += (M + 1) * r * (c_q + H * D_h);
  t += (M + 1) * r * (c_q + H * R);
  t += (M + 1) * r * (c_k + H * D_h);
  t += (M + 1) * r * (c_k + H * D_h);
  t += (M + 1) * r * (L + H * D_h);
  return t;
}

function moeComputation(B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers, r = 16) {
  return 8 * L * MoELatentDim * B * T * ActiveExperts * MoELayers;
}

function moeStore(B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers, r = 16) {
  let t = 3 * L * MoELatentDim * Experts * MoELayers;
  t += 12 * r * (L + MoELatentDim) * Experts * MoELayers;
  t += (9 * MoELatentDim + 4 * L) * B * T * ActiveExperts * MoELayers;
  return t;
}

function parseInputs() {
  const values = [];
  for (let i = 0; i < defaultValues.length; i += 1) {
    const el = document.getElementById(`input-${i}`);
    const raw = el.value.trim();
    if (raw === "") {
      values.push(defaultValues[i]);
      continue;
    }
    const num = Number(raw);
    if (Number.isNaN(num)) {
      throw new Error("请输入有效数字！");
    }
    values.push(num);
  }
  return values;
}

function calculate() {
  try {
    errorMessage.textContent = "";
    const numbers = parseInputs();

    const B = numbers[0];
    const T = numbers[1];
    const L = numbers[2];
    const c_q = numbers[3];
    const c_k = numbers[4];
    const R = 64;
    const H = numbers[5];
    const D_h = numbers[6];
    const MoELatentDim = numbers[7];
    const Experts = numbers[8];
    const ActiveExperts = numbers[9];
    const MoELayers = numbers[10];

    let total = deepseekR1Computation(B, T, L, c_q, c_k, R, H, D_h, 16);
    total /= 1024 * 1024 * 1024 * 1024;
    total *= 61;
    resultAttentionCompute.innerHTML = `Attention算力：<br>${total.toFixed(2)} TFLOPS`;

    const sizeOfDataType = 2;
    let temp = sizeOfDataType * deepseekR1Store(B, T, L, c_q, c_k, R, H, D_h, 16);
    temp /= 1024 * 1024 * 1024;
    temp *= 61;
    resultAttentionStore.innerHTML = `Attention存储：<br>${temp.toFixed(2)} GB`;

    let moeTotal = moeComputation(B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers, 16);
    moeTotal /= 1024 * 1024 * 1024 * 1024;
    resultMoeCompute.innerHTML = `MoE算力：<br>${moeTotal.toFixed(2)} TFLOPS`;

    let moeTemp = moeStore(B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers, 16);
    moeTemp /= 1024 * 1024 * 1024;
    resultMoeStore.innerHTML = `MoE存储：<br>${moeTemp.toFixed(2)} GB`;
  } catch (err) {
    errorMessage.textContent = err.message || "请输入有效数字！";
  }
}

document.getElementById("calc-btn").addEventListener("click", calculate);
