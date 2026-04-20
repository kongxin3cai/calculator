const labels = [
  "批大小（Batch）",
  "序列长度（Sequence length）",
  "隐藏维度（Latent dim）",
  "Q压缩维度（Q comp）",
  "KV压缩维度（KV comp）",
  "注意力头数（Head）",
  "单头维度（Head dim）",
  "MoE中间维度（MoE latent dim）",
  "共享专家数量（Shared experts）",
  "路由专家数量（Routed experts）",
  "激活路由专家数量（Active routed experts）",
  "MoE层数（MoE layers）"
];

const defaultValues = [8, 4096, 7168, 1536, 512, 128, 128, 2048, 1, 256, 8, 58];
const dtypeOptions = [
  { label: "FP64 (8 字节)", bytes: 8 },
  { label: "FP32 / TF32 (4 字节)", bytes: 4 },
  { label: "BF16 (2 字节)", bytes: 2 },
  { label: "FP16 (2 字节)", bytes: 2 },
  { label: "FP8 (1 字节)", bytes: 1 },
  { label: "INT8 (1 字节)", bytes: 1 },
  { label: "INT4 (0.5 字节)", bytes: 0.5 }
];
const dtypeFields = [
  { name: "原始模型权重", id: "dtype-base", defaultLabel: "FP8" },
  { name: "LoRA 权重", id: "dtype-lora", defaultLabel: "FP32" },
  { name: "中间结果", id: "dtype-intermediate", defaultLabel: "BF16" }
];
const loraRankOptions = [8, 16, 32, 64];
const loraRankDefault = 16;
const MODEL_PRESETS_PATH = "./model-presets.json";
const GPU_SPECS_PATH = "./gpu-specs.json";
const llmSizeDefault = "DeepSeek-V3 (671B MoE)";
const gpuModelDefault = "B200 SXM";
const CUSTOM_MODEL_OPTION = "自定义模型";
let modelPresets = {};
let nvidiaTrainingGpus = {};
let isApplyingPreset = false;
let isSyncingIntermediateDtype = false;
let lastLoraCandidates = [];

const inputList = document.getElementById("input-list");
const errorMessage = document.getElementById("error-message");
const resultAttentionCompute = document.getElementById("attention-compute");
const resultAttentionStore = document.getElementById("attention-store");
const resultMoeCompute = document.getElementById("moe-compute");
const resultMoeStore = document.getElementById("moe-store");
const resultTotalCompute = document.getElementById("total-compute");
const resultTotalStore = document.getElementById("total-store");
const moeComputeChart = document.getElementById("moe-compute-chart");
const moeStoreChart = document.getElementById("moe-store-chart");
const moeComputeLegend = document.getElementById("moe-compute-legend");
const moeStoreLegend = document.getElementById("moe-store-legend");
const gpuModelSelect = document.getElementById("gpu-model");
const gpuCountInput = document.getElementById("gpu-count");
const singleTrainTimeInput = document.getElementById("single-train-time-ms");
const gpuIntermediateDtypeSelect = document.getElementById("gpu-intermediate-dtype");
const gpuLoraBtn = document.getElementById("gpu-lora-btn");
const gpuLoraModelSelect = document.getElementById("gpu-lora-model-select");
const gpuLoraResultEl = document.getElementById("gpu-lora-result");
const gpuMemoryEl = document.getElementById("gpu-memory");
const gpuFp32El = document.getElementById("gpu-fp32");
const gpuBf16El = document.getElementById("gpu-bf16");
const gpuFp8El = document.getElementById("gpu-fp8");
const clusterComputeTypeEl = document.getElementById("cluster-compute-type");
const clusterComputeCapacityEl = document.getElementById("cluster-compute-capacity");
const clusterComputeWindowEl = document.getElementById("cluster-compute-window");
const clusterTotalMemoryEl = document.getElementById("cluster-total-memory");
const gpuSpecNoteEl = document.getElementById("gpu-spec-note");
const gpuSourceListEl = document.getElementById("gpu-source-list");

const llmSizeRow = document.createElement("div");
llmSizeRow.className = "input-row";

const llmSizeLabel = document.createElement("label");
llmSizeLabel.textContent = "MoE模型";
llmSizeLabel.htmlFor = "llm-size";

const llmSizeSelect = document.createElement("select");
llmSizeSelect.id = "llm-size";

llmSizeRow.appendChild(llmSizeLabel);
llmSizeRow.appendChild(llmSizeSelect);
inputList.appendChild(llmSizeRow);

const trainingGroup = document.createElement("div");
trainingGroup.className = "input-group";
const trainingGroupTitle = document.createElement("div");
trainingGroupTitle.className = "input-group-title";
trainingGroupTitle.textContent = "训练参数";
trainingGroup.appendChild(trainingGroupTitle);
inputList.appendChild(trainingGroup);

const paramsGroup = document.createElement("div");
paramsGroup.className = "input-group";
const paramsGroupTitle = document.createElement("div");
paramsGroupTitle.className = "input-group-title";
paramsGroupTitle.textContent = "模型参数";
paramsGroup.appendChild(paramsGroupTitle);
inputList.appendChild(paramsGroup);

const dtypeGroup = document.createElement("div");
dtypeGroup.className = "input-group";
const dtypeGroupTitle = document.createElement("div");
dtypeGroupTitle.className = "input-group-title";
dtypeGroupTitle.textContent = "数据类型";
dtypeGroup.appendChild(dtypeGroupTitle);
inputList.appendChild(dtypeGroup);

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
  if (idx === 0 || idx === 1) {
    trainingGroup.appendChild(row);
  } else {
    paramsGroup.appendChild(row);
  }
});

const loraRankRow = document.createElement("div");
loraRankRow.className = "input-row";

const loraRankLabel = document.createElement("label");
loraRankLabel.textContent = "LoRA秩（LoRA rank）";
loraRankLabel.htmlFor = "lora-rank";

const loraRankSelect = document.createElement("select");
loraRankSelect.id = "lora-rank";

loraRankOptions.forEach((rank) => {
  const optionEl = document.createElement("option");
  optionEl.value = String(rank);
  optionEl.textContent = String(rank);
  if (rank === loraRankDefault) {
    optionEl.selected = true;
  }
  loraRankSelect.appendChild(optionEl);
});

loraRankRow.appendChild(loraRankLabel);
loraRankRow.appendChild(loraRankSelect);
paramsGroup.appendChild(loraRankRow);

dtypeFields.forEach((field) => {
  const row = document.createElement("div");
  row.className = "input-row";

  const label = document.createElement("label");
  label.textContent = field.name;
  label.htmlFor = field.id;

  const select = document.createElement("select");
  select.id = field.id;

  dtypeOptions.forEach((option) => {
    const optionEl = document.createElement("option");
    optionEl.value = String(option.bytes);
    optionEl.textContent = option.label;
    if (option.label.startsWith(field.defaultLabel)) {
      optionEl.selected = true;
    }
    select.appendChild(optionEl);
  });

  row.appendChild(label);
  row.appendChild(select);
  dtypeGroup.appendChild(row);
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

function moeComputation(
  B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers,
  baseBytes, loraBytes, intermediateBytes, loraRank
) {
  void baseBytes;
  void loraBytes;
  void intermediateBytes;
  if (!loraRankOptions.includes(loraRank)) {
    throw new Error("请选择有效的 LoRA 秩！");
  }
  const forwardComputation = 2 * L * B * T * Experts * MoELayers + 6 * L * MoELatentDim * B * T * ActiveExperts * MoELayers;
  const backwardComputation = (12 * loraRank * (L + MoELatentDim)  + 6 * L * MoELatentDim  + 5*(L + MoELatentDim)) * B * T * ActiveExperts * MoELayers;
  return {
    forwardComputation,
    backwardComputation,
    totalComputation: forwardComputation + backwardComputation
  };
}

function moeStore(
  B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers,
  baseBytes, loraBytes, intermediateBytes, loraRank
) {
  if ([baseBytes, loraBytes, intermediateBytes].some((v) => !Number.isFinite(v) || v <= 0)) {
    throw new Error("请选择有效的数据类型！");
  }
  if (!loraRankOptions.includes(loraRank)) {
    throw new Error("请选择有效的 LoRA 秩！");
  }
  const baseWeights = 3 * L * MoELatentDim * Experts * MoELayers*baseBytes;
  const loraWeights = 3 * loraRank * (L + MoELatentDim) * Experts * MoELayers*loraBytes;
  const backwardVram = (MoELatentDim * (3 + 4 * intermediateBytes) + 2 * L * intermediateBytes) * B * T * ActiveExperts * MoELayers;
  const adamVram = 3 * loraRank * (L + MoELatentDim) * (loraBytes+ 2*4) * Experts * MoELayers;
  return {
    baseWeights,
    loraWeights,
    backwardVram,
    adamVram,
    totalStore: baseWeights + loraWeights + backwardVram + adamVram
  };
}

function renderPieChart(canvas, legendContainer, segments, valueFormatter = formatCompactValue) {
  if (!canvas || !legendContainer) {
    return;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }

  const total = segments.reduce((sum, item) => sum + item.value, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (total <= 0) {
    legendContainer.innerHTML = "";
    return;
  }

  // Ensure tiny non-zero segments (e.g. adam优化器显存) remain visible on pie.
  const minSliceFraction = 0.0022; // about 0.8 degree
  const minDisplayValue = total * minSliceFraction;
  const displayValues = segments.map((segment) => {
    if (segment.value <= 0) {
      return 0;
    }
    return Math.max(segment.value, minDisplayValue);
  });
  const displayTotal = displayValues.reduce((sum, value) => sum + value, 0);

  const centerX = canvas.width / 2;
  const centerY = canvas.height / 2;
  const radius = Math.min(centerX, centerY) - 8;
  let start = -Math.PI / 2;

  segments.forEach((segment, index) => {
    const slice = (displayValues[index] / displayTotal) * Math.PI * 2;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.arc(centerX, centerY, radius, start, start + slice);
    ctx.closePath();
    ctx.fillStyle = segment.color;
    ctx.fill();
    start += slice;
  });

  // Draw labels on slices: name + value.
  start = -Math.PI / 2;
  ctx.font = "12px Microsoft YaHei, PingFang SC, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  segments.forEach((segment, index) => {
    const slice = (displayValues[index] / displayTotal) * Math.PI * 2;
    const midAngle = start + slice / 2;
    const labelRadius = radius * 0.62;
    const x = centerX + Math.cos(midAngle) * labelRadius;
    const y = centerY + Math.sin(midAngle) * labelRadius;
    const labelLine1 = segment.label;
    const labelLine2 = valueFormatter(segment.value);

    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "rgba(0, 0, 0, 0.18)";
    ctx.lineWidth = 3;
    ctx.strokeText(labelLine1, x, y - 9);
    ctx.fillText(labelLine1, x, y - 9);
    ctx.strokeText(labelLine2, x, y + 9);
    ctx.fillText(labelLine2, x, y + 9);

    start += slice;
  });

  legendContainer.innerHTML = "";
  segments.forEach((segment) => {
    const item = document.createElement("div");
    item.className = "legend-item";

    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.backgroundColor = segment.color;

    const label = document.createElement("span");
    label.textContent = `${segment.label}：${valueFormatter(segment.value)}`;

    item.appendChild(swatch);
    item.appendChild(label);
    legendContainer.appendChild(item);
  });
}

function formatCompactValue(value, teraUnit = "T") {
  if (value >= 1e12) {
    return `${(value / 1e12).toFixed(2)}${teraUnit}`;
  }
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(2)}G`;
  }
  if (value >= 1e6) {
    return `${(value / 1e6).toFixed(2)}M`;
  }
  if (value >= 1e3) {
    return `${(value / 1e3).toFixed(2)}K`;
  }
  return value.toFixed(2);
}

function formatAsTFLOPS(value) {
  const tflops = value / (1024 ** 4);
  if (tflops >= 1024) {
    return `${(tflops / 1024).toFixed(2)} PFLOPS`;
  }
  return `${tflops.toFixed(2)} TFLOPS`;
}

function formatAsGB(value) {
  const gb = value / (1024 ** 3);
  if (gb >= 1024) {
    return `${(gb / 1024).toFixed(2)} TB`;
  }
  return `${gb.toFixed(2)} GB`;
}

function formatTFLOPSValue(tflopsValue) {
  if (tflopsValue >= 1024) {
    return `${(tflopsValue / 1024).toFixed(2)} PFLOPS`;
  }
  return `${tflopsValue.toFixed(2)} TFLOPS`;
}

function formatGBValue(gbValue) {
  if (gbValue >= 1024) {
    return `${(gbValue / 1024).toFixed(2)} TB`;
  }
  return `${gbValue.toFixed(2)} GB`;
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

function parseDtypes() {
  const baseBytes = Number(document.getElementById("dtype-base").value);
  const loraBytes = Number(document.getElementById("dtype-lora").value);
  const intermediateBytes = Number(document.getElementById("dtype-intermediate").value);

  if ([baseBytes, loraBytes, intermediateBytes].some((v) => Number.isNaN(v) || v <= 0)) {
    throw new Error("请选择有效的数据类型！");
  }

  return { baseBytes, loraBytes, intermediateBytes };
}

function parseLoraRank() {
  const loraRank = Number(document.getElementById("lora-rank").value);
  if (!loraRankOptions.includes(loraRank)) {
    throw new Error("请选择有效的 LoRA 秩！");
  }
  return loraRank;
}

function setSelectValue(selectEl, value) {
  if (!selectEl) {
    return;
  }
  const target = String(value);
  const matched = Array.from(selectEl.options).some((option) => option.value === target);
  if (matched) {
    selectEl.value = target;
  }
}

function populateModelOptions() {
  llmSizeSelect.innerHTML = "";
  Object.keys(modelPresets).forEach((modelName) => {
    const optionEl = document.createElement("option");
    optionEl.value = modelName;
    optionEl.textContent = modelName;
    llmSizeSelect.appendChild(optionEl);
  });
  const customOption = document.createElement("option");
  customOption.value = CUSTOM_MODEL_OPTION;
  customOption.textContent = CUSTOM_MODEL_OPTION;
  llmSizeSelect.appendChild(customOption);
}

function populateGpuOptions() {
  if (!gpuModelSelect) {
    return;
  }
  gpuModelSelect.innerHTML = "";
  Object.keys(nvidiaTrainingGpus).forEach((gpuName, index) => {
    const optionEl = document.createElement("option");
    optionEl.value = gpuName;
    optionEl.textContent = gpuName;
    if (gpuName === gpuModelDefault || (index === 0 && !nvidiaTrainingGpus[gpuModelDefault])) {
      optionEl.selected = true;
    }
    gpuModelSelect.appendChild(optionEl);
  });
}

function populateGpuIntermediateDtypeOptions() {
  if (!gpuIntermediateDtypeSelect) {
    return;
  }
  gpuIntermediateDtypeSelect.innerHTML = "";
  dtypeOptions.forEach((option) => {
    const optionEl = document.createElement("option");
    optionEl.value = String(option.bytes);
    optionEl.textContent = option.label;
    gpuIntermediateDtypeSelect.appendChild(optionEl);
  });
}

function syncGpuIntermediateFromLeft() {
  if (!gpuIntermediateDtypeSelect) {
    return;
  }
  const leftIntermediateEl = document.getElementById("dtype-intermediate");
  if (!leftIntermediateEl) {
    return;
  }
  isSyncingIntermediateDtype = true;
  gpuIntermediateDtypeSelect.value = leftIntermediateEl.value;
  isSyncingIntermediateDtype = false;
}

function syncLeftIntermediateFromGpu() {
  if (!gpuIntermediateDtypeSelect) {
    return;
  }
  const leftIntermediateEl = document.getElementById("dtype-intermediate");
  if (!leftIntermediateEl) {
    return;
  }
  isSyncingIntermediateDtype = true;
  leftIntermediateEl.value = gpuIntermediateDtypeSelect.value;
  isSyncingIntermediateDtype = false;
}

function lockDataTypeControls() {
  const dtypeBaseEl = document.getElementById("dtype-base");
  const dtypeLoraEl = document.getElementById("dtype-lora");
  const dtypeIntermediateEl = document.getElementById("dtype-intermediate");
  if (dtypeBaseEl) dtypeBaseEl.disabled = true;
  if (dtypeLoraEl) dtypeLoraEl.disabled = true;
  if (dtypeIntermediateEl) dtypeIntermediateEl.disabled = true;
  if (gpuIntermediateDtypeSelect) gpuIntermediateDtypeSelect.disabled = true;
}

async function initializeGpuSpecs() {
  try {
    const response = await fetch(GPU_SPECS_PATH);
    if (!response.ok) {
      throw new Error(`加载GPU规格失败（HTTP ${response.status}）`);
    }
    const loadedGpuSpecs = await response.json();
    if (!loadedGpuSpecs || typeof loadedGpuSpecs !== "object" || Object.keys(loadedGpuSpecs).length === 0) {
      throw new Error("GPU规格文件为空或格式无效");
    }
    nvidiaTrainingGpus = loadedGpuSpecs;
    populateGpuOptions();
    renderGpuSpecs(gpuModelSelect.value);
  } catch (err) {
    errorMessage.textContent = err.message || "加载GPU规格失败";
  }
}

function renderGpuSpecs(gpuName) {
  const spec = nvidiaTrainingGpus[gpuName];
  if (!spec) {
    return;
  }
  gpuMemoryEl.textContent = spec.memory;
  gpuFp32El.textContent = spec.fp32;
  gpuBf16El.textContent = spec.bf16;
  gpuFp8El.textContent = spec.fp8;
  gpuSourceListEl.innerHTML = "";
  const sourceLinks = Array.isArray(spec.sources) ? spec.sources : [];
  sourceLinks.forEach((source) => {
    const linkEl = document.createElement("a");
    linkEl.className = "gpu-source-link";
    linkEl.href = source.url;
    linkEl.target = "_blank";
    linkEl.rel = "noopener noreferrer";
    linkEl.textContent = `来源：${source.label}`;
    gpuSourceListEl.appendChild(linkEl);
  });
  gpuSpecNoteEl.textContent = "注：算力为官方公开理论峰值（TFLOPS）；部分新卡或消费卡口径因资料版本可能存在差异。";
  renderClusterCapacitySummary();
}

function renderClusterCapacitySummary() {
  const gpuCount = Number(gpuCountInput?.value);
  const validGpuCount = Number.isFinite(gpuCount) && gpuCount > 0 ? gpuCount : 1;
  const singleTrainTimeMs = Number(singleTrainTimeInput?.value);
  const validSingleTrainTimeMs = Number.isFinite(singleTrainTimeMs) && singleTrainTimeMs > 0 ? singleTrainTimeMs : 100;
  const computeTypeBytes = Number(gpuIntermediateDtypeSelect?.value);
  const fp32 = parseTflopsValue(gpuFp32El.textContent);
  const bf16 = parseTflopsValue(gpuBf16El.textContent);
  const fp8 = parseTflopsValue(gpuFp8El.textContent);
  const memoryPerGpuGb = parseGbTotal(gpuMemoryEl.textContent);

  let computeTypeText = "未定义";
  let perGpuCompute = 0;
  if (computeTypeBytes === 1) {
    computeTypeText = "FP8（1字节）";
    perGpuCompute = fp8;
  } else if (computeTypeBytes === 2) {
    computeTypeText = "BF16（2字节）";
    perGpuCompute = bf16;
  } else if (computeTypeBytes === 4) {
    computeTypeText = "FP32（4字节）";
    perGpuCompute = fp32;
  } else if (computeTypeBytes === 0.5) {
    computeTypeText = "0.5字节（当前不支持推荐计算）";
    perGpuCompute = 0;
  }

  const clusterCompute = perGpuCompute * validGpuCount;
  const clusterComputeInWindow = clusterCompute * (validSingleTrainTimeMs / 1000);
  const clusterMemoryGb = memoryPerGpuGb * validGpuCount;
  if (clusterComputeTypeEl) {
    clusterComputeTypeEl.textContent = computeTypeText;
  }
  if (clusterComputeCapacityEl) {
    clusterComputeCapacityEl.textContent = `${clusterCompute.toFixed(1)} TFLOPS`;
  }
  if (clusterComputeWindowEl) {
    clusterComputeWindowEl.textContent = `${clusterComputeInWindow.toFixed(2)} TFLOP`;
  }
  if (clusterTotalMemoryEl) {
    clusterTotalMemoryEl.textContent = formatGBValue(clusterMemoryGb);
  }
}

function parseGpuCount() {
  if (!gpuCountInput) {
    return 1;
  }
  const count = Number(gpuCountInput.value);
  if (!Number.isFinite(count) || count <= 0) {
    throw new Error("GPU数量必须是大于0的数字！");
  }
  return count;
}

function parseTflopsValue(text) {
  const matched = String(text || "").match(/(\d+(?:,\d{3})*(?:\.\d+)?)/);
  if (!matched) {
    return 0;
  }
  return Number(matched[1].replace(/,/g, ""));
}

function parseGbTotal(text) {
  const matches = String(text || "").match(/(\d+(?:\.\d+)?)\s*GB/gi);
  if (!matches) {
    return 0;
  }
  return matches.reduce((sum, item) => {
    const num = Number(item.replace(/[^0-9.]/g, ""));
    return Number.isFinite(num) ? sum + num : sum;
  }, 0);
}

function syncLoraDemandTextWithResultPanel() {
  const computeText = resultTotalCompute.innerHTML.split("<br>")[1] || "-";
  const storeText = resultTotalStore.innerHTML.split("<br>")[1] || "-";
  gpuLoraResultEl.textContent = `总算力需求 ${computeText}，总存储需求 ${storeText}`;
}

function recommendLoraFineTuneModels() {
  try {
    const gpuCount = parseGpuCount();
    const batchInputEl = document.getElementById("input-0");
    const seqLenInputEl = document.getElementById("input-1");
    const batch = Number((batchInputEl?.value || "").trim() || defaultValues[0]);
    const seqLen = Number((seqLenInputEl?.value || "").trim() || defaultValues[1]);
    const dtypes = parseDtypes();
    const loraRank = parseLoraRank();
    if (!Number.isFinite(batch) || batch <= 0 || !Number.isFinite(seqLen) || seqLen <= 0) {
      throw new Error("GPU侧批大小和序列长度必须是大于0的数字！");
    }

    const computeTypeBytes = Number(gpuIntermediateDtypeSelect?.value);
    const fp32 = parseTflopsValue(gpuFp32El.textContent);
    const bf16 = parseTflopsValue(gpuBf16El.textContent);
    const fp8 = parseTflopsValue(gpuFp8El.textContent);
    const gpuMemoryGb = parseGbTotal(gpuMemoryEl.textContent) * gpuCount;
    const singleTrainTimeMs = Number(singleTrainTimeInput?.value);
    const validSingleTrainTimeMs = Number.isFinite(singleTrainTimeMs) && singleTrainTimeMs > 0 ? singleTrainTimeMs : 100;
    let selectedComputeType = "";
    let selectedComputeValue = 0;
    if (computeTypeBytes === 0.5) {
      throw new Error("当前算力类型不支持适合的LoRA微调模型计算。");
    } else if (computeTypeBytes === 1) {
      selectedComputeType = "FP8";
      selectedComputeValue = fp8;
    } else if (computeTypeBytes === 2) {
      selectedComputeType = "BF16";
      selectedComputeValue = bf16;
    } else if (computeTypeBytes === 4) {
      selectedComputeType = "FP32";
      selectedComputeValue = fp32;
    } else {
      throw new Error("当前算力类型不支持适合的LoRA微调模型计算。");
    }
    const clusterComputeTflops = gpuCount * selectedComputeValue;
    const clusterComputeWindowTflop = clusterComputeTflops * (validSingleTrainTimeMs / 1000);

    const candidates = Object.entries(modelPresets)
      .map(([modelName, preset]) => {
        if (!Array.isArray(preset.inputs) || preset.inputs.length !== 10) {
          throw new Error(`模型预设 ${modelName} 的参数长度无效，应为10项。`);
        }

        // Only Batch/Sequence come from page; all other model parameters come from model-presets.json.
        const [
          L,
          c_q,
          c_k,
          H,
          D_h,
          MoELatentDim,
          SharedExperts,
          RoutedExperts,
          ActiveRoutedExperts,
          MoELayers
        ] = preset.inputs;
        const Experts = SharedExperts + RoutedExperts;
        const ActiveExperts = SharedExperts + ActiveRoutedExperts;
        const R = 64;
        const M = 9;

        let attentionCompute = deepseekR1Computation(batch, seqLen, L, c_q, c_k, R, H, D_h, M);
        attentionCompute /= (1024 ** 4);
        attentionCompute *= 61;
        let attentionStore = 2 * deepseekR1Store(batch, seqLen, L, c_q, c_k, R, H, D_h);
        attentionStore /= (1024 ** 3);
        attentionStore *= 61;
        const computeData = moeComputation(
          batch, seqLen, L, MoELatentDim, Experts, ActiveExperts, MoELayers,
          dtypes.baseBytes, dtypes.loraBytes, dtypes.intermediateBytes, loraRank
        );
        const storeData = moeStore(
          batch, seqLen, L, MoELatentDim, Experts, ActiveExperts, MoELayers,
          dtypes.baseBytes, dtypes.loraBytes, dtypes.intermediateBytes, loraRank
        );

        const moeComputeTflops = computeData.totalComputation / (1024 ** 4);
        const moeStoreGb = storeData.totalStore / (1024 ** 3);
        // Model requirement uses the same "算力/存储" formula basis as panel totals,
        // with current batch/seq and candidate model-specific parameters.
        const totalComputeTflops = attentionCompute + moeComputeTflops;
        const totalStoreGb = attentionStore + moeStoreGb;
        const computeFit = totalComputeTflops <= clusterComputeWindowTflop;
        const memoryFit = totalStoreGb <= gpuMemoryGb;

        return {
          modelName,
          totalComputeTflops,
          totalStoreGb,
          computeFit,
          memoryFit
        };
      })
      .filter((item) => item.computeFit && item.memoryFit)
      .sort((a, b) => a.totalStoreGb - b.totalStoreGb || a.totalComputeTflops - b.totalComputeTflops);

    gpuLoraModelSelect.innerHTML = "";
    lastLoraCandidates = [];
    if (candidates.length === 0) {
      const optionEl = document.createElement("option");
      optionEl.value = "";
      optionEl.textContent = "无满足条件的模型";
      gpuLoraModelSelect.appendChild(optionEl);
      gpuLoraResultEl.textContent =
        `输入：算力类型 ${computeTypeBytes} 字节（使用${selectedComputeType}算力），GPU数量 ${gpuCount}，单次训练时间 ${validSingleTrainTimeMs}ms，批大小 ${batch}，序列长度 ${seqLen}。在当前集群算力能力×单次训练时间 ${clusterComputeWindowTflop.toFixed(2)} TFLOP、集群总显存 ${formatGBValue(gpuMemoryGb)} 下，没有满足约束的模型。`;
      return;
    }

    lastLoraCandidates = candidates;
    candidates.forEach((item, idx) => {
      const optionEl = document.createElement("option");
      optionEl.value = item.modelName;
      optionEl.textContent = `${idx + 1}. ${item.modelName}`;
      gpuLoraModelSelect.appendChild(optionEl);
    });

    const first = candidates[0];
    gpuLoraModelSelect.value = first.modelName;
    if (modelPresets[first.modelName]) {
      llmSizeSelect.value = first.modelName;
      applyModelPreset(first.modelName);
      calculate();
    }
    syncLoraDemandTextWithResultPanel();
  } catch (err) {
    lastLoraCandidates = [];
    gpuLoraResultEl.textContent = err.message || "推荐计算失败";
  }
}

async function initializeModelPresets() {
  try {
    const response = await fetch(MODEL_PRESETS_PATH);
    if (!response.ok) {
      throw new Error(`加载模型预设失败（HTTP ${response.status}）`);
    }
    const loadedPresets = await response.json();
    if (!loadedPresets || typeof loadedPresets !== "object" || Object.keys(loadedPresets).length === 0) {
      throw new Error("模型预设文件为空或格式无效");
    }
    modelPresets = loadedPresets;
    populateModelOptions();
    const initialModel = modelPresets[llmSizeDefault] ? llmSizeDefault : Object.keys(modelPresets)[0];
    llmSizeSelect.value = initialModel;
    applyModelPreset(initialModel);
    calculate();
  } catch (err) {
    errorMessage.textContent = err.message || "加载模型预设失败";
  }
}

function applyModelPreset(modelName) {
  const preset = modelPresets[modelName];
  if (!preset) {
    return;
  }
  isApplyingPreset = true;

  preset.inputs.forEach((value, idx) => {
    const inputEl = document.getElementById(`input-${idx + 2}`);
    if (inputEl) {
      inputEl.value = String(value);
    }
  });
  syncGpuIntermediateFromLeft();
  isApplyingPreset = false;
}

function markCustomModel() {
  if (isApplyingPreset) {
    return;
  }
  llmSizeSelect.value = CUSTOM_MODEL_OPTION;
}

function calculate() {
  try {
    errorMessage.textContent = "";
    const numbers = parseInputs();
    const dtypes = parseDtypes();
    const loraRank = parseLoraRank();
    const gpuCount = parseGpuCount();

    const B = numbers[0];
    const T = numbers[1];
    const L = numbers[2];
    const c_q = numbers[3];
    const c_k = numbers[4];
    const R = 64;
    const H = numbers[5];
    const D_h = numbers[6];
    const MoELatentDim = numbers[7];
    const SharedExperts = numbers[8];
    const RoutedExperts = numbers[9];
    const ActiveRoutedExperts = numbers[10];
    const Experts = SharedExperts + RoutedExperts;
    const ActiveExperts = SharedExperts + ActiveRoutedExperts;
    const MoELayers = numbers[11];

    const M = 9;
    let attentionComputeTFLOPS = deepseekR1Computation(B, T, L, c_q, c_k, R, H, D_h, M);
    attentionComputeTFLOPS /= 1024 * 1024 * 1024 * 1024;
    attentionComputeTFLOPS *= 61;
    resultAttentionCompute.innerHTML = `Attention算力：<br>${formatTFLOPSValue(attentionComputeTFLOPS)}`;

    const sizeOfDataType = 2;
    let attentionStoreGB = sizeOfDataType * deepseekR1Store(B, T, L, c_q, c_k, R, H, D_h);
    attentionStoreGB /= 1024 * 1024 * 1024;
    attentionStoreGB *= 61;
    resultAttentionStore.innerHTML = `Attention存储：<br>${formatGBValue(attentionStoreGB)}`;

    const moeComputationData = moeComputation(
      B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers,
      dtypes.baseBytes, dtypes.loraBytes, dtypes.intermediateBytes, loraRank
    );
    const moeTotal = moeComputationData.totalComputation;
    const moeComputeTFLOPS = moeTotal / (1024 ** 4);
    resultMoeCompute.innerHTML = `MoE算力：<br>${formatAsTFLOPS(moeTotal)}`;

    const moeStoreData = moeStore(
      B, T, L, MoELatentDim, Experts, ActiveExperts, MoELayers,
      dtypes.baseBytes, dtypes.loraBytes, dtypes.intermediateBytes, loraRank
    );
    const moeTemp = moeStoreData.totalStore;
    const moeStoreGB = moeTemp / (1024 ** 3);
    resultMoeStore.innerHTML = `MoE存储：<br>${formatAsGB(moeTemp)}`;

    const totalComputeTFLOPS = attentionComputeTFLOPS + moeComputeTFLOPS;
    const totalStoreGB = attentionStoreGB + moeStoreGB;
    resultTotalCompute.innerHTML = `总算力（Attention + MoE）：<br>${formatTFLOPSValue(totalComputeTFLOPS)}`;
    resultTotalStore.innerHTML = `总存储（Attention + MoE）：<br>${formatGBValue(totalStoreGB)}`;

    renderPieChart(moeComputeChart, moeComputeLegend, [
      { label: "正向计算", value: moeComputationData.forwardComputation, color: "#2563eb" },
      { label: "反向计算", value: moeComputationData.backwardComputation, color: "#60a5fa" }
    ], formatAsTFLOPS);

    renderPieChart(moeStoreChart, moeStoreLegend, [
      { label: "原始权重", value: moeStoreData.baseWeights, color: "#1d4ed8" },
      { label: "LoRA权重", value: moeStoreData.loraWeights, color: "#0ea5e9" },
      { label: "反向传播显存", value: moeStoreData.backwardVram, color: "#14b8a6" },
      { label: "adam优化器显存", value: moeStoreData.adamVram, color: "#22c55e" }
    ], formatAsGB);
  } catch (err) {
    errorMessage.textContent = err.message || "请输入有效数字！";
  }
}

document.getElementById("calc-btn").addEventListener("click", calculate);
llmSizeSelect.addEventListener("change", (event) => {
  if (event.target.value === CUSTOM_MODEL_OPTION) {
    return;
  }
  applyModelPreset(event.target.value);
  calculate();
});

defaultValues.forEach((_, idx) => {
  const inputEl = document.getElementById(`input-${idx}`);
  if (inputEl) {
    // Batch and Sequence len are excluded from switching model to "custom".
    if (idx !== 0 && idx !== 1) {
      inputEl.addEventListener("input", markCustomModel);
      inputEl.addEventListener("change", markCustomModel);
    }
  }
});
const leftBatchInput = document.getElementById("input-0");
const leftSeqLenInput = document.getElementById("input-1");
if (leftBatchInput) {
  leftBatchInput.addEventListener("input", () => {
    calculate();
  });
}
if (leftSeqLenInput) {
  leftSeqLenInput.addEventListener("input", () => {
    calculate();
  });
}
document.getElementById("dtype-base").addEventListener("change", markCustomModel);
document.getElementById("dtype-lora").addEventListener("change", markCustomModel);
document.getElementById("dtype-intermediate").addEventListener("change", () => {
  if (isSyncingIntermediateDtype) {
    return;
  }
  syncGpuIntermediateFromLeft();
  renderClusterCapacitySummary();
  markCustomModel();
});

initializeModelPresets();
gpuModelSelect.addEventListener("change", (event) => {
  renderGpuSpecs(event.target.value);
});
initializeGpuSpecs();
populateGpuIntermediateDtypeOptions();
syncGpuIntermediateFromLeft();
lockDataTypeControls();
renderClusterCapacitySummary();
gpuIntermediateDtypeSelect.addEventListener("change", () => {
  if (isSyncingIntermediateDtype) {
    return;
  }
  syncLeftIntermediateFromGpu();
  renderClusterCapacitySummary();
  markCustomModel();
  calculate();
});
gpuCountInput.addEventListener("input", renderClusterCapacitySummary);
singleTrainTimeInput.addEventListener("input", renderClusterCapacitySummary);
gpuLoraBtn.addEventListener("click", recommendLoraFineTuneModels);
gpuLoraModelSelect.addEventListener("change", () => {
  const selected = lastLoraCandidates.find((item) => item.modelName === gpuLoraModelSelect.value);
  if (!selected) {
    return;
  }
  if (modelPresets[selected.modelName]) {
    llmSizeSelect.value = selected.modelName;
    applyModelPreset(selected.modelName);
    calculate();
  }
  syncLoraDemandTextWithResultPanel();
});
