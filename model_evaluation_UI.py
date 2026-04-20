# 导入库
import customtkinter as ctk
from tkinter import messagebox
from compute_finetuning_source import deepseek_r1_computation, deepseek_r1_store
from moe_compute import moe_computation, moe_store

# 主题设置
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

# 主窗口
app = ctk.CTk()
app.title("LoRA微调计算器")
app.geometry("875x800")
app.resizable(False, False)

# 主容器（左右布局）
main_frame = ctk.CTkFrame(app, fg_color="transparent")
main_frame.pack(pady=20, padx=20, fill="both", expand=True)

# ========== 左侧：输入区域 ==========
left_frame = ctk.CTkFrame(
    main_frame,
    width=400,
    fg_color="#ffffff",      # 纯白背景
    border_color="#e0e0e0",
    border_width=1
)
left_frame.pack(side="left", fill="both", expand=True, padx=15, pady=15)
left_frame.pack_propagate(False)

# 标题
title_label = ctk.CTkLabel(
    left_frame,
    text="LoRA 微调计算",
    font=("微软雅黑", 24, "bold")
)
title_label.pack(pady=25)

# 6个输入框容器
input_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
input_frame.pack(pady=10, padx=20, fill="x")

# 11个输入框
input_entries = []
text_para = ["Batch", "Sequence len", "Latent dim", "Q comp", "KV comp", "Head", "Head dim", "MoE latent dim", "Experts", "Active experts", "MoE layers"]
orig_para = ["8", "4096", "7168", "1536", "512", "128", "128", "2048", "257", "9", "59"]
orig_para_float = [8, 4096, 7168, 1536, 512, 128, 128, 2048, 257, 9, 59]
for i in range(1, 12):
    row = ctk.CTkFrame(input_frame, fg_color="transparent")
    row.pack(fill="x", pady=7)

    label = ctk.CTkLabel(row, text = text_para[i - 1], font=("微软雅黑", 15))
    label.pack(side="left", padx=10)

    entry = ctk.CTkEntry(
        row,
        width=220,
        font=("微软雅黑", 15),
        placeholder_text=f"不输入默认为 {orig_para[i - 1]}"
    )
    entry.pack(side="right", padx=10)
    input_entries.append(entry)

# 计算按钮
calc_btn = ctk.CTkButton(
    left_frame,
    text="计算总和",
    font=("微软雅黑", 17, "bold"),
    command=lambda: calculate(),
    height=50
)
calc_btn.pack(pady=30)

# ========== 右侧：结果显示区域 ==========
right_frame = ctk.CTkFrame(
    main_frame,
    width=300,
    fg_color="#f0f7ff"
)
right_frame.pack(side="right", fill="both", expand=True, padx=15, pady=15)
right_frame.pack_propagate(False)

# 右侧标题
right_title = ctk.CTkLabel(
    right_frame,
    text="算力/存储",
    font=("微软雅黑", 24, "bold"),
    text_color="#005fcc"
)
right_title.pack(pady=50)

# 结果显示标签
result_label = ctk.CTkLabel(
    right_frame,
    text="Attention算力：\n",
    font=("微软雅黑", 36, "bold"),
    text_color="#0078d7"
    #wraplength=280
)
result_label.pack(pady=20)

result_label1 = ctk.CTkLabel(
    right_frame,
    text="Attention存储：\n",
    font=("微软雅黑", 36, "bold"),
    text_color="#0078d7"
)
result_label1.pack(pady=20)

result_label2 = ctk.CTkLabel(
    right_frame,
    text="MoE算力：\n",
    font=("微软雅黑", 36, "bold"),
    text_color="#0078d7"
)
result_label2.pack(pady=20)

result_label3 = ctk.CTkLabel(
    right_frame,
    text="MoE存储：\n",
    font=("微软雅黑", 36, "bold"),
    text_color="#0078d7"
)
result_label3.pack(pady=20)

# ========== 计算函数 ==========
orig_para_float = [8, 4096, 7168, 1536, 512, 128, 128, 2048, 257, 9, 59]
def calculate():
    try:
        numbers = []
        for index, entry in enumerate(input_entries):
            val = entry.get().strip()
            if val == "":
                numbers.append(float(orig_para_float[index]))
            else:
                numbers.append(float(val))

        B = numbers[0]
        T = numbers[1]
        L = numbers[2]
        c_q = numbers[3]
        c_k = numbers[4]
        R = 64
        H = numbers[5]
        D_h = numbers[6]
        MoE_latent_dim = numbers[7]
        Experts = numbers[8]
        Active_experts = numbers[9]
        MoE_layers = numbers[10]
        M = 9


        total = deepseek_r1_computation(B, T, L, c_q, c_k, R, H, D_h, M)
        total /= 1024 * 1024 * 1024 * 1024
        total *= 61
        result_label.configure(text=f"Attention算力：\n{total:.2f} TFLOPS")

        size_of_data_type = 2
        temp = size_of_data_type * deepseek_r1_store(B, T, L, c_q, c_k, R, H, D_h)
        temp /= 1024 * 1024 * 1024
        temp *= 61
        result_label1.configure(text=f"Attention存储：\n{temp:.2f} GB")

        moe_total = moe_computation(B, T, L, MoE_latent_dim, Experts, Active_experts, MoE_layers)
        moe_total /= 1024 * 1024 * 1024 * 1024 
        result_label2.configure(text=f"MoE算力：\n{moe_total:.2f} TFLOPS")

        moe_temp = moe_store(B, T, L, MoE_latent_dim, Experts, Active_experts, MoE_layers)
        moe_temp /= 1024 * 1024 * 1024 
        result_label3.configure(text=f"MoE存储：\n{moe_temp:.2f} GB")

    except ValueError:
        messagebox.showerror("输入错误", "请输入有效数字！")

# 启动程序
app.mainloop()
