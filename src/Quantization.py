import torch
import torch.nn as nn
import os

# 定义一个简单的模型 (例如，一个小型卷积网络)
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 7 * 7, 10) # 假设输入是 28x28 图像

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7)
        return self.fc(x)

# 辅助函数：获取模型大小
def print_model_size(model, label="模型"):
    torch.save(model.state_dict(), "temp_model.p")
    size_mb = os.path.getsize("temp_model.p") / 1e6
    print(f"{label} 大小: {size_mb:.2f} MB")
    os.remove("temp_model.p")
    return size_mb

# 实例化模型
model_fp32 = SimpleConvNet()

# 打印原始模型大小
size_fp32 = print_model_size(model_fp32, "原始 FP32 模型")

# 1. 设置量化配置
# 'fbgemm' 是针对 x86 CPU 优化的量化后端
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print("\n量化配置:", model_fp32.qconfig)

# 2. 准备模型：插入观察器（Observer）来收集激活值的统计信息
model_prepared = torch.quantization.prepare(model_fp32, inplace=False)

# 3. 校准模型：使用少量代表性数据运行模型，收集激活值范围
# 这是一个模拟的校准过程，实际中会用验证集数据
print("\n正在校准模型...")
with torch.no_grad():
    for _ in range(10): # 模拟10个批次的校准数据
        dummy_input = torch.randn(1, 1, 28, 28) # 假设输入是 1x28x28
        model_prepared(dummy_input)
print("校准完成。")

# 4. 转换模型：将观察器替换为量化模块
model_quantized = torch.quantization.convert(model_prepared, inplace=False)

# 打印量化后模型大小
size_quantized = print_model_size(model_quantized, "量化 INT8 模型")

print(f"\n模型大小从 {size_fp32:.2f} MB 减小到 {size_quantized:.2f} MB")
print(f"减小比例: {((size_fp32 - size_quantized) / size_fp32 * 100):.2f}%")

# 验证量化后的模型
dummy_input = torch.randn(1, 1, 28, 28)
output_quantized = model_quantized(dummy_input)
print("\n量化模型输出示例 (量化后的张量):")
print(output_quantized)
