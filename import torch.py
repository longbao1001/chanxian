import torch
from ultralytics import YOLO

# 1、原权重路径 & 新标签列表【按ID顺序填写，0、1、2...一一对应】
old_pt = r"C:\\Users\\16600\\Desktop\\chanxian\\train8\\weights\\best.pt"
new_pt = "new_best.pt" # 修改后保存新权重

# =========在这里改你的新标签，顺序对应ID========
new_names = {
    0: "zangwu",
    1: "lvpian",
    2: "zhezhou"
}
# ==============================================

# 关键修复：加上 weights_only=False 解决pytorch2.6报错
ckpt = torch.load(old_pt, weights_only=False)
ckpt["model"].names = new_names
# 同步修改训练参数里的names（彻底根治）
if "args" in ckpt:
    ckpt["args"]["names"] = list(new_names.values())

# 保存新pt
torch.save(ckpt, new_pt)
print("修改完成，新权重：",new_pt)

# 验证：读取新模型标签
model = YOLO(new_pt)
print("新权重内置标签：",model.names)