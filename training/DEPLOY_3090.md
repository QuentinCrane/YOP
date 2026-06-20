# RTX 3090 训练环境部署指南

本文档说明如何将训练 Pipeline 复制到 RTX 3090 机器上并完成环境搭建。

---

## 1. 需要复制的文件

将整个 `training/` 目录复制到 3090 机器：

```bash
# 方法1：直接复制文件夹
scp -r night-road-vision/training/ user@3090-machine:~/night-road/training/

# 方法2：打包传输
tar czf night-road-training.tar.gz night-road-vision/training/
scp night-road-training.tar.gz user@3090-machine:~/night-road/
# 在 3090 机器上解压：
tar xzf night-road-training.tar.gz
```

目录结构：

```
training/
├── TRAINING_GUIDE.md          # 完整训练指南
├── DEPLOY_3090.md             # 本文档
├── configs/
│   ├── augmentation_night.yaml
│   ├── hyperparams_stage1.yaml
│   ├── hyperparams_stage2.yaml
│   └── device_rtx3090.yaml
├── scripts/
│   ├── prepare_datasets.py    # 数据预处理
│   ├── train.py               # 两阶段训练
│   ├── evaluate.py            # 评估
│   ├── export_models.py       # 模型导出
│   ├── download_model.py      # 下载预训练模型
│   └── setup_model.py         # 模型下载+导出
└── docs/
    ├── DATASET_CARD.md
    └── MODEL_CARD.md
```

---

## 2. 环境搭建

### 2.1 系统要求

| 项目 | 要求 |
|---|---|
| GPU | NVIDIA RTX 3090 24GB |
| 系统内存 | 32GB+ 推荐 |
| 磁盘 | 200GB+ SSD 剩余空间 |
| OS | Ubuntu 22.04 / Windows 10/11 |
| CUDA | 12.x |
| cuDNN | 8.9+ |

### 2.2 安装 Conda

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 或 Windows：下载安装包
# https://docs.conda.io/en/latest/miniconda.html
```

### 2.3 创建环境

```bash
conda create -n night-road python=3.11 -y
conda activate night-road
```

### 2.4 安装 PyTorch (CUDA 12.x)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

预期输出：

```
PyTorch: 2.x.x+cu121
CUDA: True
GPU: NVIDIA GeForce RTX 3090
```

### 2.5 安装项目依赖

```bash
pip install ultralytics albumentations opencv-python pandas matplotlib pyyaml tqdm
```

### 2.6 验证 Ultralytics

```bash
yolo checks
```

应该显示 GPU 信息和所有依赖状态。

---

## 3. 数据集下载

### 3.1 BDD100K（必须）

- 官网：https://bdd-data.berkeley.edu/
- 需要注册账号
- 下载内容：
  - `bdd100k_images_100k.zip`（100K 图片）
  - `bdd100k_labels_release.zip`（标注）
- 大小：约 100GB
- 许可：仅限研究

### 3.2 NightOwls（推荐）

- 官网：https://www.nightowls-dataset.org/
- 下载夜间行人标注和图片
- 大小：约 5GB
- 许可：仅限研究

### 3.3 ExDark（推荐）

- 官网：https://github.com/cs-chan/ExDark-Dataset
- GitHub 直接下载
- 大小：约 2GB
- 许可：仅限研究

### 3.4 NOD（可选）

- 论文：https://arxiv.org/abs/2110.10364
- 需要联系作者或从论文提供的链接下载

### 3.5 目录结构

下载后组织为：

```
training/
├── raw_datasets/
│   ├── bdd100k/
│   │   ├── images/100k/
│   │   └── labels/
│   ├── nightowls/
│   │   ├── images/
│   │   └── annotations/
│   └── exdark/
│       ├── images/
│       └── annotations/
└── datasets/          # 处理后的 YOLO 格式
    └── night_road/
```

---

## 4. 数据预处理

```bash
cd ~/night-road/training

# 1. 处理 BDD100K
python scripts/prepare_datasets.py bdd100k \
    --input raw_datasets/bdd100k \
    --output datasets/night_road

# 2. 处理 NightOwls
python scripts/prepare_datasets.py nightowls \
    --input raw_datasets/nightowls \
    --output datasets/night_road

# 3. 处理 ExDark
python scripts/prepare_datasets.py exdark \
    --input raw_datasets/exdark \
    --output datasets/night_road

# 4. 按序列划分数据集
python scripts/prepare_datasets.py split \
    --dataset datasets/night_road

# 5. 生成 YOLO 配置文件
python scripts/prepare_datasets.py create-yaml \
    --dataset datasets/night_road

# 6. 查看统计信息
python scripts/prepare_datasets.py stats \
    --dataset datasets/night_road
```

---

## 5. 训练

### 5.1 下载预训练模型

```bash
python scripts/download_model.py
```

会下载 `yolo26n.pt`（如果可用）或 `yolo11n.pt`。

### 5.2 阶段一：公共数据预适配

```bash
python scripts/train.py stage1 \
    --data datasets/night_road/night_road.yaml
```

- 模型：yolo26n.pt（COCO 预训练）
- 数据：BDD100K 夜间 + NightOwls + ExDark
- Epochs：120
- 图片尺寸：640
- 预估时间：8-12 小时（RTX 3090）
- 输出：`runs/night-road/stage1_public/weights/best.pt`

### 5.3 阶段二：手机数据微调

需要先采集手机数据并添加到数据集：

```bash
# 手机数据处理后：
python scripts/train.py stage2 \
    --data datasets/night_road_phone/night_road.yaml \
    --stage1-weights runs/night-road/stage1_public/weights/best.pt
```

- Epochs：50
- 学习率：0.001（更低）
- 预估时间：3-5 小时（RTX 3090）
- 输出：`runs/night-road/stage2_finetune/weights/best.pt`

### 5.4 一键两阶段

```bash
python scripts/train.py both \
    --data-stage1 datasets/night_road/night_road.yaml \
    --data-stage2 datasets/night_road_phone/night_road.yaml
```

---

## 6. 评估

```bash
# 标准评估
python scripts/evaluate.py standard \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --data datasets/night_road/night_road.yaml

# 切片评估（夜间行人、黑衣、小目标等）
python scripts/evaluate.py slices \
    --model best.pt \
    --slices-dir evaluation/slices/

# 模型对比
python scripts/evaluate.py compare \
    --models best.pt model_fp16.tflite model_int8.tflite
```

---

## 7. 模型导出

```bash
# 导出所有格式
python scripts/export_models.py all \
    --model runs/night-road/stage2_finetune/weights/best.pt \
    --calibration-data datasets/night_road/night_road.yaml

# 只导出 FP16
python scripts/export_models.py tflite-fp16 \
    --model best.pt

# 只导出 INT8
python scripts/export_models.py tflite-int8 \
    --model best.pt \
    --calibration-data datasets/night_road/night_road.yaml
```

导出产物：

| 文件 | 输入尺寸 | 格式 | 用途 |
|---|---|---|---|
| eco_416x256_fp16.tflite | 416×256 | FP16 | 低端/过热设备 |
| balanced_512x320_fp16.tflite | 512×320 | FP16 | 默认平衡档 |
| fine_640x384_fp16.tflite | 640×384 | FP16 | 高端设备 |
| balanced_512x320_int8.tflite | 512×320 | INT8 | CPU/NPU 推理 |

---

## 8. 将模型传回 Android 项目

```bash
# 从 3090 机器复制回开发机
scp user@3090-machine:~/night-road/training/exports/*.tflite \
    night-road-vision/android-app/app/src/main/assets/models/
```

---

## 9. RTX 3090 性能参考

| 指标 | 预期值 |
|---|---|
| 阶段一训练时间 | 8-12 小时 |
| 阶段二训练时间 | 3-5 小时 |
| 推荐 batch size (640px) | 48 |
| 推荐 batch size (512px) | 64 |
| 显存占用 | ~18-20 GB |
| FP16 AMP | 启用 |
| 数据加载 workers | 8 |
| 缓存策略 | disk |

---

## 10. 常见问题

**Q: CUDA OOM (显存不足)**
减小 batch size：`python scripts/train.py stage1 --batch 32`

**Q: 训练 loss 不下降**
检查学习率和数据标注质量。

**Q: 夜间行人 recall 低**
增加 NightOwls 数据量，降低 person 置信度阈值，增加难例挖掘迭代。

**Q: 导出 TFLite 失败**
确保 ultralytics 版本最新：`pip install -U ultralytics`
