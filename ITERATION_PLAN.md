# Night Road Vision App - 迭代修复计划

## 当前状态
- 工作流 `w8npumebn` 正在运行
- 检测框显示问题待修复
- 需要多轮迭代完善

---

## 第1轮：修复检测管道 (核心功能)

### 问题诊断
1. **坐标空间不匹配** - 模型输出(320x512空间) vs 屏幕显示
2. **Letterbox参数未传递** - overlay不知道如何映射坐标
3. **模型输出格式** - (1,300,6) Format A: [x1,y1,x2,y2,conf,cls]

### 修复计划
- [ ] 修复 `InferenceEngine.mapToOriginal()` 确保正确映射
- [ ] 修复 `DetectionOverlay.modelToScreen()` 坐标转换
- [ ] 从 `InferenceResult` 传递 `cameraWidth/Height` 到 overlay
- [ ] 验证 letterbox params 计算正确

---

## 第2轮：多模型支持

### 目标模型
| 模型 | 大小 | 速度 | 精度 | 用途 |
|------|------|------|------|------|
| YOLOv8n | 6.2MB | 快 | 中 | 实时检测 |
| YOLOv8s | 22.5MB | 中 | 高 | 平衡模式 |
| YOLOv8m | 52.3MB | 慢 | 最高 | 精确检测 |
| YOLO26n | 5.1MB | 快 | 中 | 自定义模型 |

### 实现计划
- [ ] 创建 `ModelManager` 类管理模型切换
- [ ] 下载 YOLOv8n/s/m TFLite 模型
- [ ] 添加模型选择UI (底部弹窗)
- [ ] 实现热切换模型（无需重启）

---

## 第3轮：完整参数控制

### 可调参数
1. **置信度阈值** (0.1-0.9) - 过滤低置信度检测
2. **IoU阈值** (0.1-0.9) - NMS重叠过滤
3. **检测模式** - ECO/BALANCED/FINE
4. **帧跳过间隔** - 1/2/3帧
5. **输入分辨率** - 416x256 / 512x320 / 640x384

### UI实现
- [ ] 设置对话框添加滑块控制
- [ ] 实时预览参数效果
- [ ] 保存用户偏好到 SharedPreferences

---

## 第4轮：GPU优化 (小米14专项)

### 小米14硬件
- SoC: 骁龙8 Gen 3
- GPU: Adreno 750
- NPU: Hexagon

### 优化策略
1. **GPU Delegate优化**
   - 允许精度损失 (FP16)
   - 启用量化模型支持
   - 设置推理偏好 (SUSTAINED_SPEED)

2. **NNAPI Delegate**
   - 作为GPU备选
   - 利用Hexagon NPU

3. **回退链**
   - GPU → NNAPI → CPU(4线程)

### 实现计划
- [ ] 研究 Adreno 750 最佳配置
- [ ] 添加 GPU benchmark 模式
- [ ] 实现自动选择最佳后端
- [ ] 添加性能监控面板

---

## 第5轮：UI完善与打磨

### 界面元素
1. **顶部栏**
   - 性能统计 (FPS、延迟、后端)
   - 当前模型名称
   - 检测模式指示器

2. **底部栏**
   - 手电筒开关
   - 声音提醒开关
   - 震动提醒开关
   - 检测物体计数

3. **设置面板**
   - 模型选择
   - 参数调节
   - 关于信息

4. **检测覆盖层**
   - 边界框 + 类别标签
   - 置信度显示
   - 追踪ID

### 样式规范
- Material3 暗色主题
- 夜视绿色 (#00FF41) 强调色
- 圆角卡片设计
- 流畅动画

---

## 验证清单

每轮完成后验证：
- [ ] 编译成功 (0 errors)
- [ ] 无内存泄漏
- [ ] 线程安全
- [ ] 坐标映射正确
- [ ] UI响应流畅
- [ ] 崩溃日志正常

---

## 文件结构

```
android-app/
├── app/src/main/
│   ├── assets/models/
│   │   ├── yolo26n_float16.tflite
│   │   ├── yolov8n.tflite
│   │   ├── yolov8s.tflite
│   │   └── yolov8m.tflite
│   ├── java/com/nightroadvision/app/
│   │   ├── inference/
│   │   │   └── InferenceEngine.kt
│   │   ├── model/
│   │   │   └── ModelManager.kt
│   │   ├── ui/
│   │   │   ├── overlay/
│   │   │   │   └── DetectionOverlay.kt
│   │   │   └── screen/
│   │   │       └── MainScreen.kt
│   │   └── camera/
│   │       └── CameraManager.kt
│   └── res/
│       ├── drawable/
│       └── values/
```
