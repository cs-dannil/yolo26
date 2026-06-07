# 失败案例分析（≥10 例）

绿色框为真值(GT)，彩色框为模型预测。

## 案例 1：漏检（场景：small）
![case1](outputs/failure_cases/case_01_漏检.jpg)
- **原因分析**：目标框 w/h 均 < 32px，特征下采样后信息丢失，置信度低于阈值被过滤。
- **改进方案**：提高输入分辨率(1280)、加入小目标数据增强(copy-paste/Mosaic)、使用 P2 检测头。

## 案例 2：漏检（场景：blur）
![case2](outputs/failure_cases/case_02_漏检.jpg)
- **原因分析**：运动/失焦模糊导致边缘特征退化，模型置信度下降造成漏检。
- **改进方案**：训练时加入高斯/运动模糊增强，提升模型对模糊样本的鲁棒性。

## 案例 3：误检（场景：normal）
![case3](outputs/failure_cases/case_03_误检.jpg)
- **原因分析**：背景中类似圆形/红色区域被误判为禁令标志，产生假阳性。
- **改进方案**：提高 conf 阈值、加入难负样本(hard negative)、增大 NMS 抑制。

## 案例 4：误检（场景：occlusion）
![case4](outputs/failure_cases/case_04_误检.jpg)
- **原因分析**：被遮挡标志残缺，模型在残缺区域生成额外冗余框。
- **改进方案**：加入遮挡增强(random erasing)，使用 Soft-NMS 抑制重叠框。

## 案例 5：类别错误（场景：normal）
![case5](outputs/failure_cases/case_05_类别错误.jpg)
- **原因分析**：限速类(pl*)数字相近，细粒度区分能力不足导致类别混淆。
- **改进方案**：引入更高分辨率 ROI、增加细粒度对比损失、平衡各类样本数量。

## 案例 6：类别错误（场景：backlight）
![case6](outputs/failure_cases/case_06_类别错误.jpg)
- **原因分析**：逆光下颜色信息失真，红/蓝标志色彩区分度降低导致误分类。
- **改进方案**：做白平衡/色彩抖动增强、引入对色彩不敏感的形状特征。

## 案例 7：小目标失败（场景：small）
![case7](outputs/failure_cases/case_07_小目标失败.jpg)
- **原因分析**：远处标志成像极小，anchor 与特征图分辨率不匹配。
- **改进方案**：使用切片推理(SAHI)、多尺度测试(TTA)、增大训练分辨率。

## 案例 8：小目标失败（场景：small）
![case8](outputs/failure_cases/case_08_小目标失败.jpg)
- **原因分析**：小目标密集时 NMS 误抑制相邻正确框。
- **改进方案**：降低 NMS IoU 阈值、采用 Soft-NMS、提升特征金字塔分辨率。

## 案例 9：夜间场景（场景：night）
![case9](outputs/failure_cases/case_09_夜间场景.jpg)
- **原因分析**：低照度下信噪比低，标志亮度不足导致漏检/低置信。
- **改进方案**：加入夜间/低光增强、Gamma 校正预处理、必要时引入红外或图像增亮。

## 案例 10：逆光场景（场景：backlight）
![case10](outputs/failure_cases/case_10_逆光场景.jpg)
- **原因分析**：强光晕染使标志过曝，纹理与数字被淹没。
- **改进方案**：HDR/曝光增强训练、加入逆光样本、使用自适应直方图均衡(CLAHE)。

## 案例 11：模糊场景（场景：blur）
![case11](outputs/failure_cases/case_11_模糊场景.jpg)
- **原因分析**：图像整体模糊，高频细节缺失影响定位精度。
- **改进方案**：去模糊预处理(DeblurGAN)、模糊数据增强、提高骨干网络感受野。

## 案例 12：遮挡场景（场景：occlusion）
![case12](outputs/failure_cases/case_12_遮挡场景.jpg)
- **原因分析**：前景物体遮挡标志主体，可见特征不足。
- **改进方案**：遮挡增强、部件级特征学习、结合时序信息(视频多帧融合)。
