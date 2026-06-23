# Spectral Calculation - 光谱重建算法项目

**版本**: v1.4
**作者**: UNS-JeromeWei  
**日期**: 2026-06-23

---

## 项目概述

本项目是一个光谱重建算法的实现，主要用于计算光谱成像系统中的光谱重建问题。核心功能是通过响应矩阵和透射曲线，利用优化算法重建原始光谱分布。

---

## 项目结构

```
spectral calculation/
│
├── README.md                           # 项目说明文档
├── spectral_cal/                       # 核心算法目录
│   ├── rebuild_curves_cwl_fwhm_loop.py           # 主算法实现文件
│   ├── rebuild_curves_cwl_fwhm_loop_20260503.py  # 算法版本备份
│   ├── rebuild_curves_cwl_fwhm_loop_analysis.md  # 算法详细解析文档
│   ├── roi_reflectance_eval.py                   # ENVI raw ROI 信号/CWL/FWHM GUI 工具
│   ├── roi_reflectance_eval.pyw                  # 无控制台窗口 GUI 入口
│   ├── algorithm_changes_report.html             # ROI 信号处理算法变化说明
│   ├── tmp/                                      # 临时测试文件目录
│   │   ├── rebuild_curves_2_peak_new_uc450_260417.py
│   │   ├── rebuild_curves_2_peak_400_1000nm.py
│   │   └── rebuild_curves_2_peak_400_700nm.py
│   ├── data analysis/                            # 数据分析结果目录
│   │   ├── 20260503_232523-...                   # 分析结果示例
│   │   ├── 20260503_232153-...
│   │   ├── 20260503_170351-...
│   │   ├── 20260503_164126-...
│   │   └── ...
│   ├── .kilo/                                    # Kilo配置目录
│   │   └── agent-manager.json
│   └── .vscode/                                  # VSCode配置
│       └── settings.json
│
├── data analysis/                       # 全局数据分析结果目录
│   ├── 20260430_112954-...              # UDE450 MEMS分析结果
│   ├── 20260416143643-...               # U500 MEMS Drift分析结果
│   └── ...
│
└── .kilo/                               # Kilo项目配置
    ├── command/
    ├── agent/
    └ kilo.json
    └ AGENTS.md
```

---

## 核心功能

### 1. 光谱重建算法

**文件**: `spectral_cal/rebuild_curves_cwl_fwhm_loop.py`

**主要功能**:
- 响应矩阵加载与处理
- 光谱重建计算（最小二乘法、正则化优化）
- FWHM（半高全宽）计算与分析
- CWL（中心波长）循环测试
- 结果可视化与保存

**核心算法**:
- 非负最小二乘法 (NNLS)
- 增广正则化最小二乘法
- PyTorch优化器方法
- 二阶差分平滑约束

### 2. ROI 信号评估 GUI

**文件**: `spectral_cal/roi_reflectance_eval.py`  
**无黑框入口**: `spectral_cal/roi_reflectance_eval.pyw`

**主要功能**:
- 读取 ENVI `.hdr/.raw` 数据目录，自动识别 `*-Rec`、`*-Ref`、`*-Dark` 数据组
- 支持三种信号处理模式，并可在 GUI 左侧按钮中实时切换：
  - `(Sign-Dark)/(Ref-Dark)`
  - `Sign/Ref`
  - `Sign`，默认模式
- 支持手动画多个矩形 ROI，并支持 `Auto 5x5` 自动 ROI 划分
- 左侧显示 CWL drift map：每个像素取当前处理模式下光谱峰值对应的波长
- 右侧显示 ROI 光谱曲线，并按当前处理模式计算 CWL、FWHM、峰值置信度、次峰比例
- CSV/PNG 导出 ROI 光谱、质量状态和峰值诊断指标
- GUI 后台计算 CWL drift map，避免加载大数据时窗口无响应
- 自动配置中文字体，避免中文路径在 GUI/PNG 中乱码

**信噪比增强策略**:
- ROI 曲线使用 trimmed mean，默认去除 ROI 内最高/最低各 5% 像素后再平均
- 每个 ROI 输出 `peak_confidence`、`peak_prominence`、`secondary_peak_ratio`
- 当次峰接近主峰时标记为 `ambiguous_peak`，避免误判不稳定 CWL
- CWL drift map 使用 3x3 中值滤波，并自适应保留当前数据主导峰邻域
- Sign-only CWL map 使用快速 argmax 路径，典型加载速度约提升 5 倍
- CSV 导出增加 `processing_mode` 和 `processing_formula`，用于记录本次 ROI 曲线采用的处理口径

**运行示例**:
```bash
python spectral_cal/roi_reflectance_eval.py
```

Windows 下如需避免控制台黑框，可直接运行:
```bash
pythonw spectral_cal/roi_reflectance_eval.pyw
```

批处理自动 5x5 ROI:
```bash
python spectral_cal/roi_reflectance_eval.py --no-gui --auto-grid --mode sign --output roi_signal_stats.csv
```

可选 `--mode` 参数:
- `dark_ratio`: `(Sign-Dark)/(Ref-Dark)`
- `sign_ref`: `Sign/Ref`
- `sign`: `Sign`

### 3. 数学模型

#### 线性模型
```
y = A · x
```
其中:
- `y`: 测量的透射曲线（通道响应）
- `A`: 响应矩阵（每个通道对不同波长的响应）
- `x`: 待重建的原始光谱

#### 正则化优化
```
min_x ||Ax - y||² + λ||D₂x||²,  x ≥ 0
```
其中:
- `λ`: 正则化参数，控制平滑约束强度
- `D₂`: 二阶差分矩阵，用于平滑约束

---

## 主要函数说明

### 核心重建函数

#### `lst_with_aug_reg_cwl_fwhm_loop()`
**功能**: CWL和FWHM双重循环的光谱重建  
**参数**:
- `src_dir`: 数据目录路径
- `cwl_range`: CWL波长范围 (起始, 结束, 步长)
- `fwhm_range`: FWHM范围 (起始, 结束, 步长)
- `amplitude`: 高斯峰幅度
- `lam`: 正则化参数
- `save_results`: 是否保存结果

**返回**: 包含所有循环结果的字典列表

#### `lst_with_aug_reg_cwl_loop()`
**功能**: 仅CWL循环的光谱重建（FWHM固定）

#### `lst_with_aug_reg_fwhm_loop()`
**功能**: 仅FWHM循环的光谱重建（CWL固定）

### 辅助函数

#### `gaussian_beam(x, mu, sig)`
**功能**: 高斯光束模型  
**公式**: `G(x, μ, σ) = exp(-(x - μ)² / (2σ²))`

#### `find_fwhm_normal(in_x, in_y)`
**功能**: 计算光谱峰的FWHM  
**方法**: 三次样条插值 + 半高位置查找

#### `build_D2(L)`
**功能**: 构建二阶差分矩阵  
**用途**: 用于正则化平滑约束

#### `load_matrix_from_file(src_dir)`
**功能**: 从文件加载波长和透过率数据  
**处理**: 波长范围筛选 (350-950nm)，透过率归一化

---

## 数据分析结果

### 结果文件说明

每个分析结果目录包含:

1. **spectrum_results.csv**: CSV格式的汇总结果
   - Input_CWL_nm: 输入中心波长
   - Output_CWL_nm: 输出中心波长
   - Input_FWHM_nm: 输入FWHM
   - Output_FWHM_nm: 输出FWHM
   - MSE: 重建误差

2. **rebuild_summary.npy**: Numpy格式的完整结果数据

3. **response_matrix.png**: 响应矩阵热图

4. **FWHM={value}/**: 不同FWHM值的重建对比图
   - cwl_{value}_fwhm_{value}.png: 光谱对比图

---

## 使用示例

### 示例1: CWL和FWHM双重循环
```python
results = lst_with_aug_reg_cwl_fwhm_loop(
    data_dir,
    cwl_range=(440, 800, 40),   # CWL: 440nm到800nm，步长40nm
    fwhm_range=(10, 30, 5),      # FWHM: 10nm到30nm，步长5nm
    amplitude=0.6,
    lam=0.01,
    save_results=True
)
```

### 示例2: 仅CWL循环
```python
results = lst_with_aug_reg_cwl_loop(
    data_dir,
    cwl_range=(440, 650, 10),   # CWL范围
    fwhm=8,                     # 固定FWHM: 8nm
    save_results=True
)
```

### 示例3: 仅FWHM循环
```python
results = lst_with_aug_reg_fwhm_loop(
    data_dir,
    cwl=550,                    # 固定CWL: 550nm
    fwhm_range=(10, 30, 5),
    save_results=True
)
```

---

## 依赖库

```python
numpy              # 数值计算
torch              # PyTorch深度学习框架
scipy              # 科学计算（插值、优化、滤波）
matplotlib         # 可视化
pandas             # 数据处理
spectral           # 光谱数据处理
tkinter            # GUI 文件选择
```

---

## 物理背景

### 光谱成像系统

光谱成像系统通过滤光片阵列将不同波长的光分配到不同通道:
- 每个通道对特定波长范围有响应
- 响应矩阵描述各通道的波长响应特性

### 重建原理

**已知**:
- 响应矩阵 `A`（系统特性）
- 测量值 `y`（各通道输出）

**求解**:
- 原始光谱 `x`（入射光光谱分布）

### 应用场景

- 计算光谱成像
- 光谱仪校准
- 薄膜设计验证
- MEMS可调滤光片评估

---

## 版本历史

### v1.4 (2026-06-23)
- **ROI GUI 增加三种处理模式**
  - 新增 `(Sign-Dark)/(Ref-Dark)`、`Sign/Ref`、`Sign` 三个模式按钮
  - 切换模式后自动清空旧 ROI，并按当前模式重算 CWL drift map
  - 手动画 ROI、`Auto 5x5`、CWL、FWHM、PNG 报告、CSV 导出统一使用当前模式
- **版本可见性增强**
  - GUI 窗口标题显示 `ROI Signal Evaluation v1.4`
  - 命令行运行时打印当前版本号
- **导出可追溯**
  - CSV 新增 `processing_mode` 和 `processing_formula`
  - 支持命令行 `--mode` 参数复现 GUI 的三种处理口径

### v1.3 (2026-06-23)
- **新增 ROI 信号评估 GUI**
  - 新增 `spectral_cal/roi_reflectance_eval.py` 和无黑框入口 `roi_reflectance_eval.pyw`
  - 支持 ENVI `.hdr/.raw` 数据读取，自动识别 `Rec/Ref/Dark` 数据目录
  - 支持多 ROI 手动画框、`Auto 5x5` 自动划分、CSV/PNG 导出
- **Sign-only 信号处理流程**
  - 曲线、CWL drift map、CWL/FWHM 指标直接基于 `Sign/Rec` 信号
  - 保留 Dark 作为噪声门限参考，但不再执行 Dark 扣减或 `Sign/Ref` 归一化
- **CWL/FWHM 与质量诊断**
  - 每个 ROI 自动计算 CWL、FWHM、主峰、次峰比例、峰值置信度
  - 新增 `ambiguous_peak` 状态，用于标记主峰与次峰接近的低可信 ROI
  - CSV 增加 `peak_confidence`、`peak_prominence`、`background_level`、`background_noise` 等诊断字段
- **展示信噪比增强**
  - ROI 曲线使用 trimmed mean，降低坏点和边缘异常点影响
  - CWL drift map 增加空间中值滤波和自适应主导峰窗口
  - 后台计算 CWL drift map，避免 GUI 加载期间无响应
- **工程体验优化**
  - 默认数据目录支持自动解析，兼容中文路径和实际数据目录层级变化
  - GUI 统一主题、按钮布局、中文字体和 CWL/FWHM 汇总信息框
  - 新增 `algorithm_changes_report.html` 记录 ROI 算法变化说明

### v1.2 (2026-05-08)
- **新增连续光谱重建功能**
  - 新增 `reconstruct_continuum_spectra()` 函数，支持从CSV文件加载连续光谱数据进行重建
  - 新增 `load_continuum_spectra_from_csv()` 函数，加载A/B/C三组反射率数据
  - 新增 `save_continuum_spectrum_comparison()` 函数，保存单条光谱对比图
  - 新增 `save_continuum_summary_plot()` 函数，保存3x5子图汇总
- **波长范围处理优化**
  - 自动计算CSV数据与响应矩阵波长范围的交集，避免外推导致的负值问题
  - 使用 `np.clip()` 确保插值后光谱值非负
- **输出结构**
  - 结果保存至 `continuum/A组/`、`continuum/B组/`、`continuum/C组/` 子目录
  - 每组包含14条光谱的独立对比图和汇总统计图
- **重建质量指标**
  - MSE（均方误差）和相关系数作为重建质量评估指标
  - 典型重建精度：MSE < 0.00001，相关系数 > 0.999

### v1.1 (2026-05-08)
- 新增 `rebuild_curves_cwl_fwhm_loop_400_980.py` 文件
- 支持自定义 CWL 列表和 FWHM 列表的灵活循环测试
- 优化波长范围支持 400-980nm 扩展波段
- 简化代码结构，移除 PyTorch 依赖，仅使用 scipy 优化
- 新增 `lst_with_aug_reg_cwl_fwhm_list_loop()` 函数，支持列表式参数输入
- 改进结果可视化，增强 FWHM 和 CWL 对比显示
- 添加波长显示范围自定义功能

### v1.0 (2026-05-06)
- 初始版本发布
- 实现核心光谱重建算法
- 支持 CWL 和 FWHM 双重循环测试
- 完善结果保存与可视化功能
- 添加详细算法解析文档

---

## GitHub仓库

**仓库地址**: https://github.com/UNS-JeromeWei/spectral-calculation.git

---

## 许可证

本项目仅供研究和学习使用。

---

## 联系方式

**作者**: UNS-JeromeWei  
**GitHub**: https://github.com/UNS-JeromeWei
