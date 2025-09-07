# GPU 加速指南

本指南介绍如何在本项目中启用并验证 GPU 加速能力。

## 环境变量与配置项

- 环境变量 `GPU_ACCELERATION` 用于全局开关 GPU 加速；在代码中会同时读取 `config.json` 的 `gpu_acceleration` 字段作为默认值或覆盖项，两者任意一处为 true 即会开启。示例：

```bash
GPU_ACCELERATION=true
```

- `GPU_TYPE` 指定 GPU 类型，支持：`nvidia`、`amd`、`intel`、`auto`（默认）。
- `config.json` 示例（与环境变量含义一致）：

```json
{
  "gpu_acceleration": true,
  "gpu_type": "auto"
}
```

> 读取优先级：若设置了环境变量，将优先生效；否则回退到配置文件中的值。

## 常见问题

- 如果部署环境未安装对应驱动或运行时（如 CUDA/ROCm/oneAPI），即使开启也可能回退到 CPU。
- 部分预处理步骤（如 ffmpeg 硬件加速）需要系统已正确识别 GPU 设备。
- 当 `GPU_ACCELERATION=false` 或 `gpu_acceleration: false` 时，强制使用 CPU 模式。

## 验证方式

- 查看启动日志，确认已检测到 GPU 并启用相应加速路径。
- 运行集成测试，对比 CPU 与 GPU 模式下的性能差异（见性能基准章节）。