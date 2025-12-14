# TPU环境一键配置指南

## 快速开始

在新镜像上运行以下命令即可完成所有环境配置：

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
bash setup_tpu_env.sh
```

## 配置脚本说明

### `setup_tpu_env.sh` - 一键配置脚本

这个脚本会自动完成以下操作：

1. ✅ **检查conda环境**
   - 自动查找并初始化conda

2. ✅ **创建conda环境**
   - 环境名称: `sglang-jax`
   - Python版本: 3.12

3. ✅ **安装JAX (TPU版本)**
   - 自动从官方源安装TPU版本的JAX
   - 验证TPU设备连接

4. ✅ **设置SGLang-JAX**
   - 自动克隆SGLang-JAX仓库（如果不存在）
   - 安装SGLang-JAX及其所有依赖

5. ✅ **安装bonsai（可选）**
   - 如果bonsai目录存在，自动安装

6. ✅ **验证环境**
   - 检查JAX和SGLang-JAX是否可以正常导入
   - 验证TPU设备

7. ✅ **创建环境激活脚本**
   - 自动创建 `activate_env.sh` 用于后续快速激活环境

## 使用方式

### 首次配置

```bash
cd /home/gcpuser/AI-sys-groupwork/sglang_jax
bash setup_tpu_env.sh
```

### 后续使用

配置完成后，每次使用前激活环境：

```bash
# 方法1: 使用激活脚本
source activate_env.sh

# 方法2: 手动激活
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sglang-jax
export PYTHONPATH=/home/gcpuser/sky_workdir/sglang-jax/python:$PYTHONPATH
```

### 运行测试

```bash
# 激活环境后运行测试
bash run_test_with_conda.sh 1  # TP=1
bash run_test_with_conda.sh 2  # TP=2
bash run_test_with_conda.sh 4  # TP=4
```

## 配置内容

### 安装的软件包

- **Python**: 3.12.12
- **JAX**: 0.8.1 (TPU版本)
- **SGLang-JAX**: 0.0.2
- **所有依赖**: transformers, safetensors, fastapi, uvicorn等

### 环境变量

配置脚本会自动设置：

- `PYTHONPATH`: 包含SGLang-JAX路径
- `JAX_COMPILATION_CACHE_DIR`: JAX编译缓存目录

### 目录结构

```
/home/gcpuser/
├── sky_workdir/
│   └── sglang-jax/          # SGLang-JAX源码
└── AI-sys-groupwork/
    └── sglang_jax/
        ├── setup_tpu_env.sh  # 一键配置脚本
        ├── activate_env.sh   # 环境激活脚本
        └── gpt-oss-20b/      # 模型文件
```

## 故障排查

### 问题1: Conda未找到

**解决**: 确保已安装miniconda或anaconda，脚本会自动查找。

### 问题2: SGLang-JAX安装失败

**解决**: 检查网络连接，确保可以访问GitHub和PyPI。

### 问题3: TPU设备未检测到

**解决**: 确保在TPU VM上运行，而不是普通VM。

### 问题4: 依赖版本冲突

**解决**: 脚本会自动处理大部分依赖，如有问题可以手动调整。

## 验证配置

运行环境检查脚本：

```bash
source activate_env.sh
bash check_test_env.sh
```

应该看到：
- ✅ Python: 3.12.12
- ✅ JAX: 0.8.1
- ✅ TPU设备数: 4
- ✅ SGLang-JAX: 已安装

## 注意事项

1. **首次运行**: 配置过程可能需要10-20分钟，取决于网络速度
2. **网络要求**: 需要能够访问GitHub和PyPI
3. **TPU环境**: 必须在TPU VM上运行，不能在普通VM上
4. **磁盘空间**: 确保有足够的磁盘空间（至少50GB）

## 更新环境

如果需要更新环境：

```bash
source activate_env.sh
pip install --upgrade jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd /home/gcpuser/sky_workdir/sglang-jax/python
pip install -e . --upgrade
```


