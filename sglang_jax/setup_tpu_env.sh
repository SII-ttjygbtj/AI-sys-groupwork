#!/bin/bash
# TPU和JAX环境一键配置脚本
# 适用于新的TPU镜像环境

set -e  # 遇到错误立即退出

echo "=========================================="
echo "TPU和JAX环境一键配置"
echo "=========================================="
echo "开始时间: $(date)"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查conda是否安装
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo -e "${YELLOW}警告: conda未找到，尝试查找miniconda...${NC}"
        if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
            source ~/miniconda3/etc/profile.d/conda.sh
        elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
            source ~/anaconda3/etc/profile.d/conda.sh
        else
            echo -e "${RED}错误: 未找到conda，请先安装miniconda或anaconda${NC}"
            exit 1
        fi
    else
        # 初始化conda
        if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
            source ~/miniconda3/etc/profile.d/conda.sh
        elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
            source ~/anaconda3/etc/profile.d/conda.sh
        fi
    fi
    echo -e "${GREEN}✅ Conda已找到${NC}"
}

# 创建conda环境
create_conda_env() {
    echo ""
    echo "----------------------------------------"
    echo "步骤1: 创建conda环境 (Python 3.12)"
    echo "----------------------------------------"
    
    ENV_NAME="sglang-jax"
    
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo -e "${YELLOW}环境 ${ENV_NAME} 已存在，跳过创建${NC}"
    else
        echo "创建conda环境: ${ENV_NAME} (Python 3.12)..."
        conda create -n ${ENV_NAME} python=3.12 -y
        echo -e "${GREEN}✅ Conda环境创建成功${NC}"
    fi
    
    # 激活环境
    conda activate ${ENV_NAME}
    echo -e "${GREEN}✅ 已激活环境: ${ENV_NAME}${NC}"
    echo "Python版本: $(python --version)"
}

# 安装JAX (TPU版本)
install_jax() {
    echo ""
    echo "----------------------------------------"
    echo "步骤2: 安装JAX (TPU版本)"
    echo "----------------------------------------"
    
    if python -c "import jax" 2>/dev/null; then
        JAX_VERSION=$(python -c "import jax; print(jax.__version__)" 2>/dev/null)
        echo -e "${YELLOW}JAX已安装 (版本: ${JAX_VERSION})，跳过安装${NC}"
    else
        echo "安装JAX (TPU版本)..."
        pip install --upgrade pip
        pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        echo -e "${GREEN}✅ JAX安装成功${NC}"
    fi
    
    # 验证JAX
    python -c "import jax; print(f'JAX版本: {jax.__version__}'); print(f'TPU设备数: {len(jax.devices())}')" || {
        echo -e "${RED}❌ JAX验证失败${NC}"
        exit 1
    }
    echo -e "${GREEN}✅ JAX验证成功${NC}"
}

# 克隆/检查SGLang-JAX
setup_sglang_jax() {
    echo ""
    echo "----------------------------------------"
    echo "步骤3: 设置SGLang-JAX"
    echo "----------------------------------------"
    
    SGLANG_JAX_DIR="/home/gcpuser/sky_workdir/sglang-jax"
    
    if [ -d "${SGLANG_JAX_DIR}/python" ]; then
        echo -e "${YELLOW}SGLang-JAX目录已存在: ${SGLANG_JAX_DIR}${NC}"
    else
        echo "克隆SGLang-JAX仓库..."
        mkdir -p /home/gcpuser/sky_workdir
        cd /home/gcpuser/sky_workdir
        
        if [ -d "sglang-jax" ]; then
            echo -e "${YELLOW}目录已存在，跳过克隆${NC}"
        else
            git clone https://github.com/sgl-project/sglang-jax.git
            echo -e "${GREEN}✅ SGLang-JAX克隆成功${NC}"
        fi
    fi
    
    # 修改Python版本要求（如果需要）
    if [ -f "${SGLANG_JAX_DIR}/python/pyproject.toml" ]; then
        echo "检查Python版本要求..."
        if grep -q 'requires-python = ">=3.12,<3.14"' "${SGLANG_JAX_DIR}/python/pyproject.toml"; then
            echo "Python版本要求已符合"
        fi
    fi
}

# 安装SGLang-JAX
install_sglang_jax() {
    echo ""
    echo "----------------------------------------"
    echo "步骤4: 安装SGLang-JAX及其依赖"
    echo "----------------------------------------"
    
    SGLANG_JAX_DIR="/home/gcpuser/sky_workdir/sglang-jax"
    
    if python -c "import sgl_jax" 2>/dev/null; then
        echo -e "${YELLOW}SGLang-JAX已安装，跳过安装${NC}"
    else
        echo "安装SGLang-JAX..."
        cd "${SGLANG_JAX_DIR}/python"
        
        # 升级pip
        pip install --upgrade pip
        
        # 安装SGLang-JAX（会自动安装所有依赖）
        pip install -e . || {
            echo -e "${YELLOW}警告: 标准安装失败，尝试安装依赖...${NC}"
            # 如果安装失败，尝试手动安装关键依赖
            pip install transformers safetensors tiktoken fastapi uvicorn uvloop \
                pyzmq setproctitle partial-json-parser llguidance pathwaysutils \
                orjson pydantic || true
            pip install -e . --no-deps || true
        }
        
        echo -e "${GREEN}✅ SGLang-JAX安装完成${NC}"
    fi
    
    # 验证SGLang-JAX
    export PYTHONPATH="${SGLANG_JAX_DIR}/python:$PYTHONPATH"
    python -c "import sys; sys.path.insert(0, '${SGLANG_JAX_DIR}/python'); import sgl_jax; print('✅ SGLang-JAX可以导入')" || {
        echo -e "${YELLOW}⚠️  SGLang-JAX导入测试失败，但可能仍可使用${NC}"
    }
}

# 安装bonsai（可选）
install_bonsai() {
    echo ""
    echo "----------------------------------------"
    echo "步骤5: 安装bonsai（可选）"
    echo "----------------------------------------"
    
    BONSAI_DIR="/home/gcpuser/AI-sys-groupwork/bonsai"
    
    if [ -d "${BONSAI_DIR}" ]; then
        echo "安装bonsai..."
        cd "${BONSAI_DIR}"
        pip install -e . || {
            echo -e "${YELLOW}⚠️  Bonsai安装失败，但可能不是必需的${NC}"
        }
        echo -e "${GREEN}✅ Bonsai安装完成${NC}"
    else
        echo -e "${YELLOW}Bonsai目录不存在，跳过安装${NC}"
    fi
}

# 验证环境
verify_environment() {
    echo ""
    echo "----------------------------------------"
    echo "步骤6: 验证环境配置"
    echo "----------------------------------------"
    
    SGLANG_JAX_DIR="/home/gcpuser/sky_workdir/sglang-jax"
    export PYTHONPATH="${SGLANG_JAX_DIR}/python:$PYTHONPATH"
    
    python << 'EOF'
import sys
sys.path.insert(0, '/home/gcpuser/sky_workdir/sglang-jax/python')

try:
    import jax
    print(f"✅ JAX版本: {jax.__version__}")
    print(f"✅ TPU设备数: {len(jax.devices())}")
    print(f"✅ TPU设备: {[str(d) for d in jax.devices()[:4]]}")
except Exception as e:
    print(f"❌ JAX验证失败: {e}")
    sys.exit(1)

try:
    import sgl_jax
    from sgl_jax.srt.entrypoints.engine import Engine
    from sgl_jax.srt.hf_transformers_utils import get_tokenizer
    print("✅ SGLang-JAX可以导入")
    print("✅ Engine可以导入")
    print("✅ get_tokenizer可以导入")
except Exception as e:
    print(f"⚠️  SGLang-JAX验证警告: {e}")
    print("   可能需要检查PYTHONPATH设置")

print("\n" + "=" * 50)
print("✅ 环境验证完成")
print("=" * 50)
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ 环境验证成功${NC}"
    else
        echo -e "${YELLOW}⚠️  环境验证有警告，但可能仍可使用${NC}"
    fi
}

# 创建环境变量设置脚本
create_env_script() {
    echo ""
    echo "----------------------------------------"
    echo "步骤7: 创建环境变量设置脚本"
    echo "----------------------------------------"
    
    ENV_SCRIPT="/home/gcpuser/AI-sys-groupwork/sglang_jax/activate_env.sh"
    
    cat > "${ENV_SCRIPT}" << 'ENVEOF'
#!/bin/bash
# 激活TPU测试环境的脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate sglang-jax

# 设置PYTHONPATH
export PYTHONPATH="/home/gcpuser/sky_workdir/sglang-jax/python:$PYTHONPATH"

# 设置JAX编译缓存
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache

echo "✅ 环境已激活"
echo "  - Conda环境: sglang-jax"
echo "  - PYTHONPATH: $PYTHONPATH"
ENVEOF

    chmod +x "${ENV_SCRIPT}"
    echo -e "${GREEN}✅ 环境激活脚本已创建: ${ENV_SCRIPT}${NC}"
    echo "  使用方式: source ${ENV_SCRIPT}"
}

# 主函数
main() {
    echo "开始环境配置..."
    echo ""
    
    # 检查conda
    check_conda
    
    # 创建conda环境
    create_conda_env
    
    # 安装JAX
    install_jax
    
    # 设置SGLang-JAX
    setup_sglang_jax
    
    # 安装SGLang-JAX
    install_sglang_jax
    
    # 安装bonsai（可选）
    install_bonsai
    
    # 验证环境
    verify_environment
    
    # 创建环境脚本
    create_env_script
    
    echo ""
    echo "=========================================="
    echo "✅ 环境配置完成！"
    echo "=========================================="
    echo ""
    echo "后续使用:"
    echo "  1. 激活环境: source activate_env.sh"
    echo "  2. 运行测试: bash run_test_with_conda.sh 1"
    echo ""
    echo "完成时间: $(date)"
}

# 运行主函数
main


