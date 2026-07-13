#!/usr/bin/env bash

# 设置遇到错误立即停止执行
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/.venv}"
PYTHON_BIN="$VENV_DIR/bin/python"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

if ! command -v uv >/dev/null 2>&1; then
    echo "❌ 错误：未找到 uv，请先安装 uv 后重新运行此脚本。"
    echo "   安装说明：https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "==================================================="
echo "🚀 欢迎使用 KDC 项目环境配置脚本！"
echo "==================================================="
echo "uv: $(uv --version)"
echo "Python: $PYTHON_VERSION"
echo "虚拟环境: $VENV_DIR"
echo "Python 包索引: $UV_DEFAULT_INDEX"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "⏳ 正在使用 uv 创建 Python $PYTHON_VERSION 虚拟环境..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
else
    _venv_python_version="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [ "$_venv_python_version" != "$PYTHON_VERSION" ]; then
        echo "❌ 错误：现有虚拟环境使用 Python $_venv_python_version，但要求 Python $PYTHON_VERSION。"
        echo "   请删除 $VENV_DIR 后重试，或通过 VENV_DIR 指定另一个目录。"
        exit 1
    fi
    echo "✅ 检测到现有 uv 虚拟环境，继续复用。"
fi

# 激活仅对当前脚本有效；安装命令仍显式指定解释器，避免包落入系统 Python。
source "$VENV_DIR/bin/activate"

uv_install() {
    uv pip install --python "$PYTHON_BIN" "$@"
}


echo "==================================================="
echo "👉 第 1 步：检查并安装 ROS 环境依赖 (requirements_ros_env.txt)"
echo "==================================================="

# 提示用户输入，并将输入结果存入变量 INSTALL_ROS
read -p "是否需要检查 ROS 环境依赖？[Y/n] (默认: Y): " CHECK_ROS

# 如果用户直接按回车，输入为空，则默认赋值为 "Y"
CHECK_ROS=${CHECK_ROS:-Y}

# 判断用户输入是否为 Y, y 或者 yes
if [[ "$CHECK_ROS" == "Y" || "$CHECK_ROS" == "y" || "$CHECK_ROS" == "yes" || "$CHECK_ROS" == "Yes" ]]; then
    # 检查文件是否存在（虽然肯定存在，但保留检查是个好习惯，防患于未然）
    if [ -f "requirements_ros_env.txt" ]; then
        echo "⏳ 正在检查并安装 ROS 环境依赖..."
        
        if uv_install -r requirements_ros_env.txt; then
            echo "✅ ROS 环境依赖检查/安装完成！"
        else
            # uv 报错时会进入这里
            echo "❌ 错误：ros依赖库不全，请仔细核对ROS是否安装好"
            # 退出脚本，防止在缺少依赖的情况下继续执行后续代码
            exit 1
        fi
    else
        echo "❌ 错误：未找到 requirements_ros_env.txt 文件，请确认它与此脚本在同一目录下！"
        exit 1
    fi
else
    # 如果用户输入 n、N 或其他字符
    echo "⏭️  已跳过 ROS 环境依赖的检查与安装。"
fi


echo ""
echo "==================================================="
echo "👉 第 2 步：安装主项目依赖 (requirements.txt) 和 lerobot基础依赖 (third_party/lerobot/pyproject.toml)"
echo "==================================================="

if [ -f "requirements.txt" ]; then
    uv_install -r requirements.txt
    echo "✅ 主项目依赖安装完成！"
else
    echo "❌ 错误：未找到 requirements.txt 文件，请确认它与此脚本在同一目录下！"
    exit 1
fi

_lerobot_empty=false
if [ ! -f "third_party/lerobot/pyproject.toml" ]; then
    _lerobot_empty=true
fi

if $_lerobot_empty; then
    _LEROBOT_COMMIT="a07f22e22ce88cddff1f6eddced9ea008fbfc37c"
    _LEROBOT_CMD='git -c url."https://gh-proxy.com/https://github.com/".insteadOf="https://github.com/" submodule update --init --recursive --jobs 8 && git -C third_party/lerobot checkout '"${_LEROBOT_COMMIT}"

    echo "⚠️  third_party/lerobot 目录为空或子模块未初始化（未找到 pyproject.toml）。"
    echo ""
    printf "是否现在自动执行 git submodule update --init --recursive 来拉取并切换到指定 commit: ${_LEROBOT_COMMIT}？[y/N] "
    read -r _ans
    case "$_ans" in
        [Yy]|[Yy][Ee][Ss])
            echo "正在拉取子模块并切换到指定 commit: ${_LEROBOT_COMMIT} ..."
            git -c url."https://gh-proxy.com/https://github.com/".insteadOf="https://github.com/" \
                submodule update --init --recursive --jobs 8 && git -C third_party/lerobot checkout "${_LEROBOT_COMMIT}"
            if [ ! -f "third_party/lerobot/pyproject.toml" ]; then
                echo "❌ 子模块拉取后仍未找到 third_party/lerobot/pyproject.toml，请检查网络或手动执行："
                echo "   ${_LEROBOT_CMD}"
                exit 1
            fi
            echo "✅ 子模块拉取并切换 commit hash: ${_LEROBOT_COMMIT} 完成！"
            ;;
        *)
            echo "❌ 已跳过。请手动执行以下命令后重新运行此脚本："
            echo "   ${_LEROBOT_CMD}"
            exit 1
            ;;
    esac
fi

uv_install -e "third_party/lerobot[training,dataset]"
echo "✅ lerobot项目基础依赖（含 training, dataset）安装完成！"


# New: Flash-attn installation moved here
install_flash_attn() {
    if [ ! -d "flash_attn-2.8.3" ]; then
        echo "未检测到 flash_attn-2.8.3 文件夹，开始下载并解压..."
        #wget https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
        # Tsinghua tuna server may be faster
        wget https://pypi.tuna.tsinghua.edu.cn/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
        tar -zxvf flash_attn-2.8.3.tar.gz
    else
        echo "文件夹 flash_attn-2.8.3 已存在，跳过下载和解压。"
    fi

    # 尝试在 Python 中导入 flash_attn，并将输出和错误信息丢弃 (&> /dev/null)
    if "$PYTHON_BIN" -c "import flash_attn" &> /dev/null; then
        echo "检测到 flash_attn 已安装，跳过编译。"
    else
        echo "未检测到 flash_attn，准备开始编译安装..."

        (
            cd flash_attn-2.8.3/ || { echo "错误: 找不到 flash_attn-2.8.3/ 目录"; exit 1; }

            echo "Installing flash-attn from flash_attn-2.8.3/dist/*.whl ..."
            if compgen -G "dist/*.whl" >/dev/null && uv_install dist/*.whl; then
                echo "检测到可用 wheel，已直接安装。"
            else
                echo "正在使用 MAX_JOBS=4 编译 flash-attn，这可能需要一些时间..."
                MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE "$PYTHON_BIN" setup.py bdist_wheel
                uv_install dist/*.whl
            fi
        )

        echo "flash_attn 安装流程执行完毕。"
    fi
}


echo ""
echo "==================================================="
echo "👉 第 3 步：安装lerobot模型及训练相关扩展依赖(third_party/lerobot/pyproject.toml)"
echo "==================================================="
echo "可用模型列表（共 10 个）："
echo "  1) act"
echo "  2) diffusion"
echo "  3) gr00t"
echo "  4) multi_task_dit"
echo "  5) pi05"
echo "  6) pi0_fast"
echo "  7) pi0"
echo "  8) smolvla"
echo "  9) wall_x"
echo " 10) xvla"
echo ""
echo "请输入要安装依赖的模型名称（多个用空格分隔，直接回车跳过）："
printf "> "
read -r _model_input

if [ -z "$_model_input" ]; then
    echo "⏭️  已跳过模型专项依赖安装。"
else
    for _model in $_model_input; do
        case "$_model" in
            act)
                echo "📦 act 无额外依赖，第 2 步的基础安装已包含所需全部依赖。"
                echo "✅ act 依赖已就绪！"
                ;;
            diffusion)
                echo "📦 安装 diffusion 依赖（diffusers）..."
                uv_install -e "third_party/lerobot[diffusion]"
                echo "✅ diffusion 依赖安装完成！"
                ;;
            pi0 | pi0_fast | pi0fast | pi05)
                echo "📦 安装 pi 系列依赖（transformers + scipy）及 peft（用于 LoRA 微调）..."
                uv_install -e "third_party/lerobot[pi,peft]"
                echo "✅ pi 系列依赖安装完成！"
                ;;
            gr00t | groot)
                echo "📦 安装 gr00t 依赖（transformers + peft + diffusers + dm-tree + timm + decord + ninja）..."
                echo "⚠️  注意：gr00t 还需要 flash-attn，需要在此安装。是否现在安装 flash-attn？安装请输入 Y，跳过请输入 N: "
                read -r INSTALL_FLASH_ATTN
                case "$INSTALL_FLASH_ATTN" in
                    [Yy])
                        install_flash_attn
                        uv_install -e "third_party/lerobot[groot]"
                        echo "✅ gr00t 依赖安装完成！"
                        ;;
                    *)
                        echo "⏭️  已跳过 flash-attn 安装。"
                        echo "⚠️  注意：gr00t 必须需要 flash-attn，gr00t依赖已经跳过安装。"
                        ;;
                esac
                ;;
            wall_x | wall-x | wallx)
                echo "📦 安装 wall_x 依赖（transformers + peft + scipy + torchdiffeq + qwen-vl-utils）..."
                uv_install -e "third_party/lerobot[wallx]"
                echo "✅ wall_x 依赖安装完成！"
                ;;
            multi_task_dit | multi-task-dit)
                echo "📦 安装 multi_task_dit 依赖（transformers + diffusers）..."
                uv_install -e "third_party/lerobot[multi_task_dit]"
                echo "✅ multi_task_dit 依赖安装完成！"
                ;;
            smolvla)
                echo "📦 安装 smolvla 依赖（transformers + num2words + accelerate）..."
                uv_install -e "third_party/lerobot[smolvla]"
                echo "✅ smolvla 依赖安装完成！"
                ;;
            xvla)
                echo "📦 安装 xvla 依赖（transformers）..."
                uv_install -e "third_party/lerobot[xvla]"
                echo "✅ xvla 依赖安装完成！"
                ;;
            *)
                echo "❌ 未知模型 '$_model'，跳过。支持的模型：act, diffusion, pi0, pi0_fast, pi05, gr00t, wall_x, multi_task_dit, smolvla, xvla"
                ;;
        esac
    done
fi

echo "✅ 所选模型依赖安装流程完成！"

echo "重新安装lerobot项目中。。。"
uv_install -e "third_party/lerobot[training,dataset]"
echo "✅ lerobot项目基础依赖（含 training, dataset）安装完成！"

# echo ""
# echo "==================================================="
# echo "👉 第 3 步：运行全局依赖冲突检查"
# echo "==================================================="
# # uv pip check 会检查当前环境中安装的所有包是否存在版本不兼容的问题
# if uv pip check --python "$PYTHON_BIN"; then
#     echo "🎉 恭喜！所有依赖均已安装且没有检测到版本冲突！"
# else
#     echo "⚠️ 注意：uv pip check 检测到了一些版本冲突，请根据上面的提示核对。"
# fi


echo ""
echo "==================================================="
echo "👉 第 4 步：检查 ffmpeg，并安装 pyarrow 和 pyaudio"
echo "==================================================="
if command -v ffmpeg >/dev/null 2>&1; then
    echo "✅ 检测到系统 ffmpeg: $(ffmpeg -version | head -n 1)"
else
    echo "⚠️ 未检测到 ffmpeg。ffmpeg 是系统程序，不能通过 uv 安装。"
    echo "   Ubuntu 可执行：sudo apt-get update && sudo apt-get install -y ffmpeg"
fi

if uv_install pyarrow==21.0.0 pyaudio; then
    echo "✅ pyarrow 和 pyaudio 安装完成！"
else
    echo "❌ pyaudio 编译失败时，请先安装系统依赖：sudo apt-get install -y portaudio19-dev"
    exit 1
fi

echo ""
echo "==================================================="
echo "👉 第 5 步: 安装 Gr00t模型（WALL-X和XVLA会条件import flash-attn,未安装不会影响使用）所需要的flash-attn,请先确认nvcc -V cuda版本大于11.7, 如需升级请访问https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_loca"
echo "==================================================="

while true; do
    read -r -p "是否安装 flash-attn？安装请输入 Y，跳过请输入 N: " INSTALL_FLASH_ATTN
    case "$INSTALL_FLASH_ATTN" in
        [Yy])
            install_flash_attn
            break
            ;;
        [Nn])
            echo "⏭️  已跳过 flash-attn 安装。"
            break
            ;;
        *)
            echo "请输入 Y 或 N。"
            ;;
    esac
done


echo "==================================================="
echo "👉 第 6 步：检查并配置 Hugging Face 镜像源"
echo "==================================================="
BASHRC_FILE="$HOME/.bashrc"

# 检查 ~/.bashrc 文件是否存在，不存在则创建（兜底防护）
if [ ! -f "$BASHRC_FILE" ]; then
    touch "$BASHRC_FILE"
fi

# 检查是否已经存在该配置
if grep -q "HF_ENDPOINT=https://hf-mirror.com" "$BASHRC_FILE"; then
    echo "✅ Hugging Face 镜像源已配置在 ~/.bashrc 中，无需重复添加。"
else
    echo "⚠️ 未检测到 Hugging Face 镜像源配置，正在添加到 ~/.bashrc..."
    # 写入配置到 bashrc 末尾
    echo "" >> "$BASHRC_FILE"
    echo "# Hugging Face Mirror Endpoint" >> "$BASHRC_FILE"
    echo "export HF_ENDPOINT=https://hf-mirror.com" >> "$BASHRC_FILE"
    
    echo "✅ 镜像源已成功添加至 ~/.bashrc！"
fi
export HF_ENDPOINT=https://hf-mirror.com

echo ""
echo "==================================================="
echo "✅ 环境安装流程完成！"
echo "==================================================="
echo "当前终端请执行以下命令激活环境："
echo "source \"$VENV_DIR/bin/activate\""
