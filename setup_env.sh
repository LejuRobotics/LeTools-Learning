#!/bin/bash

# 设置遇到错误立即停止执行
set -e
echo "==================================================="
echo "🚀 欢迎使用 KDC 项目环境配置脚本! 先进行pip换源"
echo "==================================================="
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple  # 建议首先换源，能加快下载安装速度


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
        
        # 将 pip install 放在 if 中，如果成功返回 0，失败返回非 0
        if pip install -r requirements_ros_env.txt; then
            echo "✅ ROS 环境依赖检查/安装完成！"
        else
            # pip 报错时会进入这里
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
    pip install -r requirements.txt
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
    echo "⚠️  third_party/lerobot 目录为空或子模块未初始化（未找到 pyproject.toml）。"
    echo ""
    printf "是否现在自动执行 git submodule update --init --recursive 来拉取？[y/N] "
    read -r _ans
    case "$_ans" in
        [Yy]|[Yy][Ee][Ss])
            echo "正在拉取子模块..."
            git -c url."https://gh-proxy.com/https://github.com/".insteadOf="https://github.com/" \
                submodule update --init --recursive --jobs 8
            if [ ! -f "third_party/lerobot/pyproject.toml" ]; then
                echo "❌ 子模块拉取后仍未找到 third_party/lerobot/pyproject.toml，请检查网络或手动执行："
                echo "   git submodule update --init --recursive"
                exit 1
            fi
            echo "✅ 子模块拉取完成！"
            ;;
        *)
            echo "❌ 已跳过。请手动执行以下命令后重新运行此脚本："
            echo "   git submodule update --init --recursive"
            exit 1
            ;;
    esac
fi

python -m pip install -e "third_party/lerobot[training,dataset]"
echo "✅ lerobot项目基础依赖（含 training, dataset）安装完成！"

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
                python -m pip install -e "third_party/lerobot[diffusion]"
                echo "✅ diffusion 依赖安装完成！"
                ;;
            pi0 | pi0_fast | pi0fast | pi05)
                echo "📦 安装 pi 系列依赖（transformers + scipy）及 peft（用于 LoRA 微调）..."
                python -m pip install -e "third_party/lerobot[pi,peft]"
                echo "✅ pi 系列依赖安装完成！"
                ;;
            gr00t | groot)
                echo "📦 安装 gr00t 依赖（transformers + peft + diffusers + dm-tree + timm + decord + ninja）..."
                echo "⚠️  注意：gr00t 还需要 flash-attn，将在第 5 步中单独安装。"
                python -m pip install -e "third_party/lerobot[groot]"
                echo "✅ gr00t 依赖安装完成（flash-attn 请在第 5 步安装）！"
                ;;
            wall_x | wall-x | wallx)
                echo "📦 安装 wall_x 依赖（transformers + peft + scipy + torchdiffeq + qwen-vl-utils）..."
                python -m pip install -e "third_party/lerobot[wallx]"
                echo "✅ wall_x 依赖安装完成！"
                ;;
            multi_task_dit | multi-task-dit)
                echo "📦 安装 multi_task_dit 依赖（transformers + diffusers）..."
                python -m pip install -e "third_party/lerobot[multi_task_dit]"
                echo "✅ multi_task_dit 依赖安装完成！"
                ;;
            smolvla)
                echo "📦 安装 smolvla 依赖（transformers + num2words + accelerate）..."
                python -m pip install -e "third_party/lerobot[smolvla]"
                echo "✅ smolvla 依赖安装完成！"
                ;;
            xvla)
                echo "📦 安装 xvla 依赖（transformers）..."
                python -m pip install -e "third_party/lerobot[xvla]"
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
python -m pip install -e "third_party/lerobot[training,dataset]"
echo "✅ lerobot项目基础依赖（含 training, dataset）安装完成！"

# echo ""
# echo "==================================================="
# echo "👉 第 3 步：运行全局依赖冲突检查"
# echo "==================================================="
# # pip check 会检查当前环境中安装的所有包是否存在版本不兼容的问题
# if pip check; then
#     echo "🎉 恭喜！所有依赖均已安装且没有检测到版本冲突！"
# else
#     echo "⚠️ 注意：pip check 检测到了一些版本冲突，请根据上面的提示核对。"
# fi


echo ""
echo "==================================================="
echo "👉 第 4 步：安装特定版本的 ffmpeg 和 pyarrow 以及 pyaudio"
echo "==================================================="
conda install ffmpeg=6.1.1 -y
pip uninstall pyarrow -y
pip install pyarrow==21.0.0
conda install pyaudio -y

echo ""
echo "==================================================="
echo "👉 第 5 步: 安装 Gr00t模型（WALL-X和XVLA会条件import flash-attn,未安装不会影响使用）所需要的flash-attn,请先确认nvcc -V cuda版本大于11.7, 如需升级请访问https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_loca"
echo "==================================================="

while true; do
    read -r -p "是否安装 flash-attn？安装请输入 Y，跳过请输入 N: " INSTALL_FLASH_ATTN
    case "$INSTALL_FLASH_ATTN" in
        [Yy])
            if [ ! -d "flash_attn-2.8.3" ]; then
                echo "未检测到 flash_attn-2.8.3 文件夹，开始下载并解压..."
                wget https://files.pythonhosted.org/packages/3b/b2/8d76c41ad7974ee264754709c22963447f7f8134613fd9ce80984ed0dab7/flash_attn-2.8.3.tar.gz
                tar -zxvf flash_attn-2.8.3.tar.gz
            else
                echo "文件夹 flash_attn-2.8.3 已存在，跳过下载和解压。"
            fi

            # 尝试在 Python 中导入 flash_attn，并将输出和错误信息丢弃 (&> /dev/null)
            if python -c "import flash_attn" &> /dev/null; then
                echo "检测到 flash_attn 已安装，跳过编译。"
            else
                echo "未检测到 flash_attn，准备开始编译安装..."

                # 进入目录，如果目录不存在则报错并退出
                cd flash_attn-2.8.3/ || { echo "错误: 找不到 flash_attn-2.8.3/ 目录"; exit 1; }

                # 使用 MAX_JOBS=4 限制编译核心数，防止内存溢出 (OOM)
                # if [ls "dist/*.whl" 2> /dev/null | grep -q .]; then
                #     echo "正在使用 MAX_JOBS=4 编译 flash-attn，这可能需要一些时间..."
                #     MAX_JOBS=4 pytdhon setup.py bdist_wheel
                # fi
                echo "Installing flash-attn from flash_attn-2.8.3/*.whl ..."
                pip install dist/*.whl || { echo "正在使用 MAX_JOBS=4 编译 flash-attn，这可能需要一些时间..."; MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py bdist_wheel; pip install dist/*.whl; }

                # 返回上级目录
                cd ../

                echo "flash_attn 安装流程执行完毕。"
            fi
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
source "$BASHRC_FILE"  # 立即生效配置
