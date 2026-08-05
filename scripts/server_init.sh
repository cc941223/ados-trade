#!/bin/bash
# ADOS-Trade 服务器初始化脚本
# 适用系统：Ubuntu 22.04 LTS
# 用法：SSH 登录服务器后，执行 bash server_init.sh
# 对应技术规格书 M0 阶段

set -e

echo "===== ADOS-Trade 服务器初始化开始 ====="

# 1. 更新系统
echo "[1/6] 更新系统软件包..."
sudo apt update && sudo apt upgrade -y

# 2. 安装 Docker
echo "[2/6] 安装 Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sudo sh
    sudo usermod -aG docker $USER
    echo "Docker 安装完成，需要重新登录 SSH 会话才能免 sudo 使用 docker 命令"
else
    echo "Docker 已安装，跳过"
fi

# 3. 安装 Docker Compose Plugin（新版 docker 自带 compose 子命令，此处确认可用）
echo "[3/6] 验证 Docker Compose..."
docker compose version || echo "警告：docker compose 命令不可用，请检查 Docker 安装"

# 4. 安装 Tailscale
echo "[4/6] 安装 Tailscale..."
if ! command -v tailscale &> /dev/null; then
    curl -fsSL https://tailscale.com/install.sh | sudo sh
    echo "接下来会打开登录链接，请用你的 Tailscale 账号（ccwu941223@gmail.com）登录授权"
    sudo tailscale up
else
    echo "Tailscale 已安装，跳过"
fi

# 5. 克隆项目仓库
echo "[5/6] 克隆 ados-trade 仓库..."
if [ ! -d "ados-trade" ]; then
    echo "请手动执行: git clone https://github.com/<你的用户名>/ados-trade.git"
    echo "（如果仓库是 Private，需要先配置 GitHub 访问凭证或用 SSH key）"
else
    echo "ados-trade 目录已存在，跳过克隆"
fi

# 6. 防火墙基础配置：只放行 SSH，其余端口通过 Tailscale 内网访问
echo "[6/6] 配置防火墙..."
if command -v ufw &> /dev/null; then
    sudo ufw allow OpenSSH
    sudo ufw --force enable
    echo "防火墙已启用，仅放行 SSH（22端口），其余服务通过 Tailscale 内网访问"
else
    echo "未检测到 ufw，请手动确认云服务商控制台的安全组规则：只放行 22 端口"
fi

echo ""
echo "===== 初始化完成 ====="
echo "下一步："
echo "1. 重新登录 SSH（使 docker 用户组生效）"
echo "2. cd ados-trade"
echo "3. cp env.list.example env.list，填入真实 IBKR 账号密码"
echo "4. docker compose up -d"
echo "5. 手机 IB Key 推送确认登录"
echo "6. curl https://localhost:5000/v1/api/portfolio/accounts -k 验证 Gateway"
