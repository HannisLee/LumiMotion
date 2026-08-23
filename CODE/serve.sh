#!/usr/bin/env bash
# 启动本地静态服务器（浏览器 fetch 需要 HTTP 协议）。
cd "$(dirname "$0")"
PORT="${1:-8321}"
echo "查看器地址: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT}/  (本机: http://localhost:${PORT}/)"
python -m http.server "${PORT}"
