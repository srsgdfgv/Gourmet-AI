#!/usr/bin/env python3
# 树莓派：TCP 客户端 —— 向同局域网内 K230 发「开始识别」，收一行 JSON 结果。
# 位置：CRAIC 项目根目录（与 web.uicopy.py、smart_fridge.py 同级）。
# 使用前：pip 无需额外依赖（仅用标准库）。

import json
import socket
import sys

import subprocess


def get_pi_ip():
    """获取树莓派自己的IP"""
    result = subprocess.run(
        "hostname -I | awk '{print $1}'",
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def get_k230_ip():
    """根据树莓派IP生成K230应设的IP（前3段 + .222）"""
    pi_ip = get_pi_ip()
    parts = pi_ip.split(".")
    return f"{parts[0]}.{parts[1]}.{parts[2]}.222"


# ---------- 默认 K230：按树莓派本机 IP 推导出 x.x.x.222；可用参数覆盖 ----------
TCP_PORT = 18888


def default_k230_host():
    """树莓派上通常可用；非 Linux 或失败时退回本地占位，便于开发端导入不报错。"""
    try:
        return get_k230_ip()
    except Exception:
        return "192.168.1.222"


# 与 K230 服务端约定一致
COMMAND = "START\n"
TIMEOUT_SEC = 120


def recv_line(sock, max_bytes=65536):
    data = b""
    while len(data) < max_bytes:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            line, _, _ = data.partition(b"\n")
            return line.decode("utf-8", errors="replace").strip()
    return data.decode("utf-8", errors="replace").strip()


def request_recognize(host=None, port=TCP_PORT):
    if host is None:
        host = default_k230_host()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(TIMEOUT_SEC)
    s.connect((host, port))
    try:
        s.sendall(COMMAND.encode("utf-8"))
        line = recv_line(s)
        if not line:
            return {"ok": False, "error": "K230 无返回数据"}
        return json.loads(line)
    finally:
        s.close()


def ping(host=None, port=TCP_PORT):
    if host is None:
        host = default_k230_host()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect((host, port))
    try:
        s.sendall(b"PING\n")
        line = recv_line(s)
        return json.loads(line) if line else {}
    finally:
        s.close()


def main():
    host = get_k230_ip()
    if len(sys.argv) > 1:
        host = sys.argv[1]
    port = TCP_PORT
    if len(sys.argv) > 2:
        port = int(sys.argv[2])

    print("连接", host + ":" + str(port), "...")
    try:
        result = request_recognize(host, port)
    except socket.timeout:
        print("超时：K230 是否在跑 k230_tcp_server.py？网络是否同一 WiFi？")
        sys.exit(1)
    except ConnectionRefusedError:
        print("拒绝连接：请先在 K230 上启动 TCP 服务，并检查 IP/端口。")
        sys.exit(1)
    except Exception as e:
        print("错误:", e)
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("ok"):
        name = result.get("name", "")
        score = result.get("score", "")
        print("识别:", name, "置信度:", score)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
