#!/usr/bin/env python3
# 树莓派：TCP 客户端 —— 向同局域网内 K230 发「开始识别」，收一行 JSON 结果。
# 使用前：pip 无需额外依赖（仅用标准库）。把 K230_HOST 改成 K230 串口/屏幕打印的 IP。

import json
import socket
import sys

# ---------- 改成你 K230 的局域网 IP（与 K230 上 ifconfig/打印的 IP 一致）----------
K230_HOST = "192.168.152.222"
TCP_PORT = 18888

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


def request_recognize(host=K230_HOST, port=TCP_PORT):
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


def ping(host=K230_HOST, port=TCP_PORT):
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
    host = K230_HOST
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
