# -*- coding: utf-8 -*-
"""
本地私密配置模板。

使用方法：
1. 复制本文件为 local_config.py
2. 填入真实密钥 / IP / WiFi
3. local_config.py 已在 .gitignore 中，不会上传 GitHub

K230 板端：将 local_config.py 与 recognition.py / k230_tcp_server.py / api.py 一起拷到设备。
"""

# === DeepSeek ===
DEEPSEEK_API_KEY = "sk-your-deepseek-key"
DEEPSEEK_API_BASE = "https://api.deepseek.com"

# === 百度语音（ASR / TTS，供 smart_fridge 使用）===
BAIDU_SPEECH_API_KEY = "your-baidu-speech-api-key"
BAIDU_SPEECH_SECRET_KEY = "your-baidu-speech-secret-key"

# === 百度食材图像识别（recognition / k230 / api）===
BAIDU_VISION_API_KEY = "your-baidu-vision-api-key"
BAIDU_VISION_SECRET_KEY = "your-baidu-vision-secret-key"

# === K230 WiFi ===
WIFI_SSID = "your-wifi-ssid"
WIFI_PASSWORD = "your-wifi-password"

# === 局域网设备 ===
K230_HOST = "192.168.x.x"
K230_HOST_FALLBACK = "192.168.x.222"
TCP_PORT = 18888

# === 本地调试图片路径（api.py；板子上可改为 /sdcard/...）===
IMAGE_PATH = r"/sdcard/1.jpg"
