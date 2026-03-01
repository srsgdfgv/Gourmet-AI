# MicroPython / CPython 兼容
try:
    import urequests as requests
except ImportError:
    import requests

# base64：优先 ubinascii（MicroPython），否则 base64
try:
    import ubinascii
    if hasattr(ubinascii, "b64encode"):
        def b64encode(data):
            return ubinascii.b64encode(data)
    else:
        def b64encode(data):
            return ubinascii.b2a_base64(data).rstrip(b"\n")
except ImportError:
    import base64
    b64encode = base64.b64encode

# URL 编码：不依赖 .isalnum()，用 ord 判断，兼容 CanMV/K230
def quote_plus(s, safe=""):
    res = []
    for c in s:
        if isinstance(c, int):
            o = c
        elif isinstance(c, (bytes, bytearray)):
            o = c[0] if len(c) > 0 else 0
        else:
            o = ord(c) if c else 0
        if (48 <= o <= 57) or (65 <= o <= 90) or (97 <= o <= 122) or o in (45, 46, 95, 126):
            res.append(chr(o))
        elif o == 32:
            res.append("+")
        else:
            res.append("%" + ("%02X" % o))
    return "".join(res)

# 百度智能云「图像识别」应用的 API Key / Secret Key（必填）
# 若报错 invalid_client / Client authentication failed，说明密钥错误或已失效，请到控制台复制最新密钥：
# https://console.bce.baidu.com/iam/#/applications  → 应用列表 → 你的应用 → 查看 API Key / Secret Key
API_KEY = "SRzYgU0UU80mIrvnWtQD4nZZ"
SECRET_KEY = "ylp6NXSRKoROksochVtg9ZIpH6twZWHd"

# WiFi（板子联网用；PC 可忽略）
WIFI_SSID = "1"
WIFI_PASSWORD = "lkjhgfdsa"

# 本机图片路径：K230 板子用 /sd/1.jpg 或 /flash/1.jpg 或 1.jpg（当前目录）；PC 上改为 r"E:\...\1.jpg"
IMAGE_PATH = "E:\qianren\K2301\code\1.jpg"


def wifi_connect(ssid=WIFI_SSID, password=WIFI_PASSWORD):
    """板子先连 WiFi，再发请求，避免 OSError: no available NIC"""
    try:
        import network
        import time
    except ImportError:
        return True  # PC 无 network，跳过
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if wlan.isconnected():
        print("WiFi 已连接")
        return True
    print("正在连接 WiFi:", ssid)
    wlan.connect(ssid, password)
    for _ in range(20):
        time.sleep(1)
        if wlan.isconnected():
            print("WiFi 连接成功:", wlan.ifconfig()[0])
            return True
    print("WiFi 连接超时")
    return False


def get_file_content_as_base64(path, urlencoded=False):
    """读取本地图片 → base64 → 可选 urlencode（API 要求）"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        errno = getattr(e, "errno", None)
        if errno == 2 or "ENOENT" in str(e) or "No such file" in str(e):
            raise OSError(
                "ENOENT: 图片不存在。当前路径: " + repr(path) +
                "  K230 请用 /sd/1.jpg 或 /flash/1.jpg 或 1.jpg（把 1.jpg 放到 SD/flash/当前目录）"
            ) from e
        raise
    content = b64encode(raw)
    if isinstance(content, bytes):
        content = content.decode("utf8")
    if urlencoded:
        content = quote_plus(content)
    return content


def get_access_token():
    """使用 AK/SK 获取 Access Token（MicroPython 兼容：URL 拼接，无 params）"""
    url = "https://aip.baidubce.com/oauth/2.0/token"
    url_full = url + "?grant_type=client_credentials&client_id=" + API_KEY + "&client_secret=" + SECRET_KEY
    resp = requests.post(url_full)
    # 兼容多种 response：.json() 方法 / .json 属性 / 用 ujson 解析 .text
    data = None
    try:
        j = getattr(resp, "json", None)
        if callable(j):
            data = j()
        elif j is not None:
            data = j
    except Exception:
        pass
    if not isinstance(data, dict):
        raw = getattr(resp, "text", None) or (resp.content.decode("utf-8") if getattr(resp, "content", None) else None)
        if raw:
            try:
                import ujson
                data = ujson.loads(raw)
            except Exception:
                try:
                    import json
                    data = json.loads(raw)
                except Exception:
                    data = {}
        else:
            data = {}
    token = data.get("access_token") if isinstance(data, dict) else None
    if not token:
        raise RuntimeError("获取 Access Token 失败，请检查 API_KEY/SECRET_KEY。返回: " + str(data))
    return token


def main():
    # 板子上先连 WiFi，否则会 OSError: no available NIC
    if not wifi_connect():
        raise OSError("无法连接 WiFi，请检查 WIFI_SSID / WIFI_PASSWORD 或硬件")
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient?access_token=" + get_access_token()
    image_base64 = get_file_content_as_base64(IMAGE_PATH, urlencoded=True)
    payload = "image=" + image_base64 + "&baike_num=5"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
    }
    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    response = requests.request("POST", url, headers=headers, data=body)
    # 兼容不同 response 实现
    if hasattr(response, "encoding"):
        response.encoding = "utf-8"
    text = response.text if hasattr(response, "text") else response.content.decode("utf-8")
    print(text)


if __name__ == "__main__":
    main()
