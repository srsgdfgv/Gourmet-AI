import time, gc, os
# 与 api 一致：MicroPython 用 urequests
try:
    import urequests as requests
except ImportError:
    import requests
# base64：优先 ubinascii（MicroPython/CanMV），否则 base64
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
# URL 编码：不依赖 .isalnum()/format()，兼容 CanMV/K230（与 api 一致）
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

def get_file_content_as_base64(path, urlencoded=False):
    """读取本地图片 → base64 → 可选 urlencode（与 api 一致）"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        errno = getattr(e, "errno", None)
        if errno == 2 or "ENOENT" in str(e) or "No such file" in str(e):
            raise OSError("ENOENT: 图片不存在。路径: " + repr(path)) from e
        raise
    content = b64encode(raw)
    if isinstance(content, bytes):
        content = content.decode("utf8")
    if urlencoded:
        content = quote_plus(content)
    return content

# 导入K230媒体处理模块
from media.sensor import *
from media.media import *
from media.display import *

# 定义图像宽度和高度常量
WIDTH = 640
HEIGHT = 480

# WiFi配置（请替换为您的WiFi信息）
WIFI_SSID = "1"      # 替换为您的WiFi名称
WIFI_PASSWORD = "lkjhgfdsa"  # 替换为您的WiFi密码

# 百度API配置（与 api 一致，可改为你的密钥）
BAIDU_API_KEY = "SRzYgU0UU80mIrvnWtQD4nZZ"
BAIDU_SECRET_KEY = "HdFhvVvlkypkXoHwMWaWVNscGOvfBiLO"

# 拍摄照片保存路径：存到 /sdcard/，该路径同时作为调用 API 的图片（与 api 的 IMAGE_PATH 一致）
SDCARD_IMAGE_PATH = "/sdcard/capture.jpg"

# === 1. WiFi连接 ===
def wifi_connect(ssid, password):
    """
    连接WiFi网络
    """
    try:
        import network
        wlan = network.WLAN(network.STA_IF)
        wlan.active(True)

        if not wlan.isconnected():
            print(f"正在连接WiFi: {ssid}...")
            wlan.connect(ssid, password)

            # 等待连接，最多等待20秒
            timeout = 20
            while not wlan.isconnected() and timeout > 0:
                time.sleep(1)
                timeout -= 1
                print(".", end="")
            print()

            if wlan.isconnected():
                ip = wlan.ifconfig()[0]
                print(f"WiFi连接成功! IP地址: {ip}")
                return True
            else:
                print("WiFi连接失败，请检查SSID和密码")
                return False
        else:
            ip = wlan.ifconfig()[0]
            print(f"WiFi已连接，IP地址: {ip}")
            return True
    except Exception as e:
        print(f"WiFi连接异常: {e}")
        # 如果network模块不存在，尝试使用其他方式
        try:
            import socket
            # 尝试创建一个socket来测试网络
            s = socket.socket()
            s.connect(("www.baidu.com", 80))
            s.close()
            print("网络连接正常（使用socket测试）")
            return True
        except:
            print("无法连接网络，请检查WiFi配置")
            return False

# === 2. 获取百度 Access Token（与 api 一致：URL 拼接 + 兼容 json 解析） ===
def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    url_full = url + "?grant_type=client_credentials&client_id=" + BAIDU_API_KEY + "&client_secret=" + BAIDU_SECRET_KEY
    resp = requests.post(url_full)
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

# === 4. 摄像头初始化 (参考camera.py的语法) ===
def camera_init():
    """
    初始化K230摄像头传感器
    """
    # 创建传感器对象，设置分辨率
    sensor = Sensor(width=WIDTH, height=HEIGHT, fps=30)
    # 传感器复位
    sensor.reset()
    # 设置输出图像尺寸
    sensor.set_framesize(width=WIDTH, height=HEIGHT)
    # 设置输出格式为RGB888，便于后续编码为JPEG
    sensor.set_pixformat(Sensor.RGB888)
    # 初始化媒体管理器
    MediaManager.init()
    # 启动传感器运行
    sensor.run()
    return sensor

# === 5. 图像捕获与编码（带尺寸和大小检查） ===
def capture_and_encode(sensor):
    """
    捕获一帧图像并编码为JPEG格式
    符合百度API要求：
    - 最短边至少15px，最长边最大4096px
    - base64编码后大小不超过4M
    - 支持jpg/png/bmp格式（这里使用jpg）
    """
    # 捕获一帧图像
    img = sensor.snapshot()

    # 检查图像尺寸（使用预定义的常量）
    img_width = WIDTH
    img_height = HEIGHT
    min_side = min(img_width, img_height)
    max_side = max(img_width, img_height)

    # 检查尺寸要求：最短边至少15px，最长边最大4096px
    if min_side < 15:
        raise Exception(f"图像尺寸不符合要求：最短边{min_side}px，需要至少15px")
    if max_side > 4096:
        raise Exception(f"图像尺寸不符合要求：最长边{max_side}px，需要不超过4096px")

    print(f"图像尺寸: {img_width}x{img_height} (符合API要求: 最短边≥15px, 最长边≤4096px)")

    # 尝试不同的编码方法（根据K230实际API选择）
    # 方法1：使用compress方法（常见）
    if hasattr(img, 'compress'):
        # 根据图像大小调整质量，确保base64编码后不超过4M
        # 4M = 4 * 1024 * 1024 = 4194304 字节
        # base64编码会增加约33%的大小，所以原始数据应该不超过约3M
        # 先尝试质量85，如果太大再降低
        quality = 85
        jpeg_data = img.compress(quality=quality)

        # 检查base64编码后的大小（base64编码会增加约33%）
        base64_size_estimate = len(jpeg_data) * 4 // 3
        max_base64_size = 4 * 1024 * 1024  # 4M

        # 如果估计大小超过4M，降低质量重新编码
        if base64_size_estimate > max_base64_size:
            print(f"图像较大({base64_size_estimate}字节)，降低质量重新编码...")
            quality = 60
            jpeg_data = img.compress(quality=quality)
            base64_size_estimate = len(jpeg_data) * 4 // 3
            if base64_size_estimate > max_base64_size:
                quality = 40
                jpeg_data = img.compress(quality=quality)

        print(f"JPEG编码完成，质量: {quality}，大小: {len(jpeg_data)} 字节")

    # 方法2：使用to_jpeg方法
    elif hasattr(img, 'to_jpeg'):
        jpeg_data = img.to_jpeg(quality=85)
        base64_size_estimate = len(jpeg_data) * 4 // 3
        if base64_size_estimate > 4 * 1024 * 1024:
            jpeg_data = img.to_jpeg(quality=60)
    # 方法3：使用media模块的编码器
    else:
        try:
            from media.media import Encoder
            encoder = Encoder()
            encoder.create("jpeg")
            encoder.encode(img)
            jpeg_data = encoder.get_result()
            encoder.destroy()
        except:
            # 如果以上方法都不可用，抛出异常提示用户
            raise Exception("无法编码图像为JPEG格式。请检查K230 media API文档，确认Image对象的编码方法。")

    # 最终检查base64编码后的大小
    base64_encoded = b64encode(jpeg_data)
    base64_size = len(base64_encoded)
    max_base64_size = 4 * 1024 * 1024  # 4M

    if base64_size > max_base64_size:
        raise Exception(f"base64编码后大小({base64_size}字节)超过4M限制，请降低图像质量或尺寸")

    print(f"base64编码后大小: {base64_size} 字节 ({base64_size / 1024 / 1024:.2f}M)")

    # 保存到 /sdcard/，该路径作为 API 使用的图片（与 api 的 IMAGE_PATH 一致）
    try:
        with open(SDCARD_IMAGE_PATH, "wb") as f:
            f.write(jpeg_data)
        print("已保存到:", SDCARD_IMAGE_PATH)
    except Exception as e:
        print("保存到 /sdcard 失败:", e)

    return SDCARD_IMAGE_PATH

# === 6. 按图片路径调用百度 API 识别（与 api 一致：读文件 base64+urlencode） ===
def baidu_api_recognize_by_path(image_path, access_token):
    """
    使用保存的图片路径调用百度食材识别 API（与 api 逻辑一致）
    """
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient?access_token=" + access_token
    image_base64 = get_file_content_as_base64(image_path, urlencoded=True)
    payload = "image=" + image_base64 + "&baike_num=5"
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    response = requests.request("POST", url, headers=headers, data=body)
    raw = getattr(response, "text", None) or (response.content.decode("utf-8") if getattr(response, "content", None) else "")
    try:
        j = getattr(response, "json", None)
        if callable(j):
            return j()
        if j is not None:
            return j
    except Exception:
        pass
    try:
        import ujson
        return ujson.loads(raw)
    except Exception:
        try:
            import json
            return json.loads(raw)
        except Exception:
            return {}

def baidu_api_recognize(image_data, access_token):
    """
    调用百度API识别图像中的果蔬类型
    符合API要求：
    - base64编码后去掉编码头（data:image/jpg;base64,）
    - 进行urlencode
    """
    url = f"https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient?access_token={access_token}"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    # 将字节数据转换为base64字符串（使用开头定义的b64encode函数）
    img_base64_bytes = b64encode(image_data)
    img_base64 = img_base64_bytes.decode('utf-8')

    # 注意：API要求去掉编码头（data:image/jpg;base64,），直接使用base64字符串再 quote_plus
    img_base64_encoded = quote_plus(img_base64)

    # 构建payload，使用urlencode后的base64字符串
    payload = f'image={img_base64_encoded}&baike_num=5'

    # 发送POST请求
    response = requests.post(url, headers=headers, data=payload)
    return response.json()

# === 7. 解析识别结果，提取置信度最大的类别 ===
def parse_result(result):
    """
    解析API返回结果，返回置信度最大的果蔬类别
    """
    if "error_code" in result:
        return None, f"API错误: {result.get('error_msg', '未知错误')}"

    if "result" not in result or not result["result"]:
        return None, "未识别到任何果蔬"

    # 获取结果列表
    results = result["result"]
    if not results:
        return None, "未识别到任何果蔬"

    # 找到置信度最大的结果
    best_result = max(results, key=lambda x: x.get("score", 0))
    name = best_result.get("name", "未知")
    score = best_result.get("score", 0)

    return name, score

# === 8. 主函数 ===
def main():
    print("K230果蔬识别程序启动...")
    sensor = None

    try:
        # 首先连接WiFi
        print("正在连接WiFi网络...")
        if not wifi_connect(WIFI_SSID, WIFI_PASSWORD):
            raise Exception("WiFi连接失败，无法继续执行")

        # 初始化摄像头
        print("正在初始化摄像头...")
        sensor = camera_init()
        print("摄像头初始化成功")

        # 获取百度云Access Token
        print("正在获取百度API访问令牌...")
        access_token = get_access_token()
        print("访问令牌获取成功")
        print("拍摄一次照片并识别...")

        # 仅拍摄一次：捕获并保存到 /sdcard/capture.jpg
        image_path = capture_and_encode(sensor)
        print("使用图片路径:", image_path)

        # 按路径调用云端 API 识别
        result = baidu_api_recognize_by_path(image_path, access_token)
        name, score = parse_result(result)

        if name:
            print(f"识别结果: {name} (置信度: {score:.2%})")
        else:
            print(f"识别失败: {score}")

    except KeyboardInterrupt:
        print("\n程序被用户终止")
    except BaseException as e:
        print(f"程序异常: {e}")
    finally:
        # 释放资源
        print("正在清理资源...")
        if sensor and isinstance(sensor, Sensor):
            sensor.stop()
        MediaManager.deinit()
        print("资源清理完成")

if __name__ == '__main__':
    main()
