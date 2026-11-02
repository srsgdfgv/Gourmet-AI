# K230 / CanMV：TCP 服务器 —— 监听局域网内树莓派发来的「开始识别」指令，拍照并调用百度 API，返回一行 JSON。
# 使用：把本文件拷到 K230，改 WIFI / 端口后运行；树莓派上运行 raspberry_pi_client.py，填 K230 的 IP。

import time
import socket

try:
    import ujson as json
except ImportError:
    import json

# 与 api 一致：MicroPython 用 urequests
try:
    import urequests as requests
except ImportError:
    import requests

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


_EMBEDDED = False
Sensor = None
MediaManager = None


def _ensure_media():
    """WiFi 就绪后再加载 media，避免 main.py 自启动时与 network 争用导致 IDE interrupt。"""
    global _EMBEDDED, Sensor, MediaManager
    if _EMBEDDED:
        return True
    try:
        from media.sensor import Sensor as _Sensor
        from media.media import MediaManager as _MediaManager
        import media.display  # 模块级 side effect；函数内不可用 import *
        Sensor = _Sensor
        MediaManager = _MediaManager
        _EMBEDDED = True
        return True
    except ImportError:
        return False

WIDTH = 640
HEIGHT = 480
# 拍照前等待自动曝光：先 sleep 再丢帧，避免首帧过暗（弱光/冰箱场景）
CAPTURE_WARMUP_SEC = 0.4
CAPTURE_SKIP_FRAMES = 8
# TCP 长驻时 sensor 一直 run()，AE 易锁暗场；每次 START 拍照前额外丢帧（capture_only 每次新建 sensor 无此问题）
TCP_SERVICE_EXTRA_SKIP = 15

# ---------- 密钥 / WiFi / 端口来自 local_config.py（板端请拷贝到同目录）----------
try:
    import local_config as _cfg
except ImportError:
    try:
        import sys
        sys.path.insert(0, "..")
        import local_config as _cfg
    except ImportError:
        raise ImportError("缺少 local_config.py，请复制 local_config.example.py 为 local_config.py 并填写")

WIFI_SSID = _cfg.WIFI_SSID
WIFI_PASSWORD = _cfg.WIFI_PASSWORD
# main.py 自启动：上电后稍等 WiFi 模块就绪
BOOT_WIFI_DELAY_SEC = 3
# connect 后固定等待（DHCP/关联）；此前验证有效，须保留
WIFI_CONNECT_POST_DELAY_SEC = 5
WIFI_CONNECT_TIMEOUT_SEC = 25
WIFI_CONNECT_RETRIES = 5

BAIDU_API_KEY = _cfg.BAIDU_VISION_API_KEY
BAIDU_SECRET_KEY = _cfg.BAIDU_VISION_SECRET_KEY
SDCARD_IMAGE_PATH = "/sdcard/capture.jpg"

# 监听所有网卡，端口可改；树莓派客户端里 PORT 要一致
TCP_HOST = "0.0.0.0"
TCP_PORT = getattr(_cfg, "TCP_PORT", 18888)

# 树莓派发来的指令（UTF-8 一行，以 \\n 结尾）
CMD_START = "START"
CMD_START_CN = "开始识别"
CMD_PING = "PING"


def _is_wlan_connected(wlan):
    try:
        ic = wlan.isconnected
        return ic() if callable(ic) else bool(ic)
    except Exception:
        return False


def _is_ide_interrupt(exc):
    if isinstance(exc, KeyboardInterrupt):
        return True
    return "ide interrupt" in str(exc).lower()


def wifi_connect(ssid, password, retries=WIFI_CONNECT_RETRIES):
    """
    连接 WiFi。main.py 自启动时 CanMV 可能抛 IDE interrupt，需重试而非一次失败即退出。
    connect 后保留 WIFI_CONNECT_POST_DELAY_SEC（默认 5s）等待关联/DHCP。
    """
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            import network
            wlan = network.WLAN(network.STA_IF)
            wlan.active(True)
            if _is_wlan_connected(wlan):
                print("WiFi 已连接，IP:", wlan.ifconfig()[0])
                return True
            print("正在连接 WiFi:", ssid, "...（第", attempt, "/", retries, "次）")
            try:
                wlan.disconnect()
            except Exception:
                pass
            time.sleep(0.5)
            wlan.connect(ssid, password)
            # K230 连 WiFi 后需等待 DHCP/关联完成，立即轮询 isconnected() 易误判失败
            time.sleep(WIFI_CONNECT_POST_DELAY_SEC)
            timeout = WIFI_CONNECT_TIMEOUT_SEC
            while not _is_wlan_connected(wlan) and timeout > 0:
                time.sleep(1)
                timeout -= 1
                print(".", end="")
            print()
            if _is_wlan_connected(wlan):
                print("WiFi 成功，IP:", wlan.ifconfig()[0])
                return True
            print("WiFi 本轮未连上")
        except BaseException as e:
            last_err = e
            if _is_ide_interrupt(e):
                print("WiFi 被 IDE 调试打断，2 秒后重试…")
                time.sleep(2)
                continue
            print("WiFi 异常:", e)
        if attempt < retries:
            time.sleep(2)
    if last_err is not None:
        print("WiFi 最终失败:", last_err)
    else:
        print("WiFi 失败，请检查 SSID/密码")
    return False


def get_access_token():
    url = "https://aip.baidubce.com/oauth/2.0/token"
    url_full = (
        url
        + "?grant_type=client_credentials&client_id="
        + BAIDU_API_KEY
        + "&client_secret="
        + BAIDU_SECRET_KEY
    )
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
        raw = getattr(resp, "text", None) or (
            resp.content.decode("utf-8") if getattr(resp, "content", None) else None
        )
        if raw:
            try:
                import ujson as uj
                data = uj.loads(raw)
            except Exception:
                try:
                    data = json.loads(raw)
                except Exception:
                    data = {}
        else:
            data = {}
    token = data.get("access_token") if isinstance(data, dict) else None
    if not token:
        raise RuntimeError("获取 access_token 失败: " + str(data))
    return token


def camera_init():
    if not _ensure_media():
        raise RuntimeError("无法加载 media.sensor（请在 K230 上运行）")
    sensor = Sensor(width=WIDTH, height=HEIGHT, fps=30)
    sensor.reset()
    sensor.set_framesize(width=WIDTH, height=HEIGHT)
    sensor.set_pixformat(Sensor.RGB888)
    _try_enable_auto_exposure(sensor)
    MediaManager.init()
    sensor.run()
    return sensor


def _try_enable_auto_exposure(sensor):
    """若 SDK 支持则打开自动曝光/白平衡（不同固件接口名可能不同）。"""
    for call in (
        lambda: sensor.set_auto_exposure(1),
        lambda: sensor.set_auto_whitebal(1),
        lambda: sensor.set_auto_gain(1),
    ):
        try:
            call()
        except Exception:
            pass


def _warmup_before_snapshot(sensor):
    if CAPTURE_WARMUP_SEC > 0:
        time.sleep(CAPTURE_WARMUP_SEC)
    n = int(CAPTURE_SKIP_FRAMES) if CAPTURE_SKIP_FRAMES else 0
    for _ in range(n):
        try:
            sensor.snapshot()
        except Exception:
            break


def _prepare_sensor_for_capture(sensor):
    """
    每次拍照前准备。TCP 服务与 capture_only 的差异：
    capture_only 启动→立刻 init→预热→拍→stop；AE 对应当前场景。
    TCP 服务 init 后长期 run()，空闲时 AE 可能已适应暗画面，需重新拉 AE 并多丢帧。
    """
    _try_enable_auto_exposure(sensor)
    _warmup_before_snapshot(sensor)
    extra = int(TCP_SERVICE_EXTRA_SKIP) if TCP_SERVICE_EXTRA_SKIP else 0
    for _ in range(extra):
        try:
            sensor.snapshot()
        except Exception:
            break


def capture_and_encode(sensor):
    _prepare_sensor_for_capture(sensor)
    img = sensor.snapshot()
    min_side = min(WIDTH, HEIGHT)
    max_side = max(WIDTH, HEIGHT)
    if min_side < 15:
        raise Exception("图像最短边不足 15px")
    if max_side > 4096:
        raise Exception("图像最长边超过 4096px")

    if hasattr(img, "compress"):
        quality = 85
        jpeg_data = img.compress(quality=quality)
        if len(jpeg_data) * 4 // 3 > 4 * 1024 * 1024:
            quality = 60
            jpeg_data = img.compress(quality=quality)
            if len(jpeg_data) * 4 // 3 > 4 * 1024 * 1024:
                jpeg_data = img.compress(quality=40)
    elif hasattr(img, "to_jpeg"):
        jpeg_data = img.to_jpeg(quality=85)
        if len(jpeg_data) * 4 // 3 > 4 * 1024 * 1024:
            jpeg_data = img.to_jpeg(quality=60)
    else:
        from media.media import Encoder
        encoder = Encoder()
        encoder.create("jpeg")
        encoder.encode(img)
        jpeg_data = encoder.get_result()
        encoder.destroy()

    if len(b64encode(jpeg_data)) > 4 * 1024 * 1024:
        raise Exception("base64 超过 4M")

    try:
        with open(SDCARD_IMAGE_PATH, "wb") as f:
            f.write(jpeg_data)
    except Exception as e:
        print("保存图片警告:", e)

    return SDCARD_IMAGE_PATH


def baidu_api_recognize_by_path(image_path, access_token):
    url = (
        "https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient?access_token="
        + access_token
    )
    image_base64 = get_file_content_as_base64(image_path, urlencoded=True)
    payload = "image=" + image_base64 + "&baike_num=5"
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    response = requests.request("POST", url, headers=headers, data=body)
    raw = getattr(response, "text", None) or (
        response.content.decode("utf-8") if getattr(response, "content", None) else ""
    )
    try:
        j = getattr(response, "json", None)
        if callable(j):
            return j()
        if j is not None:
            return j
    except Exception:
        pass
    try:
        import ujson as uj
        return uj.loads(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return {}


def parse_result(result):
    if "error_code" in result:
        return None, "API错误: " + str(result.get("error_msg", "未知"))
    if "result" not in result or not result["result"]:
        return None, "未识别到果蔬"
    results = result["result"]
    best = max(results, key=lambda x: x.get("score", 0))
    return best.get("name", "未知"), best.get("score", 0)


def run_recognition(sensor, access_token):
    """执行一次拍照 + 百度识别，返回可 JSON 序列化的 dict。"""
    try:
        path = capture_and_encode(sensor)
        api_ret = baidu_api_recognize_by_path(path, access_token)
        name, score = parse_result(api_ret)
        if name:
            return {"ok": True, "name": name, "score": score}
        return {"ok": False, "error": str(score)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _is_eagain(exc):
    """errno 11 / EAGAIN：非阻塞下暂无连接或暂无数据，应重试而非退出。"""
    n = getattr(exc, "errno", None)
    if n is None and getattr(exc, "args", None):
        n = exc.args[0]
    return n == 11 or n == "EAGAIN"


def _socket_set_blocking(sock, blocking=True):
    """K230/CanMV 上监听 socket 可能默认非阻塞，accept 会立刻抛 EAGAIN；强制阻塞模式。"""
    try:
        sock.setblocking(blocking)
    except Exception:
        pass


def _sleep_short():
    if hasattr(time, "sleep_ms"):
        time.sleep_ms(20)
    else:
        time.sleep(0.02)


def accept_blocking(srv):
    """阻塞式 accept；若仍收到 EAGAIN 则短暂等待后重试，避免服务直接停掉。"""
    while True:
        try:
            return srv.accept()
        except OSError as e:
            if _is_eagain(e):
                _sleep_short()
                continue
            raise


def recv_line(sock, max_len=512):
    """读一行 UTF-8（以 \\n 结束）；遇 EAGAIN 时重试（兼容非阻塞边缘情况）。"""
    buf = b""
    while len(buf) < max_len:
        try:
            chunk = sock.recv(1)
        except OSError as e:
            if _is_eagain(e):
                _sleep_short()
                continue
            raise
        if not chunk:
            break
        if chunk == b"\n":
            break
        buf += chunk
    try:
        return buf.decode("utf-8").strip()
    except Exception:
        return ""


def send_json_line(sock, obj):
    line = json.dumps(obj) + "\n"
    data = line.encode("utf-8") if isinstance(line, str) else line
    sock.sendall(data)


def handle_client(sock, sensor, access_token):
    line = recv_line(sock)
    raw = line.strip()
    if not raw:
        send_json_line(sock, {"ok": False, "error": "未收到指令"})
        return
    raw_upper = raw.upper()
    if raw == CMD_START_CN or raw_upper == CMD_START.upper():
        print("收到开始识别，处理中...")
        out = run_recognition(sensor, access_token)
        send_json_line(sock, out)
        print("已返回:", out)
    elif raw_upper == CMD_PING.upper():
        send_json_line(sock, {"ok": True, "msg": "pong"})
    else:
        send_json_line(
            sock,
            {
                "ok": False,
                "error": "未知指令，请发 START 或 开始识别",
                "got": raw[:80],
            },
        )


def main():
    sensor = None
    try:
        print("=== K230 果蔬识别 TCP 服务 ===")
        if BOOT_WIFI_DELAY_SEC > 0:
            print("等待 WiFi 模块就绪", BOOT_WIFI_DELAY_SEC, "秒…")
            time.sleep(BOOT_WIFI_DELAY_SEC)
        if not wifi_connect(WIFI_SSID, WIFI_PASSWORD):
            raise RuntimeError("WiFi 未连接")

        print("初始化摄像头...")
        sensor = camera_init()
        print("获取百度 token...")
        access_token = get_access_token()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((TCP_HOST, TCP_PORT))
        srv.listen(1)
        _socket_set_blocking(srv, True)
        print("TCP 监听", TCP_PORT, "（树莓派连接本机 IP 与此端口）")

        while True:
            conn, addr = accept_blocking(srv)
            print("来自", addr)
            try:
                _socket_set_blocking(conn, True)
                conn.settimeout(120)
                handle_client(conn, sensor, access_token)
            except Exception as e:
                try:
                    send_json_line(conn, {"ok": False, "error": str(e)})
                except Exception:
                    pass
                print("处理连接异常:", e)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        print("用户中断")
    except BaseException as e:
        print("异常:", e)
    finally:
        print("清理资源...")
        if sensor is not None and Sensor is not None and isinstance(sensor, Sensor):
            try:
                sensor.stop()
            except Exception:
                pass
        if MediaManager is not None:
            try:
                MediaManager.deinit()
            except Exception:
                pass
        print("结束")


if __name__ == "__main__":
    main()
