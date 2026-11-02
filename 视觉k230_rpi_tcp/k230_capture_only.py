# K230 / CanMV：仅拍照并保存到 /sdcard/capture.jpg
# 拍照逻辑与 k230_tcp_server.py 一致（预热丢帧 + JPEG 编码）。
# 使用：将本文件拷到 K230 后运行；无需 WiFi、百度 API、TCP。

import time

try:
    from media.sensor import Sensor
    from media.media import MediaManager
    from media.display import *  # noqa: F401,F403
except ImportError:
    raise RuntimeError("请在 K230 / CanMV 板子上运行本脚本（需要 media.sensor）")

WIDTH = 640
HEIGHT = 480
CAPTURE_WARMUP_SEC = 0.4
CAPTURE_SKIP_FRAMES = 8
SDCARD_IMAGE_PATH = "/sdcard/capture.jpg"


def _reset_ae(sensor):
    toggles = ("set_auto_exposure", "set_auto_whitebal", "set_auto_gain")
    for name in toggles:
        fn = getattr(sensor, name, None)
        if callable(fn):
            try:
                fn(0)
            except Exception:
                pass
    time.sleep(0.05)
    for name in toggles:
        fn = getattr(sensor, name, None)
        if callable(fn):
            try:
                fn(1)
            except Exception:
                pass


def _try_enable_auto_exposure(sensor):
    _reset_ae(sensor)


def camera_init():
    sensor = Sensor(width=WIDTH, height=HEIGHT, fps=30)
    sensor.reset()
    sensor.set_framesize(width=WIDTH, height=HEIGHT)
    sensor.set_pixformat(Sensor.RGB888)
    _try_enable_auto_exposure(sensor)
    MediaManager.init()
    sensor.run()
    return sensor


def _warmup_before_snapshot(sensor):
    if CAPTURE_WARMUP_SEC > 0:
        time.sleep(CAPTURE_WARMUP_SEC)
    n = int(CAPTURE_SKIP_FRAMES) if CAPTURE_SKIP_FRAMES else 0
    for _ in range(n):
        try:
            sensor.snapshot()
        except Exception:
            break


def _encode_jpeg(img):
    if hasattr(img, "compress"):
        quality = 85
        jpeg_data = img.compress(quality=quality)
        if len(jpeg_data) * 4 // 3 > 4 * 1024 * 1024:
            quality = 60
            jpeg_data = img.compress(quality=quality)
            if len(jpeg_data) * 4 // 3 > 4 * 1024 * 1024:
                jpeg_data = img.compress(quality=40)
        return jpeg_data, quality
    if hasattr(img, "to_jpeg"):
        jpeg_data = img.to_jpeg(quality=85)
        if len(jpeg_data) * 4 // 3 > 4 * 1024 * 1024:
            jpeg_data = img.to_jpeg(quality=60)
        return jpeg_data, 85
    from media.media import Encoder
    encoder = Encoder()
    encoder.create("jpeg")
    encoder.encode(img)
    jpeg_data = encoder.get_result()
    encoder.destroy()
    return jpeg_data, 85


def capture_and_save(sensor, path=SDCARD_IMAGE_PATH):
    """与 k230_tcp_server.capture_and_encode 相同流程，仅保存 JPEG。"""
    _warmup_before_snapshot(sensor)
    img = sensor.snapshot()

    min_side = min(WIDTH, HEIGHT)
    max_side = max(WIDTH, HEIGHT)
    if min_side < 15:
        raise Exception("图像最短边不足 15px")
    if max_side > 4096:
        raise Exception("图像最长边超过 4096px")

    jpeg_data, quality = _encode_jpeg(img)
    with open(path, "wb") as f:
        f.write(jpeg_data)

    print("已保存:", path)
    print("尺寸:", WIDTH, "x", HEIGHT, "JPEG质量:", quality, "字节:", len(jpeg_data))
    return path


def main():
    sensor = None
    try:
        print("=== K230 仅拍照 ===")
        print("初始化摄像头...")
        sensor = camera_init()
        print("拍照中（预热", CAPTURE_WARMUP_SEC, "s，丢帧", CAPTURE_SKIP_FRAMES, "）...")
        capture_and_save(sensor)
        print("完成。")
    except KeyboardInterrupt:
        print("用户中断")
    except BaseException as e:
        print("失败:", e)
        raise
    finally:
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
        try:
            MediaManager.deinit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
