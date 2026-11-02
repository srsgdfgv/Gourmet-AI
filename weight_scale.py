# -*- coding: utf-8 -*-
"""
重量秤逻辑：对应 HX711.c / HX711.h / main.c 的 Python 实现
- 中值滤波（5 点）
- 去皮、标定系数换算为克
- 无 OLED，仅重量采集与滤波

稳定判定：连续「超过 10 次」即至少 11 次读到相同且非零的重量（允许 ±1g 抖动），才作为输出。
零点：先判定空秤读数稳定并固定皮重，再放食材；避免「启动时秤上有物」去皮把食材当成皮重导致净重一直为 0。
"""
import time
import RPi.GPIO as GPIO
from hx711v0_5_1 import HX711

# 与 main.c 一致的常量
MEDIAN_LEN = 5
MEDIAN_IDX = 2
WEIGHT_SCALE = 100000
HX711_XISHU_DEFAULT = 31263   # 标定系数：1000g 砝码显示 934g 则 原值*1000/934

# 连续多少次相同且非零判为稳定 —「10次以上」取至少 11 次
STABLE_COUNT = 11
# 空秤判定：读数落在此带宽内视为 0（抗噪声）
ZERO_BAND_G = 3
# 两次读数相差不超过此值视为「相同」
WEIGHT_MATCH_TOLERANCE_G = 1


def _median_filter_add(buf, length, val):
    """
    插入新值并保持升序，返回 (新buffer, 新length, 是否输出中值, 中值)。
    与 C 的 MedianFilter_Add 一致。
    """
    buf = list(buf)
    if length == 0:
        buf[0] = val
        return buf, 1, False, 0
    # 插入排序
    for i in range(length):
        if buf[i] > val:
            buf[i], val = val, buf[i]
    buf[length] = val
    length += 1
    if length >= MEDIAN_LEN:
        return buf, 0, True, buf[MEDIAN_IDX]
    return buf, length, False, 0


def _same_stable_weight(a, b, tol=WEIGHT_MATCH_TOLERANCE_G):
    return abs(int(a) - int(b)) <= tol


class WeightScale:
    """基于 HX711 的重量秤：去皮 + 标定系数 + 中值滤波，与 main.c 逻辑一致。"""

    def __init__(self, dout=5, pd_sck=6, gain=128, hx711_xishu=HX711_XISHU_DEFAULT):
        self.hx = HX711(dout=dout, pd_sck=pd_sck, gain=gain)
        self.hx711_xishu = hx711_xishu
        self.pi_weight = 0          # 皮重（缩放 100 后的原始值）
        self._median_buf = [0] * MEDIAN_LEN
        self._median_len = 0
        self._last_weight_g = 0    # 上次滤波后的重量，未满 5 点时不更新

    def _raw_scaled(self, channel='A'):
        """读一次 HX711 并转为与 C 一致的缩放值 get = (uint)(raw*0.01)。"""
        raw = self.hx.getLong(channel=channel)
        if raw is None:
            return None
        return int(raw * 0.01)

    def get_tare(self, channel='A'):
        """去皮：5 次采样中值作为皮重。"""
        self._median_buf = [0] * MEDIAN_LEN
        self._median_len = 0
        median_val = 0
        for _ in range(MEDIAN_LEN):
            v = self._raw_scaled(channel)
            if v is None:
                continue
            self._median_buf, self._median_len, ready, median_val = _median_filter_add(
                self._median_buf, self._median_len, v
            )
        self.pi_weight = median_val

    def reset_median_filter(self):
        self._median_buf = [0] * MEDIAN_LEN
        self._median_len = 0
        self._last_weight_g = 0

    def get_weight_raw(self, channel='A'):
        """
        单次重量计算（与 C 的 Get_Weight 一致）：
        get = raw*0.01; 若 get>pi_weight 则再读一次，aa = a*0.01 - pi_weight，weight = aa * xishu / 100000。
        """
        get = self._raw_scaled(channel)
        if get is None:
            return None
        if get > self.pi_weight:
            a = self._raw_scaled(channel)
            if a is None:
                return None
            aa = a - self.pi_weight
            weight = int(aa * self.hx711_xishu / WEIGHT_SCALE)
            return max(0, weight)
        return 0

    def get_weight_g(self, channel='A'):
        """
        带中值滤波的重量（克）：满 5 个样本输出中值并更新，否则返回上次滤波结果（与 main.c 一致）。
        """
        w = self.get_weight_raw(channel=channel)
        if w is None:
            return self._last_weight_g
        self._median_buf, self._median_len, ready, median_val = _median_filter_add(
            self._median_buf, self._median_len, w
        )
        if ready:
            self._last_weight_g = median_val
            return median_val
        return self._last_weight_g

    def cleanup(self):
        GPIO.cleanup()


PROGRESS_PING_SEC = 1.5  # 称重阶段向 UI 汇报间隔（秒）


def wait_stable_empty_scale(scale, interval=0.2, stable_zero_count=STABLE_COUNT,
                            zero_band_g=ZERO_BAND_G, timeout_sec=60):
    """
    空秤上的读数连续 stable_zero_count 次落在 [0, zero_band_g] 内，认为「0 状态」已稳定。
    返回 True 表示空秤就绪；超时返回 False。
    调用前应已执行过一次 get_tare()，且秤上应无食材。
    """
    scale.reset_median_filter()
    deadline = time.time() + timeout_sec
    consecutive = 0
    while time.time() < deadline:
        w = scale.get_weight_g()
        if w is None:
            time.sleep(interval)
            continue
        if w <= zero_band_g:
            consecutive += 1
            if consecutive >= stable_zero_count:
                return True
        else:
            consecutive = 0
        time.sleep(interval)
    return False


def iter_read_stable_weight_progress(
    dout=5,
    pd_sck=6,
    interval=0.2,
    stable_count=STABLE_COUNT,
    timeout_sec=120,
    zero_band_g=ZERO_BAND_G,
    empty_confirm_timeout_sec=60,
    progress_ping_sec=PROGRESS_PING_SEC,
):
    """
    与 read_stable_weight_g 相同逻辑，期间 yield 进度 dict，便于 Web 流式调试。
    成功：yield {"phase": "stable_weight", "weight_g": float}
    失败：yield {"phase": "failed", "reason": "empty_timeout"|"weight_timeout", "message": str}
    """
    scale = WeightScale(dout=dout, pd_sck=pd_sck)
    try:
        yield {"phase": "tare_first", "message": "首次去皮采样"}
        scale.get_tare()
        scale.reset_median_filter()

        yield {
            "phase": "empty_wait",
            "message": "等待空秤稳定（请勿在秤上放置食材）…",
            "need_consecutive": stable_count,
            "zero_band_g": zero_band_g,
        }
        deadline_empty = time.time() + empty_confirm_timeout_sec
        consecutive = 0
        last_ping = 0.0
        while time.time() < deadline_empty:
            w = scale.get_weight_g()
            if w is None:
                time.sleep(interval)
                continue
            if w <= zero_band_g:
                consecutive += 1
                if consecutive >= stable_count:
                    break
            else:
                consecutive = 0
            now = time.time()
            if now - last_ping >= progress_ping_sec:
                last_ping = now
                yield {
                    "phase": "empty_tick",
                    "consecutive": consecutive,
                    "need": stable_count,
                    "sample_g": w,
                }
            time.sleep(interval)
        else:
            yield {
                "phase": "failed",
                "reason": "empty_timeout",
                "message": "空秤未在时限内稳定，请取下重物后重试",
            }
            return

        yield {"phase": "tare_lock", "message": "空秤已确认，再次去皮并锁定零点"}
        scale.get_tare()
        scale.reset_median_filter()

        yield {"phase": "zero_locked", "message": "零点已固定，请将果蔬置于秤上并等待读数稳定"}

        last_w = None
        consecutive_w = 0
        deadline_w = time.time() + timeout_sec
        last_ping = 0.0
        while time.time() < deadline_w:
            w = scale.get_weight_g()
            if w is None:
                time.sleep(interval)
                continue
            if w <= zero_band_g:
                consecutive_w = 0
                last_w = None
            else:
                if last_w is not None and _same_stable_weight(w, last_w):
                    consecutive_w += 1
                    if consecutive_w >= stable_count:
                        yield {"phase": "stable_weight", "weight_g": float(w)}
                        return
                else:
                    consecutive_w = 1
                    last_w = w
            now = time.time()
            if now - last_ping >= progress_ping_sec:
                last_ping = now
                yield {
                    "phase": "weight_tick",
                    "consecutive": consecutive_w,
                    "need": stable_count,
                    "sample_g": w,
                    "message": "等待重量稳定（需连续相同读数）…",
                }
            time.sleep(interval)

        yield {
            "phase": "failed",
            "reason": "weight_timeout",
            "message": "称重超时：未得到稳定重量，请检查接线与托盘",
        }
    finally:
        scale.cleanup()


def run_scale_loop(dout=5, pd_sck=6, interval=0.2, stable_count=STABLE_COUNT):
    """
    循环读取重量：先固定空秤零点，再监视食材。
    连续 stable_count 次（默认 11，即「超过 10 次」）读到相同非零重量后输出一次。
    """
    scale = WeightScale(dout=dout, pd_sck=pd_sck)
    try:
        scale.get_tare()
        print("请保持秤盘为空，正在确认零点…")
        if not wait_stable_empty_scale(scale, interval=interval, stable_zero_count=stable_count):
            print("超时：未检测到稳定空秤，请取下重物后重试。")
            return
        scale.get_tare()
        scale.reset_median_filter()
        print("零点已固定，请放置食材。")

        last_w = None
        consecutive = 0
        stable_reported = False
        while True:
            w = scale.get_weight_g()
            if w is None:
                time.sleep(interval)
                continue
            if w <= ZERO_BAND_G:
                print("实时重量: 0 g（未放置食物）")
                consecutive = 0
                last_w = None
                stable_reported = False
            else:
                if last_w is not None and _same_stable_weight(w, last_w):
                    consecutive += 1
                    if consecutive >= stable_count:
                        if not stable_reported:
                            print("当前食材的重量为{}g".format(w))
                            stable_reported = True
                    else:
                        print("实时重量: {} g".format(w))
                else:
                    consecutive = 1
                    last_w = w
                    stable_reported = False
                    print("实时重量: {} g".format(w))
            time.sleep(interval)
    finally:
        scale.cleanup()


def read_stable_weight_g(
    dout=5,
    pd_sck=6,
    interval=0.2,
    stable_count=STABLE_COUNT,
    timeout_sec=120,
    zero_band_g=ZERO_BAND_G,
    empty_confirm_timeout_sec=60,
):
    """
    单次会话：
    1) 先去皮一次，再等待空秤读数连续 stable_count 次稳定在零点带内，再去皮一次并固定为会话零点；
       避免「程序启动时秤上已有食材」把食材计入皮重，导致放上后净重一直为 0。
    2) 再循环读取，直到同一非零重量连续出现 stable_count 次（默认 11 =「超过 10 次」相同），返回该重量（克）。
    失败或超时返回 None。
    """
    for ev in iter_read_stable_weight_progress(
        dout=dout,
        pd_sck=pd_sck,
        interval=interval,
        stable_count=stable_count,
        timeout_sec=timeout_sec,
        zero_band_g=zero_band_g,
        empty_confirm_timeout_sec=empty_confirm_timeout_sec,
    ):
        if ev.get("phase") == "stable_weight":
            return ev.get("weight_g")
        if ev.get("phase") == "failed":
            return None
    return None


if __name__ == "__main__":
    run_scale_loop(dout=5, pd_sck=6, interval=0.2)
