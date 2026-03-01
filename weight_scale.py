# -*- coding: utf-8 -*-
"""
重量秤逻辑：对应 HX711.c / HX711.h / main.c 的 Python 实现
- 中值滤波（5 点）
- 去皮、标定系数换算为克
- 无 OLED，仅重量采集与滤波
"""
import time
import RPi.GPIO as GPIO
from hx711v0_5_1 import HX711

# 与 main.c 一致的常量
MEDIAN_LEN = 5
MEDIAN_IDX = 2
WEIGHT_SCALE = 100000
HX711_XISHU_DEFAULT = 31263   # 标定系数：1000g 砝码显示 934g 则 原值*1000/934
STABLE_COUNT = 10              # 连续多少次重量相同且非 0 时输出「当前食材的重量为 xxg」


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


def run_scale_loop(dout=5, pd_sck=6, interval=0.2, stable_count=STABLE_COUNT):
    """
    循环读取重量：未稳定时打印实时重量；当连续 stable_count 次读到相同重量（且非 0）时
    输出一次「当前食材的重量为 xxg」。0 视为未放置食物，不参与连续判断也不输出该句。
    """
    scale = WeightScale(dout=dout, pd_sck=pd_sck)
    try:
        scale.get_tare()
        scale._median_buf = [0] * MEDIAN_LEN
        scale._median_len = 0
        last_w = None
        consecutive = 0
        stable_reported = False  # 已输出过稳定重量后，在重量未变前不再重复打印
        while True:
            w = scale.get_weight_g()
            if w is None:
                time.sleep(interval)
                continue
            if w == 0:
                print("实时重量: 0 g（未放置食物）")
                consecutive = 0
                last_w = 0
                stable_reported = False
            else:
                if w == last_w:
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


if __name__ == "__main__":
    run_scale_loop(dout=5, pd_sck=6, interval=0.2)
