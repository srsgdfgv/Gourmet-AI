#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能冰箱助手 - DeepSeek 强化版（v3 完整可运行版）
包含：
- DeepSeek 隐形意图识别（evaluate_intent_only）与正常对话（chat）
- 宽容解析 <<ACTION:...>>（支持 JSON 与宽松字符串）
- 中文数量/量词解析（如“五个西红柿”-> name="西红柿", quantity=5）
- 进一步减少破坏性判定：仅批量/全量删除要求确认
- 当用户明确要求“删除错误信息/移出不是食材/删除冰箱里的错误”等时：
- 从 AI 的自然语言回复中提取候选项并尝试直接删除：唯一匹配直接 EXECUTED，模糊/未匹配转 PENDING
- 防止同轮重复执行、超时与错误处理、对 AI 调用失败进行优雅降级

说明：
- 请把真实的 API Key 放入配置（DEEPSEEK_API_KEY / BAIDU_API_KEY / BAIDU_SECRET_KEY）。
- 依赖：requests, numpy, sqlite3 （db.py 提供 SqliteDatabase）。
"""

import os
import re
import time
import json
import base64
import signal
import tempfile
import subprocess
import threading
import shutil
from collections import deque
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta

import requests
import numpy as np

from db import SqliteDatabase

# ==================== 配置（请替换为真实值） ====================
BAIDU_API_KEY = "tTzIVpAh18XePahIJIMXzjdR"
BAIDU_SECRET_KEY = "b9joA9cDxyRmwqsmuXVS1q0B0MtGKya6"
DEEPSEEK_API_KEY = "sk-c4f53f7902f545d4bf58976793d6bff6"
DEEPSEEK_API_BASE = "https://api.deepseek.com"

MIC_DEVICE = "plughw:2,0"
CALIBRATION_DURATION = 1.0
THRESHOLD_OFFSET = 3
INITIAL_WAIT_SEC = 12
POST_SPEECH_WAIT_SEC = 2
CHUNK_DURATION = 0.08
VOICE_DEBOUNCE = 5
MAX_RECORD_DURATION = 12
DB_SMOOTH_WIN = 5
MIN_THRESHOLD = 3

# 自动执行置信度阈值（0.0 - 1.0）
AUTO_EXECUTE_CONFIDENCE_THRESHOLD = 0.75

# 允许的操作（提供给 AI 用于约束）
ALLOWED_OPERATIONS = {
    # 食材 CRUD
    "add_ingredient": {
        "name": "string", 
        "quantity": "number", 
        "unit": "string", 
        "expiry_days": "number", 
        "expiry_date": "string",
        "category": "string",  # 新增：食材类别（果蔬类/肉蛋类/奶制品类/其他）
        "fridge_area": "string"  # 新增：冰箱区域（冷藏区/冷冻区）
    },
    "update_ingredient": {
        "id": "number_or_string", 
        "name": "string", 
        "quantity": "number", 
        "unit": "string", 
        "expiry_date": "string", 
        "expiry_days": "number",
        "category": "string",  # 新增：食材类别
        "fridge_area": "string",  # 新增：冰箱区域
    },
    "remove_ingredient": {"id": "number_or_string", "name": "string"},
    "get_ingredient": {"id": "number_or_string", "name": "string"},  # 新增查
    "clear_inventory": {},
    # 偏好 CRUD
    "set_preference": {"pref_type": "string", "value": "string"},
    "update_preference": {"old_value": "string", "new_value": "string", "pref_type": "string"},  # 新增改
    "remove_preference": {"pref_type": "string", "value": "string"},  # 新增删
    "get_preferences": {"pref_type": "string"},  # 新增查
    "clear_preferences": {},
    # 菜谱 CRUD
    "add_recipe": {"title": "string", "ingredients": "list", "instructions": "string"},
    "update_recipe": {"title": "string", "new_title": "string", "ingredients": "list", "instructions": "string"},  # 新增改
    "delete_recipe": {"title": "string"},  # 新增删
    "get_recipe": {"title": "string", "id": "number_or_string"},  # 新增查
    "mark_recipe_made": {"id": "number_or_string", "title": "string"},
    # 对话记录 CRUD（新增）
    "log_conversation": {"role": "string", "content": "string"},
    "get_conversation": {"limit": "number"},
    "clear_conversation": {},  # 新增清
    # internal aggregate action
    "clean_non_food": {}
}

# 进一步减少破坏性判定：仅在明显一次性删除大量条目或明确全量操作等场景要求确认
DESTRUCTIVE_OPS = {"clear_inventory"}
BULK_DELETE_CONFIRM_THRESHOLD = 10  # 超过该数量的删除视为批量删除，需要确认

class SpeechRecognizer:
    def __init__(self):
        self.access_token = None
        self.token_expire_time = 0
        self.temp_audio_file = None

        # 新增：停止事件 + 播放控制（TTS需要）
        self._stop_event = threading.Event()
        self.playback_proc: Optional[subprocess.Popen] = None
        self.playback_lock = threading.Lock()
        # 新增：音频监听停止事件（用于播放时的分贝检测）
        self._monitor_stop_event = threading.Event()

    # 新增：播放时监听音频分贝的方法
    def _monitor_audio_during_playback(self, silence_threshold):
        """播放音频时后台监听麦克风，分贝超过阈值则停止播放"""
        chunk_bytes = int(16000 * 2 * CHUNK_DURATION)
        db_window = deque(maxlen=DB_SMOOTH_WIN)
        try:
            proc = subprocess.Popen([
                "arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", "-q"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            while not self._monitor_stop_event.is_set() and self.is_playing():
                audio_chunk = proc.stdout.read(chunk_bytes)
                if not audio_chunk:
                    break
                
                # 计算当前分贝
                raw_db = self.calculate_db(audio_chunk)
                db_window.append(raw_db)
                mean_db = round(float(np.mean(db_window)), 2) if db_window else 0.0

                # 分贝超过阈值，停止播放
                if mean_db >= silence_threshold:
                    print(f"\n🔊 检测到说话（分贝：{mean_db:.1f}），停止当前语音播放")
                    self.stop_playback()
                    break

                time.sleep(CHUNK_DURATION)

        except Exception as e:
            print(f"\n⚠️ 播放时监听音频异常：{e}")
        finally:
            try:
                if 'proc' in locals() and proc.poll() is None:
                    proc.terminate()
                    proc.wait()
            except:
                pass

    # 新增：停止控制方法
    def request_stop(self):
        """外部请求：请求停止正在进行的录音/播放"""
        try:
            self._stop_event.set()
        except Exception:
            pass

    def clear_stop(self):
        """清除停止请求（在开始新录音前调用）"""
        try:
            self._stop_event.clear()
        except Exception:
            pass

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    # 新增：播放检测方法
    def is_playing(self) -> bool:
        """返回当前是否正在播放（线程安全）。"""
        with self.playback_lock:
            if self.playback_proc is None:
                return False
            try:
                return self.playback_proc.poll() is None
            except Exception:
                return False

    def calculate_db(self, audio_data: bytes) -> float:
        try:
            if len(audio_data) < 100:
                return 0.0
            samples = np.frombuffer(audio_data, dtype=np.int16)
            valid_samples = samples[(np.abs(samples) <= 32767)]
            if len(valid_samples) == 0:
                return 0.0
            rms = np.sqrt(np.mean(np.square(valid_samples)))
            if rms < 1e-6:
                return 0.0
            db = 20 * np.log10(rms / 32767) + 100
            db = max(0.0, min(100.0, db))
            return round(db, 22) if False else round(db, 2)
        except Exception:
            return 0.0

    def calibrate_background_noise(self) -> float:
        chunk_bytes = int(16000 * 2 * CHUNK_DURATION)
        calibration_chunks = max(1, int(CALIBRATION_DURATION / CHUNK_DURATION))
        noise_dbs = []
        cmd = ["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", "-q"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for _ in range(calibration_chunks):
                chunk = proc.stdout.read(chunk_bytes)
                if chunk:
                    db = self.calculate_db(chunk)
                    if db > 0:
                        noise_dbs.append(db)
                time.sleep(CHUNK_DURATION)
            proc.send_signal(signal.SIGINT)
            proc.wait()
            if not noise_dbs:
                return MIN_THRESHOLD + THRESHOLD_OFFSET
            background_db = float(np.median(noise_dbs))
            dynamic_threshold = max(background_db + THRESHOLD_OFFSET, MIN_THRESHOLD)
            return dynamic_threshold
        except Exception:
            return MIN_THRESHOLD + THRESHOLD_OFFSET

    def record_audio_adaptive(self) -> bool:
        temp_file = tempfile.NamedTemporaryFile(suffix=".pcm", delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        self.temp_audio_file = temp_filename

        silence_threshold = self.calibrate_background_noise()
        silence_threshold = max(silence_threshold, MIN_THRESHOLD)

        chunk_bytes = int(16000 * 2 * CHUNK_DURATION)
        audio_file = open(temp_filename, "wb")

        start_time = time.time()
        last_speech_time = None
        is_voice_active = False
        activation_counter = 0
        db_window = deque(maxlen=DB_SMOOTH_WIN)

        try:
            proc = subprocess.Popen([
                "arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", "-q"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            print(f"🎤 正在监听...（阈值：{silence_threshold:.1f}dB）")

            while True:
                total_elapsed = time.time() - start_time
                if total_elapsed >= MAX_RECORD_DURATION:
                    print(f"\n⏰ 已达最大录音时长（{MAX_RECORD_DURATION}秒），停止录音")
                    break

                audio_chunk = proc.stdout.read(chunk_bytes)
                if not audio_chunk:
                    break

                audio_file.write(audio_chunk)

                raw_db = self.calculate_db(audio_chunk)
                db_window.append(raw_db)
                mean_db = round(float(np.mean(db_window)), 2) if db_window else 0.0

                # 如果在录音过程中检测到分贝超过阈值，立即停止当前播放（确保即时响应）
                try:
                    if mean_db >= silence_threshold:
                        # 在每次检测到“有人说话”的时候，确保停止当前正在播放的 TTS
                        if self.is_playing():
                            print(f"\n🔊 在录音中检测到说话（分贝：{mean_db:.1f}），立即停止播放")
                            try:
                                # 终止播放器进程
                                self.stop_playback()
                            except Exception:
                                pass
                            try:
                                # 也通知监控线程停止（冗余但安全）
                                self._monitor_stop_event.set()
                            except Exception:
                                pass
                except Exception:
                    pass

                if not is_voice_active:
                    if mean_db >= silence_threshold:
                        activation_counter += 1
                    else:
                        activation_counter = 0
                    if activation_counter >= VOICE_DEBOUNCE:
                        is_voice_active = True
                        last_speech_time = time.time()
                        print(f"\n✅ 检测到说话（分贝：{mean_db:.1f}），开始录音...")
                    elif int(total_elapsed) >= INITIAL_WAIT_SEC:
                        print(f"\n⏰ {INITIAL_WAIT_SEC}秒内无有效说话，停止监听")
                        audio_file.close()
                        if os.path.exists(temp_filename):
                            os.unlink(temp_filename)
                        if proc.poll() is None:
                            proc.terminate()
                            proc.wait()
                        return False
                else:
                    if mean_db >= silence_threshold:
                        last_speech_time = time.time()
                    elif last_speech_time and (time.time() - last_speech_time) >= POST_SPEECH_WAIT_SEC:
                        print(f"\n✅ 说话后{POST_SPEECH_WAIT_SEC}秒静默，结束录音")
                        break

        except Exception as e:
            print(f"\n❌ 录音异常：{e}")
            audio_file.close()
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
            except:
                pass
            return False
        finally:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
            except:
                pass
            audio_file.close()

        if os.path.exists(temp_filename):
            size = os.path.getsize(temp_filename)
            if size < 1000:
                print("⚠️ 录音文件过小，无有效音频")
                os.unlink(temp_filename)
                return False
            print(f"✅ 录音完成（{size} 字节）")
            return True
        else:
            print("❌ 录音文件未生成")
            return False

    def get_access_token(self) -> Optional[str]:
        current_time = time.time()
        if self.access_token and current_time < self.token_expire_time:
            return self.access_token
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": BAIDU_API_KEY, "client_secret": BAIDU_SECRET_KEY}
        try:
            r = requests.post(url, params=params, timeout=10)
            j = r.json()
            if "access_token" in j:
                self.access_token = j["access_token"]
                self.token_expire_time = current_time + j.get("expires_in", 2592000) - 300
                return self.access_token
            else:
                print(f"❌ Token 获取失败: {j}")
                return None
        except Exception as e:
            print(f"❌ Token 获取异常: {e}")
            return None

    def recognize_speech(self, audio_file_path: str) -> Optional[str]:
        print("🔊 正在识别语音...")
        token = self.get_access_token()
        if not token:
            return None
        try:
            with open(audio_file_path, "rb") as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            data = {
                "format": "pcm", "rate": 16000, "channel": 1,
                "cuid": "smart_fridge_voice", "token": token,
                "speech": audio_base64, "len": len(audio_data)
            }
            headers = {"Content-Type": "application/json"}
            r = requests.post("https://vop.baidu.com/server_api", headers=headers, data=json.dumps(data), timeout=20)
            j = r.json()
            if j.get("err_no") == 0:
                recognized = j.get("result", [""])[0]
                print(f"🎯 识别结果: {recognized}")
                return recognized
            else:
                print(f"❌ 识别失败: {j.get('err_msg', '未知错误')}")
                return None
        except Exception as e:
            print(f"❌ 识别过程出错: {e}")
            return None

    def listen_and_recognize(self) -> Optional[str]:
        if not self.record_audio_adaptive():
            return None
        text = None
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            text = self.recognize_speech(self.temp_audio_file)
        try:
            if self.temp_audio_file and os.path.exists(self.temp_audio_file):
                os.unlink(self.temp_audio_file)
                self.temp_audio_file = None
        except:
            pass
        return text

    # 新增：TTS合成与播放相关方法
    def synthesize_speech(self, text: str, out_path: str, per: int = 0, vol: int = 20, spd: int = 8, pit: int = 7) -> bool:
        token = self.get_access_token()
        if not token:
            print("❌ 无法合成语音：缺少 token")
            return False
        tts_url = "https://tsn.baidu.com/text2audio"
        data = {
            "tex": text,
            "tok": token,
            "cuid": "smart_fridge_tts",
            "ctp": 1,
            "lan": "zh",
            "per": per,
            "spd": spd,
            "pit": pit,
            "vol": vol
        }
        print(f"🔈 请求 TTS（vol={vol}, spd={spd}, pit={pit}, per={per}）: {text[:60]}{'...' if len(text)>60 else ''}")
        try:
            resp = requests.post(tts_url, data=data, timeout=20)
            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code == 200 and "audio" in content_type:
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                return True
            else:
                try:
                    err = resp.json()
                    print(f"❌ TTS失败: {err}")
                except Exception:
                    print(f"❌ TTS返回未知内容（content-type={content_type}）")
                return False
        except Exception as e:
            print(f"❌ TTS请求异常: {e}")
            return False

    def _start_player_for_file(self, file_path: str) -> Optional[subprocess.Popen]:
        players = [
            ("mpg123", ["mpg123", "-q", file_path]),
            ("mpg321", ["mpg321", "-q", file_path]),
            ("cvlc", ["cvlc", "--play-and-exit", "--quiet", file_path]),
            ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path]),
            ("vlc", ["vlc", "--play-and-exit", "--intf", "dummy", file_path]),
            ("paplay", ["paplay", file_path])
        ]
        for exe, cmd in players:
            if shutil.which(exe):
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return proc
                except Exception:
                    continue
        print("⚠️ 未找到可用的音频播放器。建议在树莓派上安装 mpg123 或 ffmpeg（提供 ffplay）或 VLC。")
        return None

    def play_audio(self, file_path: str) -> bool:
        proc = self._start_player_for_file(file_path)
        if not proc:
            return False
        with self.playback_lock:
            if self.playback_proc and self.playback_proc.poll() is None:
                try:
                    self.playback_proc.terminate()
                except:
                    pass
                try:
                    self.playback_proc.wait(timeout=0.5)
                except:
                    try:
                        self.playback_proc.kill()
                    except:
                        pass
            self.playback_proc = proc

        print("▶️ 开始播放 TTS 音频:", file_path)
        # 新增：启动播放时的音频监听线程
        self._monitor_stop_event.clear()
        # 先校准背景噪音阈值（复用原有逻辑）
        silence_threshold = self.calibrate_background_noise()
        silence_threshold = max(silence_threshold, MIN_THRESHOLD)
        # 启动监听线程
        monitor_thread = threading.Thread(
            target=self._monitor_audio_during_playback,
            args=(silence_threshold,),
            daemon=True
        )
        monitor_thread.start()

        try:
            proc.wait()
        except Exception:
            pass
        finally:
            # 新增：停止监听线程
            self._monitor_stop_event.set()
            monitor_thread.join(timeout=1.0)  # 等待线程退出
            with self.playback_lock:
                if self.playback_proc is proc:
                    self.playback_proc = None
            print("⏹️ 播放结束:", file_path)
        return True

    def stop_playback(self):
        with self.playback_lock:
            proc = self.playback_proc
            if not proc:
                return False
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=0.8)
                    except Exception:
                        try:
                            proc.kill()
                        except:
                            pass
                self.playback_proc = None
                return True
            except Exception as e:
                print(f"⚠️ 停止播放时出错: {e}")
                try:
                    self.playback_proc = None
                except:
                    pass
                return False

    def speak_async_worker(self, text: str, per: int = 0, vol: int = 15, spd: int = 8, pit: int = 7):
        temp_mp3_path = None
        try:
            self.stop_playback()
            temp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_mp3_path = temp_mp3.name
            temp_mp3.close()

            ok = self.synthesize_speech(text, temp_mp3_path, per=per, vol=vol, spd=spd, pit=pit)
            if not ok:
                if os.path.exists(temp_mp3_path):
                    os.unlink(temp_mp3_path)
                return

            self.request_stop()
            self.play_audio(temp_mp3_path)
        except Exception as e:
            print(f"❌ 后台TTS线程出错: {e}")
        finally:
            try:
                if temp_mp3_path and os.path.exists(temp_mp3_path):
                    os.unlink(temp_mp3_path)
            except:
                pass

    def speak(self, text: str, per: int = 0, vol: int = 15, spd: int = 8, pit: int = 7):
        if not text or not text.strip():
            return
        try:
            # 清理文本：只播报核心回复，去掉DB ACTIONS技术内容
            clean_text = text.split("--- 系统操作 ---")[0].strip()
            t = threading.Thread(target=self.speak_async_worker, args=(clean_text, per, vol, spd, pit), daemon=True)
            t.start()
        except Exception as e:
            print(f"❌ 启动TTS后台线程失败: {e}")

# =================== 辅助解析工具（中文数字、量词、AI 指令宽容解析） ===================
CHINESE_NUM_MAP = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
}
CHINESE_UNIT_TOKENS = ['个', '斤', '克', 'kg', '千克', '盒', '瓶', '包', '片', '根', '两', '毫升', 'ml']

num_pat = re.compile(r'([0-9]+(?:\.[0-9]+)?)')
cn_digit_pat = re.compile(r'[零一二两三四五六七八九十百千]+')

def chinese_to_number(s: str) -> Optional[float]:
    """把简单中文数字或阿拉伯数字字符串转为 float。支持：'五' '十五' '二十' '两' '12' '3.5'"""
    s = (s or "").strip()
    if not s:
        return None
    # try arabic
    m = num_pat.search(s)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    # handle simple chinese numerals up to thousands (basic)
    if not cn_digit_pat.search(s):
        return None
    try:
        total = 0
        tmp = 0
        mapping = {'十':10, '百':100, '千':1000}
        for ch in s:
            if ch in CHINESE_NUM_MAP:
                tmp = CHINESE_NUM_MAP[ch]
            elif ch in mapping:
                if tmp == 0:
                    tmp = 1
                total += tmp * mapping[ch]
                tmp = 0
            else:
                # unknown char
                return None
        total += tmp
        return float(total)
    except Exception:
        return None

def extract_leading_quantity_and_unit(name: str) -> Tuple[Optional[float], Optional[str], str]:
    """
    如果 name 是像 '五个西红柿'、'2个苹果'、'一斤猪肉'，尝试提取数量和单位，返回 (quantity, unit, remaining_name)
    """
    if not name:
        return None, None, name
    s = name.strip()
    # pattern: leading arabic number + unit
    m = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)\s*([%s])\s*(.+)$' % ''.join(re.escape(u) for u in CHINESE_UNIT_TOKENS), s)
    if m:
        q = float(m.group(1))
        unit = m.group(2)
        rest = m.group(3).strip()
        return q, unit, rest
    # pattern: leading chinese numerals + unit, e.g., 五个西红柿 or 三斤猪肉
    m2 = re.match(r'^\s*([零一二两三四五六七八九十百千]+)\s*([%s])\s*(.+)$' % ''.join(re.escape(u) for u in CHINESE_UNIT_TOKENS), s)
    if m2:
        cn = m2.group(1)
        unit = m2.group(2)
        rest = m2.group(3).strip()
        q = chinese_to_number(cn)
        return q, unit, rest
    # pattern: '五个西红柿' without space, try find chinese number + unit prefix
    m3 = re.match(r'^\s*([零一二两三四五六七八九十百千]+)(%s)(.+)$' % '|'.join(re.escape(u) for u in CHINESE_UNIT_TOKENS), s)
    if m3:
        cn = m3.group(1)
        unit = m3.group(2)
        rest = m3.group(3).strip()
        q = chinese_to_number(cn)
        return q, unit, rest
    # pattern: arabic number + unit directly attached: '2个苹果'
    m4 = re.match(r'^\s*([0-9]+(?:\.[0-9]+)?)(%s)(.+)$' % '|'.join(re.escape(u) for u in CHINESE_UNIT_TOKENS), s)
    if m4:
        q = float(m4.group(1))
        unit = m4.group(2)
        rest = m4.group(3).strip()
        return q, unit, rest
    return None, None, name

def sanitize_ingredient_name(name: str) -> str:
    """从名字中去掉前置的数量/单位，返回干净的名称。"""
    return name

def parse_action_string(raw: str) -> Optional[Dict[str, Any]]:
    """
    尝试将非 JSON 的 ACTION 内容解析为标准化字典 {"action": "...", "params": {...}, "confidence": ...}
    支持示例：
     - REMOVE_ITEM:五个西红柿:1.0个
     - DELETE name="羽毛球"
     - UPDATE name="西红柿" quantity=5
    这是宽容解析：尽量提取动作名与键值对。
    """
    s = raw.strip()
    if not s:
        return None
    kv_pairs = {}
    # try to find key="value" or key='value'
    for m in re.finditer(r'(\w+)\s*=\s*"(.*?)"|(\w+)\s*=\s*\'(.*?)\'', s):
        if m.group(1):
            kv_pairs[m.group(1)] = m.group(2)
        else:
            kv_pairs[m.group(3)] = m.group(4)
    # find key=value (no quotes)
    for m in re.finditer(r'(\w+)\s*=\s*([^\s,;]+)', s):
        k = m.group(1)
        v = m.group(2).strip().strip('"').strip("'")
        kv_pairs.setdefault(k, v)

    # if kv_pairs non-empty, guess action as first token
    tokens = re.split(r'[:\s]+', s, maxsplit=1)
    first = tokens[0].upper()
    mapping = {
        'REMOVE': 'remove_ingredient', 'REMOVE_ITEM': 'remove_ingredient', 'DELETE': 'remove_ingredient',
        'DEL': 'remove_ingredient',
        'ADD': 'add_ingredient', 'INSERT': 'add_ingredient',
        'UPDATE': 'update_ingredient', 'SET': 'update_ingredient',
        'SET_SHELF_LIFE': 'set_shelf_life', 'MARK_MADE': 'mark_recipe_made',
        'CLEAR_INVENTORY': 'clear_inventory', 'CLEAN_NON_FOOD': 'clean_non_food'
    }
    action_name = mapping.get(first) or first.lower()
    # If still unknown, try to match known operation names inside s
    for k in ALLOWED_OPERATIONS.keys():
        if k in s:
            action_name = k
            break

    params = {}
    # if kv_pairs found, use them
    if kv_pairs:
        params.update(kv_pairs)
    else:
        # attempt colon-separated positional parsing: ACTION:NAME:QTYUNIT
        parts = s.split(':')
        if len(parts) >= 2:
            # parts[0] is action token
            name_candidate = parts[1].strip()
            qty_candidate = parts[2].strip() if len(parts) >= 3 else None
            params['name'] = name_candidate
            if qty_candidate:
                # try parse number within qty_candidate
                q = chinese_to_number(qty_candidate) or (num_pat.search(qty_candidate).group(1) if num_pat.search(qty_candidate) else None)
                if q:
                    params['quantity'] = float(q)
    # Post-process params: try to coerce numeric strings
    if 'quantity' in params:
        try:
            params['quantity'] = float(params['quantity'])
        except:
            q = chinese_to_number(str(params['quantity']))
            if q is not None:
                params['quantity'] = q
    # sanitize name if present
    if 'name' in params and isinstance(params['name'], str):
        name = params['name']
        q, unit, rest = extract_leading_quantity_and_unit(name)
        if rest and rest.strip():
            params['name'] = rest.strip()
            if q and 'quantity' not in params:
                params['quantity'] = q
            if unit and 'unit' not in params:
                params['unit'] = unit
    return {"action": action_name, "params": params, "confidence": None}

# =================== AI 服务（DeepSeek） ===================
class AIService:
    # 更宽容的 ACTION_PATTERN：捕获 <<ACTION: ... >> 内任意内容，后续解析会尝试 JSON 或 fallback
    ACTION_PATTERN = re.compile(r'<<ACTION:([\s\S]*?)>>')

    def __init__(self, api_key: str, db: Optional[SqliteDatabase] = None, api_base: str = DEEPSEEK_API_BASE):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.messages: List[Dict] = []
        self.system_prompt = (
            "你是**专属智能厨房助手**，严格专注于厨房相关场景服务，**禁止涉及任何厨房无关内容**，核心行为规范如下：\n"
            "1.  **话题范围强制约束（红线规则）**\n"
            "    - 仅允许围绕**厨房核心场景**交互：食材管理（增删改查）、菜谱推荐/制作、烹饪技巧、食材保质期/储存、饮食偏好设置、厨房用品使用建议；\n"
            "    - 严禁回应任何厨房无关话题：天气、新闻、娱乐、科技、游戏、社交八卦、健康医疗（非饮食相关）、生活琐事等，所有无关提问一律**礼貌引导回厨房主题**或**明确拒绝**；\n"
            "    - 示例：用户问「今天天气怎么样」→ 回复「抱歉，我是厨房助手，专注于帮你处理食材、菜谱相关的问题哦~」；用户问「推荐一部电影」→ 回复「我的职责是帮你解决厨房相关需求，比如食材管理、菜谱推荐，有这方面的问题可以随时找我」。\n"
            "2.  **意图识别与回复规则（执行标准）**\n"
            "    - **数据操作意图**：用户明确要求对食材、偏好、菜谱、保质期、对话记录进行**增删改查**时（比如：火龙果的保质期是多少？），仅回复固定友好空话，不涉及任何操作细节、数据信息，下面是三条示例：\n"
            "      ▶ 好的，我来帮你查找/操作\n"
            "      ▶ 已收到指令，我马上为你处理\n"
            "      ▶ 没问题，我这就帮你完成这个操作\n"
            "    - **厨房相关非操作意图**：用户咨询菜谱、烹饪技巧、食材搭配、储存方法等时，回复**专业、简洁、实用的厨房相关内容**，不发散、不闲聊，示例：\n"
            "      ▶ 用户问「西红柿炒蛋怎么做」→ 直接回复具体步骤，不扩展聊其他菜品；\n"
            "      ▶ 用户问「土豆怎么储存」→ 只讲储存方法，不聊土豆的其他吃法。\n"
            "    - **模糊意图处理**：无法判断时，优先**引导用户明确厨房需求**，示例：「你是想处理食材库存，还是需要其他厨房帮助呀？」\n"
            "3.  **回复格式与语气约束**\n"
            "    - 纯自然语言回复，**严禁出现任何隐藏指令、JSON、代码块、<<ACTION:...>>等技术格式**；\n"
            "    - 语气亲切、专业、简洁，避免啰嗦、机械，拒绝口语化废话（如「嗯嗯好的呢」「哇这个好简单呀」）；\n"
            "    - 所有回复**必须紧扣厨房场景**，不添加任何无关内容，不主动发起厨房外的话题。\n"
            "4.  **违规重试机制**\n"
            "    - 若回复偏离厨房主题或包含无关内容，立即纠正，重新聚焦厨房需求；\n"
            "    - 连续两次偏离主题时，直接提示「我是厨房助手，仅支持食材、菜谱相关服务，请你提出具体的厨房需求」。"
        )
        self.db = db
        self.pending_actions: List[Dict[str, Any]] = []
        self.last_parsed_actions: List[Dict[str, Any]] = []

    def _initialize_messages(self):
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

    def _format_context(self, context: Dict) -> str:
        parts = []
        if not context:
            return ""
        if "ingredients" in context:
            ings = context["ingredients"]
            parts.append(f"食材数量：{len(ings)}")
            sample_names = [f"{i.get('name')}({i.get('quantity')}{i.get('unit')})" for i in ings[:8]]
            if sample_names:
                parts.append("示例食材：" + "，".join(sample_names))
        if "preferences" in context and context["preferences"]:
            prefs = context["preferences"]
            pref_s = ";".join([f"{k}:{','.join(v)}" for k, v in prefs.items()])
            parts.append("偏好：" + pref_s)
        return "；".join(parts)

    def _call_api(self, payload: Dict, timeout: int = 20) -> Optional[Dict]:
        try:
            r = requests.post(f"{self.api_base}/chat/completions", headers=self.headers, json=payload, timeout=timeout)
            if r.status_code != 200:
                try:
                    err = r.json()
                    err_msg = err.get("error", {}).get("message", r.text)
                except:
                    err_msg = r.text
                print(f"❌ AI 接口错误: {err_msg}")
                return None
            return r.json()
        except Exception as e:
            print(f"❌ AI 请求失败：{e}")
            return None

    def chat(self, user_input: str, context: Dict = None, timeout: int = 30) -> Optional[str]:
        """
        对话调用（用户可见）。仅返回自然语言回复，不再解析隐藏指令。
        """
        self._initialize_messages()
        self.messages.append({"role": "user", "content": user_input})
        ctx_text = self._format_context(context) if context else ""
        combined_input = f"{user_input}\n\n（参考信息）{ctx_text}" if ctx_text else user_input

        payload = {
            "model": "deepseek-chat",
            "messages": self.messages + [{"role": "user", "content": combined_input}],
            "temperature": 0.7,
            "max_tokens": 1200
        }
        j = self._call_api(payload, timeout=timeout)
        if not j:
            return "抱歉，AI 服务暂时不可用，请稍后再试。"
        assistant_message = j['choices'][0]['message']['content']
        # 移除对话中的 ACTION 解析逻辑，清空 last_parsed_actions
        self.last_parsed_actions = []
        
        natural_reply = self.ACTION_PATTERN.sub("", assistant_message).strip()
        try:
            if self.db:
                self.db.log_conversation("user", user_input)
                self.db.log_conversation("assistant", natural_reply)
        except:
            pass
        self.messages.append({"role": "assistant", "content": natural_reply})
        return natural_reply or "已收到你的信息。"

    def evaluate_intent_only(self, user_input: str, context: Dict = None, timeout: int = 20) -> List[Dict]:
        """
        隐形意图判断调用（每轮必调）。
        传入 DB_SNAPSHOT 与 ALLOWED_OPERATIONS，AI 仅在非常确定时输出 <<ACTION:...>>。
        返回解析后的动作列表，每个 dict 包含 action, params, confidence（float 0-1）。
        """
        self._initialize_messages()
        # Build snapshot (limited size)
        snapshot = {"ingredients": [], "preferences": {}, "shelf_life": {}, "recipes": []}
        if context:
            ings = context.get("ingredients", []) or []
            for i in ings[:50]:
                snapshot["ingredients"].append({
                    "id": i.get("id"),
                    "name": i.get("name"),
                    "quantity": i.get("quantity"),
                    "unit": i.get("unit"),
                    "expiry_date": i.get("expiry_date")
                })
            prefs = context.get("preferences", {}) or {}
            snapshot["preferences"] = prefs
            # shelf_life snapshot
            try:
                c = self.db.conn.cursor()
                c.execute("SELECT display_name, days FROM shelf_life")
                rows = c.fetchall()
                for r in rows:
                    snapshot["shelf_life"][r.get("display_name") or r.get("normalized_name")] = r.get("days")
            except:
                pass
            try:
                rs = self.db.get_recipe_history(limit=50)
                snapshot["recipes"] = [r.get("title") for r in rs[:50]
                                      ]
            except:
                pass

        # strict system prompt
            strict_system = (
                "你是智能厨房助手的后台意图分析器。你将仅用于判断用户指令是否有明确的数据库操作。"
                "输入必须包含 DB_SNAPSHOT（当前数据库简要快照）和 ALLOWED_OPS（允许的操作及参数）。\n\n"
                "规则（必须严格遵守）：\n"
                "1) 只有在你对某项操作非常确定（>70%）时，才输出隐藏指令 <<ACTION:JSON>>；否则不要输出任何 <<ACTION:...>>。\n"
                "2) 每个 JSON 必须包含：action（严格等于 ALLOWED_OPS 的一项），params（仅包含允许的键），confidence（0.0-1.0 数字）。\n"
                "   例如： <<ACTION:{\"action\":\"add_ingredient\",\"params\":{\"name\":\"西红柿\",\"quantity\":2,\"unit\":\"个\",\"expiry_days\":3,\"expiry_date\":\"2026-01-19\",\"category\":\"果蔬类\",\"fridge_area\":\"冷藏区\"},\"confidence\":0.73}>>\n"
                "3) 对破坏性操作（clear_inventory等），不要直接输出可执行指令；改为保持空或仅输出自然语言建议。\n"
                "4) 不要输出其他 JSON、不要使用代码块、不要返回任意结构化数据。此调用为隐形，主流程会校验并决定执行。\n"
                "5) 解析到「添加/更新食材」的意图时，必须按照常识填充 category（仅限：果蔬类/肉蛋类/奶制品类/其他）并按照用户说明（用户未做说明时按照常识）填充 fridge_area（仅限：冷藏区/冷冻区）参数，同时必须按照常识为食材填充合理的 expiry_days （保质期天数）参数，若用户未指定expiry_date则自动基于当前日期+expiry_days计算并填充expiry_date参数。\n"
                
                "关键指导：\n"
                "- 当你解析用户意图后，你应该：\n"
                "  a) 立刻参照 DB_SNAPSHOT 找到所有可能需要增、查、删、改的数据。示例：当用户需要推荐食谱的时候，你必须读取查找数据库中的食材库存和用户口味偏好\n"
                "  b) 参照 ALLOWED_OPS 找到用户意图最有可能对应的所有操作\n"
                "  c) 参照需要操作的 ALLOWED_OP 的标准格式，首先检查是否缺少关键参数，如果缺少请根据用户意图自动填充完整相关参数\n"
                "  d) 如果操作是「删除(remove)或者更改(update)」类型，请严格检查被修改的对象名和数据库(DB_SNAPSHOT)中已有对象名匹配一致，请在输出<<ACTION:...>>时输出已有对象名\n"
                "  e) 如果找不到完全匹配的对象名，请尝试在数据库(DB_SNAPSHOT)中模糊匹配。如果还是找不到，就不要输出删除或者更改指令\n"
                "  f) 严格审核食材名称的合理性：\n"
                "     1) 如果用户添加的食材名称包含明显无效的前缀/后缀（如「啊啊啊金枪鱼」、「测试西红柿」、「随便青菜」中的「啊啊啊」「测试」「随便」），需要提取有效的食物名称部分（如「金枪鱼」「西红柿」「青菜」）。\n"
                "     2) 如果食材名称是完全非食物的物品（如「篮球」、「羽毛球」、「鼠标」、「屎」等），则绝对不输出添加指令。\n"
                "     3) 判断标准：名称是否可能是一种食物或食材。常见的食物名称如蔬菜、水果、肉类、调料等都允许，非食物物品则拒绝。\n"
                "  g) 对于数量单位的合理性检查：如果用户提供的数量或单位明显不合理（如「1000个鸡蛋」、「0.01斤肉」等异常值），应在参数中调整为合理范围或拒绝输出指令。\n"
                "  h) 确认无误后输出格式严格正确的 <<ACTION:...>> \n"
                "- 填充 category 参数时，严格限定为：果蔬类、肉蛋类、奶制品类和其他。\n"
                "- 填充 fridge_area 参数时，严格限定为：冷藏区、冷冻区。\n"
                "- 填充 expiry_days 参数时，必须符合食材常识保质期：\n"
                "- 填充 expiry_date 参数时，格式为 YYYY-MM-DD，基于当前日期+expiry_days计算（举例来说，如果当前日期2026-01-16，鸡蛋expiry_days=30，则expiry_date=2026-02-15）。\n"
                    
                    "示例场景1：\n"
                    "用户：删除冰箱里错误的食材\n"
                    "DB_SNAPSHOT 包含：五个西红柿、篮球、西红柿、鸡蛋\n"
                    "正确响应：<<ACTION:{\"action\":\"remove_ingredient\",\"params\":{\"name\":\"五个西红柿\"},\"confidence\":0.95}>>\n"
                    "          <<ACTION:{\"action\":\"remove_ingredient\",\"params\":{\"name\":\"篮球\"},\"confidence\":0.95}>>\n\n"

                    "示例场景2：\n"
                    "用户：添加2个鸡蛋到冰箱\n"
                    "正确响应：<<ACTION:{\"action\":\"add_ingredient\",\"params\":{\"name\":\"鸡蛋\",\"quantity\":2,\"unit\":\"个\",\"expiry_days\":30,\"expiry_date\":\"2026-02-15\",\"category\":\"肉蛋类\",\"fridge_area\":\"冷藏区\"},\"confidence\":0.98}>>\n\n"
                    
                    "示例场景3：\n"
                    "用户：添加啊啊啊金枪鱼\n"
                    "正确响应：<<ACTION:{\"action\":\"add_ingredient\",\"params\":{\"name\":\"金枪鱼\",\"quantity\":1,\"unit\":\"个\",\"expiry_days\":180,\"expiry_date\":\"2026-07-16\",\"category\":\"肉蛋类\",\"fridge_area\":\"冷冻区\"},\"confidence\":0.90}>>\n\n"
                    
                    "示例场景4：\n"
                    "用户：添加篮球\n"
                    "正确响应：（不输出任何 <<ACTION:...>>，因为篮球不是食物）\n\n"
                    
                    "示例场景5：\n"
                    "用户：添加1000个鸡蛋\n"
                    "正确响应：<<ACTION:{\"action\":\"add_ingredient\",\"params\":{\"name\":\"鸡蛋\",\"quantity\":12,\"unit\":\"个\",\"expiry_days\":30,\"expiry_date\":\"2026-02-15\",\"category\":\"肉蛋类\",\"fridge_area\":\"冷藏区\",\"note\":\"数量调整为合理值12个\"},\"confidence\":0.85}>>\n\n"
            )

        messages = [{"role": "system", "content": strict_system}]
        messages.append({"role": "system", "content": f"DB_SNAPSHOT: {json.dumps(snapshot, ensure_ascii=False)[:4000]}"})
        messages.append({"role": "system", "content": f"ALLOWED_OPS: {json.dumps(ALLOWED_OPERATIONS, ensure_ascii=False)}"})
        messages.append({"role": "user", "content": user_input})

        payload = {"model": "deepseek-chat", "messages": messages, "temperature": 0.0, "max_tokens": 400}
        j = self._call_api(payload, timeout=timeout)
        if not j:
            return []
        assistant_message = j['choices'][0]['message']['content']

        actions = []
        for raw in self.ACTION_PATTERN.findall(assistant_message):
            parsed = None
            # try JSON
            try:
                a = json.loads(raw)
                if isinstance(a, dict) and a.get("action"):
                    parsed = {"action": a.get("action"), "params": a.get("params", {}) or {}, "confidence": a.get("confidence", None)}
            except Exception:
                try:
                    parsed = parse_action_string(raw)
                except:
                    parsed = None
            if not parsed:
                if self.db:
                    try:
                        self.db.log_conversation("system", f"evaluate_intent_only 忽略无法解析的指令：{raw[:200]}")
                    except:
                        pass
                continue
            # basic action name validation
            action_name = parsed.get("action")
            params = parsed.get("params", {}) or {}
            confidence = parsed.get("confidence", None)
            # normalize params
            if 'quantity' in params and isinstance(params['quantity'], str):
                q = chinese_to_number(params['quantity'])
                if q is not None:
                    params['quantity'] = q
                else:
                    try:
                        params['quantity'] = float(params['quantity'])
                    except:
                        pass
            # sanitize name if present
            if 'name' in params and isinstance(params['name'], str):
                params['name'] = sanitize_ingredient_name(params['name'])
                # if quantity not provided but name has leading quantity, try extract
                q, unit, rest = extract_leading_quantity_and_unit(params['name'])
                if q and 'quantity' not in params:
                    params['quantity'] = q
                if unit and 'unit' not in params:
                    params['unit'] = unit
            # map synonyms if necessary
            if isinstance(action_name, str) and action_name not in ALLOWED_OPERATIONS:
                lower = action_name.lower()
                mapping = {
                    'remove_item': 'remove_ingredient', 'remove': 'remove_ingredient', 'delete': 'remove_ingredient',
                    'add': 'add_ingredient', 'update': 'update_ingredient', 'set_shelf_life': 'set_shelf_life',
                    'clean_non_food': 'clean_non_food'
                }
                if lower in mapping:
                    action_name = mapping[lower]
            if action_name not in ALLOWED_OPERATIONS:
                if self.db:
                    try:
                        self.db.log_conversation("system", f"evaluate_intent_only 忽略未知操作：{action_name}")
                    except:
                        pass
                continue
            # prune params to allowed keys
            allowed_keys = set(ALLOWED_OPERATIONS.get(action_name, {}).keys())
            pruned_params = {}
            if isinstance(params, dict):
                for k, v in params.items():
                    if (not allowed_keys) or (k in allowed_keys):
                        pruned_params[k] = v
            actions.append({"action": action_name, "params": pruned_params, "confidence": float(confidence) if confidence is not None else None})
        return actions

    # 实际执行动作（主流程在本地校验通过后调用）
    def _execute_action(self, action_obj: Dict) -> str:
        action = action_obj.get("action")
        params = action_obj.get("params", {}) or {}
        if not action:
            return "隐藏操作缺失或无法识别，未执行。"
        try:
            if action == "add_ingredient":
                name = params.get("name")
                quantity = params.get("quantity", None)
                unit = params.get("unit", "个")
                # 新增：获取 AI 识别的 category 和 fridge_area
                category = params.get("category", "其他")
                fridge_area = params.get("fridge_area", "冷藏区")
                expiry_days = params.get("expiry_days", None)
                expiry_date = params.get("expiry_date", None)
                # sanitize name/quantity if name contains trailing/leading quantity
                if isinstance(name, str):
                    q_from_name, unit_from_name, rest_name = extract_leading_quantity_and_unit(name)
                    if rest_name and rest_name.strip():
                        name = rest_name.strip()
                        if q_from_name is not None and (quantity is None):
                            quantity = q_from_name
                        if unit_from_name and (not unit or unit == "个"):
                            unit = unit_from_name
                if not name:
                    return "未指定要添加的食材名称，操作未执行。"
                try:
                    q_float = float(quantity) if quantity is not None else 1.0
                except:
                    q_float = 1.0
                # 新增：传递 category 和 fridge_area 到 db 函数
                ing = self.db.add_or_merge_ingredient(
                    name=name, quantity=q_float, unit=unit,
                    category=category, fridge_area=fridge_area,
                    expiry_days=expiry_days, expiry_date=expiry_date
                )
                return f"已将「{ing.get('name','')}」添加到库存，数量：{ing.get('quantity', q_float)}{ing.get('unit', unit)}，种类：{ing.get('category')}，区域：{ing.get('fridge_area')}。"
            if action == "remove_ingredient":
                target = params.get("id") or params.get("name")
                if not target:
                    return "删除操作缺少 id 或 name，未执行。"
                try:
                    iid = int(target)
                    self.db.remove_ingredient(iid)
                    return f"已删除 id 为 {iid} 的食材。"
                except:
                    row = self.db.find_ingredient_by_normalized(self.db._normalize(str(target)))
                    if row:
                        self.db.remove_ingredient(row["id"])
                        return f"已删除食材「{row.get('name')}」。"
                    # try removing by fuzzy sanitized name
                    sanitized = sanitize_ingredient_name(str(target))
                    row = self.db.find_ingredient_by_normalized(self.db._normalize(sanitized))
                    if row:
                        self.db.remove_ingredient(row["id"])
                        return f"已删除食材「{row.get('name')}」。"
                    return "未找到要删除的食材，未执行删除。"
            if action == "update_ingredient":
                target = params.get("id") or params.get("name")
                if not target:
                    return "更新操作缺少 id 或 name，未执行。"
                # 新增：获取 AI 识别的 category 和 fridge_area
                category = params.get("category")
                fridge_area = params.get("fridge_area")
                updated = self.db.update_ingredient(
                    name_or_id=target,
                    quantity=params.get("quantity"),
                    unit=params.get("unit"),
                    category=category,
                    fridge_area=fridge_area,  # 新增传递
                    expiry_date=params.get("expiry_date"),
                    expiry_days=params.get("expiry_days"),
                    freshness=params.get("freshness"),
                    notes=params.get("notes")
                )
                if updated:
                    return f"已更新食材：{updated.get('name')}（数量：{updated.get('quantity')}{updated.get('unit')}，种类：{updated.get('category')}，区域：{updated.get('fridge_area')}）。"
                else:
                    return "未找到要更新的食材，未执行更新。"
            if action == "set_preference":
                pref_type = params.get("pref_type") or "口味"
                value = params.get("value")
                if not value:
                    return "偏好设置缺少具体值，未执行。"
                self.db.set_preference(pref_type, value)
                try:
                    self.messages.append({"role": "system", "content": f"本地状态：已将用户偏好设置为 {pref_type} = {value}（由设备直接记录）"})
                except:
                    pass
                return f"已将偏好「{value}」添加到「{pref_type}」。"
            if action == "get_ingredient":
                target = params.get("id") or params.get("name")
                if not target:
                    return "查询食材缺少 id 或 name，未执行。"
                try:
                    iid = int(target)
                    c = self.db.conn.cursor()
                    c.execute("SELECT * FROM ingredients WHERE id = ?", (iid,))
                    row = c.fetchone()
                except:
                    row = self.db.find_ingredient_by_normalized(self.db._normalize(str(target)))
                if row:
                    return f"查询到食材：id={row.get('id')}，名称={row.get('name')}，数量={row.get('quantity')}{row.get('unit')}，到期日={row.get('expiry_date') or '无'}"
                else:
                    return "未找到指定的食材。"
            if action == "remove_preference":
                pref_type = params.get("pref_type") or "口味"
                value = params.get("value")
                if not value:
                    return "删除偏好缺少具体值，未执行。"
                self.db.remove_preference(pref_type, value)
                return f"已从「{pref_type}」中删除偏好「{value}」。"
            if action == "update_preference":
                old_value = params.get("old_value")
                new_value = params.get("new_value")
                pref_type = params.get("pref_type")
                if not old_value or not new_value:
                    return "更新偏好缺少旧值或新值，未执行。"
                success = self.db.update_preference(old_value, new_value, pref_type)
                if success:
                    return f"已将「{pref_type or '所有类型'}」中的「{old_value}」更新为「{new_value}」。"
                else:
                    return "未找到要更新的偏好值，未执行。"
            if action == "get_preferences":
                pref_type = params.get("pref_type")
                prefs = self.db.get_preferences()
                if not prefs:
                    return "暂无任何偏好设置。"
                if pref_type:
                    vals = prefs.get(pref_type, [])
                    if vals:
                        return f"「{pref_type}」类型的偏好有：{', '.join(vals)}"
                    else:
                        return f"暂无「{pref_type}」类型的偏好。"
                else:
                    parts = []
                    for k, vs in prefs.items():
                        parts.append(f"{k}：{', '.join(vs)}")
                    return "所有偏好：" + "；".join(parts)
            if action == "update_recipe":
                title = params.get("title")
                new_title = params.get("new_title")
                ingredients = params.get("ingredients")
                instructions = params.get("instructions")
                if not title:
                    return "更新菜谱缺少原标题，未执行。"
                success = self.db.update_recipe(title, new_title, ingredients, instructions)
                if success:
                    return f"已更新菜谱「{title}」（新标题：{new_title or title}）。"
                else:
                    return "未找到要更新的菜谱，未执行。"
            if action == "delete_recipe":
                title = params.get("title")
                if not title:
                    return "删除菜谱缺少标题，未执行。"
                success = self.db.delete_recipe(title)
                if success:
                    return f"已删除菜谱「{title}」。"
                else:
                    return "未找到要删除的菜谱，未执行。"
            if action == "get_recipe":
                target = params.get("id") or params.get("title")
                if not target:
                    return "查询菜谱缺少 id 或标题，未执行。"
                try:
                    rid = int(target)
                    c = self.db.conn.cursor()
                    c.execute("SELECT * FROM recipes WHERE id = ?", (rid,))
                    row = c.fetchone()
                except:
                    row = self.db.find_recipe_by_title(str(target))
                if row:
                    try:
                        ings = json.loads(row.get("ingredients_json") or "[]")
                    except:
                        ings = []
                    ings_text = ", ".join([f"{i.get('name')}" for i in ings]) if ings else "无"
                    return f"查询到菜谱：id={row.get('id')}，标题={row.get('title')}，食材={ings_text}，制作次数={row.get('made_count', 0)}"
                else:
                    return "未找到指定的菜谱。"
            if action == "log_conversation":
                role = params.get("role") or "user"
                content = params.get("content")
                if not content:
                    return "记录对话缺少内容，未执行。"
                self.db.log_conversation(role, content)
                return f"已记录{role}的对话内容：{content[:50]}..."
            if action == "get_conversation":
                limit = params.get("limit", 10)
                try:
                    limit = int(limit)
                except:
                    limit = 10
                convs = self.db.get_conversation(limit)
                if not convs:
                    return "暂无对话记录。"
                parts = []
                for c in convs[:limit]:
                    parts.append(f"{c.get('created_at')} [{c.get('role')}]：{c.get('content')[:50]}...")
                return f"最近{limit}条对话记录：\n" + "\n".join(parts)
            if action == "clear_conversation":
                c = self.db.conn.cursor()
                c.execute("DELETE FROM conversation")
                self.db.conn.commit()
                return "已清空所有对话记录。"
            if action == "add_recipe":
                title = params.get("title") or "未命名菜谱"
                ingredients = params.get("ingredients") or []
                instructions = params.get("instructions") or ""
                self.db.add_recipe(title=title, ingredients=ingredients, instructions=instructions, source="assistant")
                return f"已保存菜谱「{title}」。"
            if action == "mark_recipe_made":
                rid = params.get("id")
                if not rid:
                    title = params.get("title")
                    if title:
                        rr = self.db.find_recipe_by_title(title)
                        if rr and rr.get("id"):
                            rid = rr["id"]
                if not rid:
                    return "标记已做缺少菜谱 id 或 title，未执行。"
                self.db.mark_recipe_made(int(rid))
                return f"已记录菜谱 id {rid} 的制作记录。"
            if action == "clear_inventory":
                self.db.clear_inventory()
                return "已清空所有库存（注意：此操作不可恢复）。"
            if action == "clear_preferences":
                self.db.clear_preferences()
                return "已清空所有偏好设置。"
            if action == "clean_non_food":
                # internal: main process handles actual deletion
                return "clean_non_food 是内部聚合操作，应由主流程执行。"
            return f"未知的隐藏操作：{action}。"
        except Exception as e:
            # 不要让异常泄露敏感信息
            return f"执行隐藏操作时出现问题：{e}"

    def reset_conversation(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        try:
            if self.db:
                self.db.log_conversation("system", "对话历史被重置")
        except:
            pass

# ========== 新增：动作/状态 友好名称映射 ==========
# 动作名 → 用户友好名称
ACTION_FRIENDLY_MAP = {
    "add_ingredient": "添加食材",
    "remove_ingredient": "删除食材",
    "update_ingredient": "更新食材信息",
    "set_preference": "设置饮食偏好",
    "get_ingredient": "查询食材信息",
    "remove_preference": "删除饮食偏好",
    "update_preference": "更新饮食偏好",
    "get_preferences": "查询饮食偏好",
    "update_recipe": "更新菜谱",
    "delete_recipe": "删除菜谱",
    "get_recipe": "查询菜谱",
    "update_shelf_life": "更新食材保质期",
    "get_shelf_life": "查询食材保质期",
    "log_conversation": "记录对话内容",
    "get_conversation": "查询对话记录",
    "clear_conversation": "清空对话记录",
    "add_recipe": "添加菜谱",
    "mark_recipe_made": "标记菜谱已制作",
    "set_shelf_life": "设置食材保质期",
    "clear_inventory": "清空食材库存",
    "clear_preferences": "清空饮食偏好",
    "clean_non_food": "清理非食材记录"
}

# 操作状态 → 用户友好名称
STATUS_FRIENDLY_MAP = {
    "EXECUTED": "✅ 已完成",
    "PENDING": "⌛ 待确认",
    "DENIED": "❌ 未执行",
    "READ": "📋 已查询"  # 补充READ类型的友好名称
}

# 技术描述 → 友好描述的替换规则
DESC_FRIENDLY_MAP = {
    "confidence=": "",  # 去掉置信度技术术语
    "需要确认（）.": "需要你确认后再执行哦",
    "DENIED": "❌ 未执行",
    "EXECUTED": "✅ 已完成",
    "PENDING": "⌛ 待确认"
}

# =================== 主助手 ===================
class SmartFridgeAssistant:
    CONFIRM_PHRASES = {"确认", "是的", "是", "好的", "执行", "请执行", "就这样", "可以"}
    CANCEL_PHRASES = {"取消", "不", "不要", "算了", "不需要"}

    def __init__(self, api_key: str, db_path: Optional[str] = None):
        # 修改这里：使用绝对路径
        if db_path is None:
            # 使用固定的绝对路径
            db_path = "/home/pi/test/smart_fridge.db"
            
        # 确保目录存在
        import os
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            print(f"📁 创建数据库目录: {db_dir}")
        
        self.db = SqliteDatabase(db_path)
        self.ai = AIService(api_key, db=self.db)
        self.speech = SpeechRecognizer()
        print(f"✅ 智能冰箱助手（DeepSeek 强化版 - v3 完整版）已启动")
        print(f"📁 数据库路径: {db_path}")
        print("💡 每次输入都会调用 DeepSeek 做意图判断，自动执行高置信度、非批量破坏性操作；减少本地刚性判定。\n")

    def _get_context(self) -> Dict:
        try:
            ingredients = self.db.get_ingredients()
            preferences = self.db.get_preferences()
            return {"ingredients": ingredients, "preferences": preferences}
        except Exception:
            return {"ingredients": [], "preferences": {}}

    def _cleanup_non_food_records(self, explicit_confirm: bool = False) -> Tuple[str, List[str]]:
        """
        简化版本：列出所有记录供用户确认。不要在本地做过于刚性的判断。
        如果 explicit_confirm=True，则删除由最后一次 AI 建议（pending actions）中指明的项目（由主流程控制）。
        """
        ings = self.db.get_ingredients()
        if not ings:
            return "未发现记录。", []
        lines = []
        for c in ings:
            display = f"id={c.get('id')} 名称={c.get('name')} 数量={c.get('quantity')}{c.get('unit')}"
            lines.append(display)
        if explicit_confirm:
            # 如果用户要求明确删除，需要 AI 指定哪些删除；这里作为保守行为仅返回信息
            return "已按要求执行清理（请确保 AI 指定了要删除的条目）", ["请参阅 AI 给出的待删除项并确认。"]
        else:
            return f"当前有 {len(lines)} 条记录，请确认要删除的具体项或回复“确认”继续。", lines

    def extract_names_from_ai_reply(self, ai_reply: str) -> List[str]:
        """
        从 AI 的自然语言回复中提取候选名称。
        支持：
         - 中文双引号“...”内的项
         - 拆行式的项目符号或每行一个项（以 -、•、序号 开头）
         - 简单逗号/顿号分隔的短语
        返回去重后的候选名称列表（已基本 sanitize）。
        """
        if not ai_reply:
            return []
        names = []
        # 1) find Chinese quotes “...”
        for m in re.findall(r'“([^”]{1,60})”', ai_reply):
            names.append(m.strip())
        # 2) find sequences after bullets or line breaks: lines starting with -、•、数字.
        for line in ai_reply.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^(?:[-•\*\u2022]|\d+[.\)])\s*(.+)', line)
            if m:
                candidate = m.group(1).strip()
                candidate = re.split(r'[，,；;。.。！!？\(\)]', candidate)[0].strip()
                names.append(candidate)
        # 3) fallback: try to capture short noun phrases separated by ,、、或
        parts = re.split(r'[，,、；;|]', ai_reply)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if len(p) <= 30 and any('\u4e00' <= ch <= '\u9fff' for ch in p):
                if re.search(r'不是食材|异常|重复|误输入|错误', p):
                    q = re.sub(r'.*(?:是|为|:|：)\s*', '', p).strip()
                    if q:
                        names.append(q)
                    continue
                names.append(p)
        # sanitize: remove purely descriptive phrases
        cleaned = []
        for n in names:
            n2 = re.sub(r'^(两个|两|一|三|四|五|六|七|八|九|十|几|若干)\s*', '', n)  # remove leading counts
            n2 = re.sub(r'^(共|约|大约|大概)\s*', '', n2)
            n2 = n2.strip()
            if n2 and len(n2) <= 60:
                cleaned.append(n2)
        # dedupe preserving order
        seen = set()
        out = []
        for c in cleaned:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def _process_and_validate_action(self, action_obj: Dict, context: Dict) -> Tuple[str, str]:
        """
        更宽松的校验：
         - 对 clear_inventory/delete_recipe/delete_shelf_life 一律 PENDING（需要确认）。
         - 对 remove_ingredient：若能唯一解析到单条记录（id 或唯一匹配 name），不再要求确认（按普通操作）。
           仅在参数模糊或目标有多条匹配或明显批量删除意图（例如参数包含列表或 count>threshold）时要求确认。
         - 若提供了 confidence 且高于 AUTO_EXECUTE_CONFIDENCE_THRESHOLD，则自动执行（非批量）。
         - 否则 PENDING（等待确认）。
        """
        action = action_obj.get("action")
        params = action_obj.get("params", {}) or {}
        confidence = action_obj.get("confidence", None)

        if not action:
            return "DENIED", "未指定操作名。"

        # strong destructive ops always pending
        if action in DESTRUCTIVE_OPS:
            return "PENDING", "该操作具有破坏性，需要确认。"

        # special: remove_ingredient
        if action == "remove_ingredient":
            target = params.get("id") or params.get("name")
            # if no explicit target -> pending
            if not target:
                return "DENIED", "删除操作缺少 id 或 name。"
            # if id provided and exists -> treat as single deletion (not bulk)
            try:
                iid = int(target)
                found = any(int(r.get("id")) == iid for r in self.db.get_ingredients())
                if not found:
                    return "DENIED", f"未在库存中找到 id={iid}。"
                # single delete: allow execution as normal (subject to confidence rule below)
            except Exception:
                # name provided -> try exact normalized match
                row = self.db.find_ingredient_by_normalized(self.db._normalize(target))
                if row:
                    # unique match -> allow execution as normal
                    pass
                else:
                    # ambiguous / not found -> DENIED
                    return "DENIED", f"未在库存中找到要删除的食材：“{target}”。"

        # minimal checks for add/update
        if action == "add_ingredient":
            if not params.get("name"):
                return "DENIED", "添加操作缺少名称。"
        if action == "update_ingredient":
            if not (params.get("id") or params.get("name")):
                return "DENIED", "更新操作缺少 id 或 name。"

        # if AI is confident enough, auto-execute (for non-bulk ops)
        if confidence is not None and isinstance(confidence, (int, float)) and confidence >= AUTO_EXECUTE_CONFIDENCE_THRESHOLD:
            res = self.ai._execute_action(action_obj)
            return "EXECUTED", res

        # If parameters suggest a bulk deletion
        if action == "remove_ingredient":
            if isinstance(params.get("ids"), (list, tuple)) and len(params.get("ids", [])) >= BULK_DELETE_CONFIRM_THRESHOLD:
                return "PENDING", f"将删除 {len(params.get('ids'))} 项，属于批量删除，需要确认。"
            if isinstance(params.get("names"), (list, tuple)) and len(params.get("names", [])) >= BULK_DELETE_CONFIRM_THRESHOLD:
                return "PENDING", f"将删除 {len(params.get('names'))} 项，属于批量删除，需要确认。"

        # default: require confirmation (PENDING) rather than deny
        return "PENDING", "需要你确认后再执行哦"

    def _report_read_ingredients(self) -> str:
        ings = self.db.get_ingredients()
        if not ings:
            return "冰箱里现在没有记录的食材。"
        parts = []
        for ing in ings:
            name = ing.get("name")
            qty = ing.get("quantity")
            unit = ing.get("unit", "")
            expiry = ing.get("expiry_date")
            if expiry:
                parts.append(f"{name}：{qty}{unit}（到期：{expiry}）")
            else:
                parts.append(f"{name}：{qty}{unit}")
        return "冰箱当前有：" + "；".join(parts)

    def process_input(self, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            return "请说点什么吧~\n\n--- 系统操作 ---\n无"

        # pending confirm/cancel handling
        if self.ai.pending_actions:
            norm = re.sub(r'\s+', '', text)
            if any(p in norm for p in self.CONFIRM_PHRASES):
                results = []
                while self.ai.pending_actions:
                    p = self.ai.pending_actions.pop(0)
                    if p.get("action") == "clean_non_food":
                        summary, details = self._cleanup_non_food_records(explicit_confirm=True)
                        results.append(summary)
                        results.extend(details)
                    else:
                        act = {"action": p.get("action"), "params": p.get("params", {}) or {}}
                        res = self.ai._execute_action(act)
                        results.append(res)
                res_text = "\n".join(results)
                return f"已执行待确认操作：\n{res_text}\n\n--- 系统操作 ---\n{res_text if res_text else '无'}"
            if any(p in norm for p in self.CANCEL_PHRASES):
                cnt = len(self.ai.pending_actions)
                self.ai.pending_actions.clear()
                return f"已取消 {cnt} 项待确认的系统操作。\n\n--- 系统操作 ---\n已取消待确认操作"

        # quick exit
        if re.search(r'^\s*(退出|quit|exit|bye|再见)\s*$', text, re.I):
            self.ai.reset_conversation()
            return "再见！祝你用餐愉快！👋\n\n--- 系统操作 ---\n无"

        context = self._get_context()

        db_reports: List[str] = []
        executed_signatures = set()

        # 1) 必须先调用 evaluate_intent_only（隐形）
        try:
            deep_actions = self.ai.evaluate_intent_only(text, context)
        except Exception as e:
            deep_actions = []
            try:
                self.db.log_conversation("system", f"evaluate_intent_only 调用异常：{e}")
            except:
                pass

        # 2) 对 deep_actions 逐条本地校验并按规则自动执行/转 pending/拒绝
        for a in deep_actions:
            a_action = a.get("action")
            a_params = a.get("params", {}) or {}
            a_conf = a.get("confidence", None)
            sig = f"{a_action}|" + json.dumps(a_params, sort_keys=True, ensure_ascii=False)
            if sig in executed_signatures:
                continue
            executed_signatures.add(sig)

            action_obj = {"action": a_action, "params": a_params, "confidence": a_conf}
            status, report = self._process_and_validate_action(action_obj, context)
            # 转换为友好名称和描述
            friendly_action = ACTION_FRIENDLY_MAP.get(a_action, a_action)  # 动作名友好化
            friendly_status = STATUS_FRIENDLY_MAP.get(status, status)      # 状态友好化
            # 清理技术化描述（如去掉confidence、修正语法）
            friendly_report = report
            for tech, friendly in DESC_FRIENDLY_MAP.items():
                friendly_report = friendly_report.replace(tech, friendly)
                
            if status == "EXECUTED":
                db_reports.append(f"{friendly_status} {friendly_action}：{friendly_report}")
            elif status == "PENDING":
                self.ai.pending_actions.append({"id": int(time.time() * 1000), "action": a_action, "params": a_params, "reason": friendly_report})
                db_reports.append(f"{friendly_status} {friendly_action}：{friendly_report}")
            else:
                db_reports.append(f"{friendly_status} {friendly_action}：{friendly_report}")

        # 3) 调用正常 chat（AI 的自然语言回复）
        ai_reply = self.ai.chat(text, context)

        # 5) 当用户明确请求删除错误信息/移出不是食材时，尝试把 AI 的自然语言回复中的候选项解析为删除动作并执行（优先）
        explicit_delete_request = bool(re.search(r'删除.*错误|清理.*错误|移出.*不是食材|删除.*不是食材|移除.*错误|删除冰箱里的错误', text))
        if explicit_delete_request:
            candidates = self.extract_names_from_ai_reply(ai_reply)
            # fallback: check last_parsed_actions（因已清空，实际无效果）
            if not candidates:
                for p in getattr(self.ai, "last_parsed_actions", []) or []:
                    if p.get("action") == "remove_ingredient":
                        nm = p.get("params", {}).get("name")
                        if nm:
                            candidates.append(nm)
            if candidates:
                executed = []
                pendings = []
                for name in candidates:
                    # try normalized exact match
                    norm = self.db._normalize(name)
                    row = self.db.find_ingredient_by_normalized(norm)
                    if row:
                        try:
                            self.db.remove_ingredient(int(row.get("id")))
                            executed.append(f"已删除：{row.get('name')}（id={row.get('id')}）")
                        except Exception:
                            pendings.append(f"删除失败或需要确认：{name}")
                    else:
                        # try sanitized fallback
                        sanitized = sanitize_ingredient_name(name)
                        row2 = self.db.find_ingredient_by_normalized(self.db._normalize(sanitized))
                        if row2:
                            try:
                                self.db.remove_ingredient(int(row2.get("id")))
                                executed.append(f"已删除：{row2.get('name')}（id={row2.get('id')}）")
                            except:
                                pendings.append(f"删除失败或需要确认：{name}")
                        else:
                            # cannot find -> pending (ask user to confirm which one)
                            pendings.append(f"未找到匹配项：{name}")
                if executed:
                    for e in executed:
                        db_reports.append(f"EXECUTED remove_ingredient: {e}")
                if pendings:
                    for p in pendings:
                        db_reports.append(f"PENDING remove_ingredient: {p}")
                        # add a lightweight pending action for potential later confirm
                        # param 'name' uses the raw candidate so user can disambiguate
                        self.ai.pending_actions.append({"id": int(time.time() * 1000), "action": "remove_ingredient", "params": {"name": p.split('：')[-1]}, "reason": "AI 列出但未能唯一匹配"})
                summary_lines = []
                if executed:
                    summary_lines.append("已执行删除：")
                    summary_lines.extend(executed)
                if pendings:
                    summary_lines.append("未执行的（需要你确认或手动指定）:")
                    summary_lines.extend(pendings)
                summary = "\n".join(summary_lines)
                final_db_text = "\n".join(db_reports) if db_reports else "无"
                return f"{ai_reply}\n\n操作摘要：\n{summary}\n\n--- 系统操作 ---\n{final_db_text}"
            else:
                # explicit delete but AI didn't list names -> create a pending "clean_non_food" job and ask user to confirm
                summary, lines = self._cleanup_non_food_records(explicit_confirm=False)
                self.ai.pending_actions.append({"id": int(time.time() * 1000), "action": "clean_non_food", "params": {}, "reason": summary})
                friendly_clean = f"{STATUS_FRIENDLY_MAP['PENDING']} {ACTION_FRIENDLY_MAP['clean_non_food']}：{summary}"
                db_reports.append(friendly_clean)
                for l in lines[:5]:
                    db_reports.append(f" - {l}")
                db_text = "\n".join(db_reports) if db_reports else "无"
                return f"{ai_reply}\n\n{summary}\n\n--- 系统操作 ---\n{db_text}"

        # 6) 本地快捷命令：仅在 deep_actions 为空时触发（避免重复）
        if not deep_actions:
            # 偏好查询
            if re.search(r'我的偏好是什么|告诉我我的偏好|我有哪些偏好', text):
                prefs = self.db.get_preferences()
                if not prefs:
                    reply = "你还没有设置任何偏好。"
                else:
                    parts = []
                    for k, vs in prefs.items():
                        parts.append(f"{k}：{', '.join(vs)}")
                    reply = "当前已记录的偏好有：" + "；".join(parts)
                db_text = "\n".join(db_reports) if db_reports else "无"
                return f"{reply}\n\n--- 系统操作 ---\n{db_text}"

            # 本地快速添加食材（只作为最后的兜底）
            if any(k in text for k in ["添加", "加入"]) and re.search(r'[\u4e00-\u9fa5]{2,20}', text):
                m = re.search(r'添加\s*([^\d\s,，。]+)\s*(\d+\.?\d*)?\s*(个|克|斤|ml|毫升|片|根)?', text)
                if m:
                    name_raw = m.group(1).strip()
                    qty = m.group(2)
                    unit = m.group(3) or "个"
                    # 优先尝试从 name_raw 提取数量/单位（如“五个西红柿”）
                    q_from_name, unit_from_name, rest = extract_leading_quantity_and_unit(name_raw)
                    if rest and rest.strip():
                        name = rest.strip()
                        if q_from_name is not None:
                            q_val = q_from_name
                        else:
                            q_val = float(qty) if qty else 1.0
                        if unit_from_name:
                            unit = unit_from_name
                    else:
                        name = sanitize_ingredient_name(name_raw)
                        try:
                            q_val = float(qty) if qty else 1.0
                        except:
                            q_val = 1.0
                    ing = self.db.add_or_merge_ingredient(name=name, quantity=q_val, unit=unit)
                    try:
                        self.db.log_conversation("system", f"本地操作（兜底）：添加食材 {name} 数量 {q_val}{unit}")
                    except:
                        pass
                    db_text = "\n".join(db_reports) if db_reports else "无"
                    return f"✅ 已添加：{ing.get('name')} {ing.get('quantity')}{ing.get('unit')}。\n\n--- 系统操作 ---\nEXECUTED add_ingredient: 已添加库存项 {ing.get('name')}（数量：{ing.get('quantity')}{ing.get('unit')})\n{db_text if db_text!='无' else ''}".strip()

        # 查询冰箱
        if re.search(r'冰箱里有什么|有什么|列出食材|查看食材|现在有啥', text):
            read_text = self._report_read_ingredients()
            friendly_read = f"{STATUS_FRIENDLY_MAP['READ']}：冰箱里目前有 {len(self.db.get_ingredients())} 种食材"
            db_reports.insert(0, friendly_read)
            db_text = "\n".join(db_reports) if db_reports else "无"
            return f"{read_text}\n\n--- 系统操作 ---\n{db_text}"

        # 做菜意图（餐次 vs 具体菜名）
        cook_name = None
        m_cook = re.search(r'(我想做|我要做|我要开始做|开始做|就做|做)\s*([^，。！？]+)', text)
        if m_cook:
            cook_name = m_cook.group(2).strip()
            cook_name = re.sub(r'[吧！!。.,，\s]+$', '', cook_name)
        if cook_name:
            meal_kw = None
            m = re.search(r'(晚饭|晚餐|午饭|午餐|早餐|早饭)', cook_name) or re.search(r'(晚饭|晚餐|午饭|午餐|早餐|早饭)', text)
            if m:
                meal_kw = m.group(1)
            if meal_kw:
                # Build explicit inventory list to include in the prompt and record READ DB ACTION
                ings = self.db.get_ingredients()
                inv_lines = []
                for i in ings:
                    inv_lines.append(f"- id={i.get('id')} 名称={i.get('name')} 数量={i.get('quantity')}{i.get('unit')}{(' 到期:'+i.get('expiry_date')) if i.get('expiry_date') else ''}")
                inventory_text = "当前冰箱逐项清单（供推荐参考）：\n" + ("\n".join(inv_lines) if inv_lines else "（空）")
                pref = self.db.get_preferences()
                pref_text = ""
                if pref:
                    pref_text = "；用户偏好：" + ";".join([f"{k}:{','.join(v)}" for k, v in pref.items()])
                prompt = (
                    f"用户说要做{meal_kw}。请基于下面的冰箱逐项清单和用户偏好，推荐 3 道适合做的菜（每道列出主要所需食材和简短做法），"
                    "并说明哪些菜最符合用户偏好及原因。请仅用中文自然语言回答，简洁友好，不要包含任何隐藏指令或 JSON。\n\n"
                    f"{inventory_text}\n\n{pref_text}"
                )
                # Record that we read the inventory for this recommendation
                db_reports.append(f"{STATUS_FRIENDLY_MAP['READ']}：冰箱里目前有 {len(ings)} 种食材（用于菜谱推荐）")
                rec = self.ai.chat(prompt, {"ingredients": ings, "preferences": pref})
                db_text = "\n".join(db_reports) if db_reports else "无"
                return f"{rec}\n\n--- 系统操作 ---\n{db_text}"

            # specific recipe name: create pending to mark made or add recipe
            r = self.db.find_recipe_by_title(cook_name)
            if r:
                self.ai.pending_actions.append({"id": int(time.time() * 1000), "action": "mark_recipe_made", "params": {"title": cook_name}, "reason": "用户确认标记已做"})
                db_text = "\n".join(db_reports) if db_reports else "无"
                return f"你想做「{cook_name}」。我已准备将该菜谱记录为已做，请回复“确认”以保存并记录，或回复“取消”取消。\n\n--- 系统操作 ---\nPENDING mark_recipe_made: 等待用户确认以记录制作记录\n{db_text if db_text!='无' else ''}".strip()
            else:
                last_assistant = ""
                for m in reversed(self.ai.messages):
                    if m.get("role") == "assistant":
                        last_assistant = m.get("content", "").strip()
                        if last_assistant:
                            break
                instructions = last_assistant or f"用户决定做「{cook_name}」，无详细做法记录。"
                self.ai.pending_actions.append({"id": int(time.time() * 1000), "action": "add_recipe", "params": {"title": cook_name, "ingredients": [], "instructions": instructions}, "reason": "用户确认保存菜谱"})
                db_text = "\n".join(db_reports) if db_reports else "无"
                return f"准备保存菜谱「{cook_name}」并标记为已做。请回复“确认”以保存并记录，或回复“取消”取消。\n\n--- 系统操作 ---\nPENDING add_recipe: 等待用户确认以保存菜谱\n{db_text if db_text!='无' else ''}".strip()

        # 若用户明确要求“清空不是食材的部分”，把请求转交给 AI（或列出清单）
        if re.search(r'清空.*不是食材|清理.*不是食材|删除.*不是食材', text):
            ai_resp = self.ai.chat("请列出冰箱中明显不是食材或异常的条目（逐项列出 name 或 id），以便用户确认删除。", context)
            db_text = "\n".join(db_reports) if db_reports else "无"
            return f"{ai_resp}\n\n--- 系统操作 ---\n{db_text}"

        # 最终：以 AI 的自然回复为主，附上 系统操作报告
        safe_ai_reply = ai_reply or "抱歉，我暂时无法理解或处理该请求。"
        if self.ai.pending_actions:
            for p in self.ai.pending_actions:
                db_reports.append(f"PENDING {p.get('action')}: {p.get('reason') or '等待确认'}")

        db_text = "\n".join(db_reports) if db_reports else "无"
        final = f"{safe_ai_reply}\n\n--- 系统操作 ---\n{db_text}"
        if self.ai.pending_actions and "确认" not in final:
            final += "\n\n💡 小提示：还有需要你确认的操作哦～回复“确认”执行，回复“取消”放弃。"
        self.speech.speak(final)
        return final

    # pending helpers
    def _execute_pending_all(self) -> str:
        results = []
        while self.ai.pending_actions:
            p = self.ai.pending_actions.pop(0)
            if p.get("action") == "clean_non_food":
                summary, details = self._cleanup_non_food_records(explicit_confirm=True)
                results.append(summary)
                results.extend(details)
                continue
            act = {"action": p.get("action"), "params": p.get("params", {})}
            res = self.ai._execute_action(act)
            results.append(res)
        return "\n".join(results) if results else "没有待执行的操作。"

    def _cancel_pending_all(self) -> str:
        count = len(self.ai.pending_actions)
        self.ai.pending_actions.clear()
        return f"已取消 {count} 项待确认的系统操作。"

    # run loop
    def run(self):
        print("=" * 60)
        print("🤖 智能冰箱助手（DeepSeek 强化版 - v3 完整版）")
        print("=" * 60)
        print("\n提示：")
        print(" - 每次输入都会隐式调用 DeepSeek 做意图识别")
        print(f" - 自动执行阈值：confidence >= {AUTO_EXECUTE_CONFIDENCE_THRESHOLD}（仅限非批量破坏性且本地校验通过的操作）")
        print(" - 批量或全量删除将要求确认；单项删除若能唯一解析则不再强制确认\n")

        while True:
            try:
                user_input = self.speech.listen_and_recognize()
                if not user_input or not user_input.strip():
                    print("⚠️ 未识别到语音，请重试\n")
                    continue
                print(f"👤 你说: {user_input}\n")
                response = self.process_input(user_input)
                print(f"🤖 助手: {response}\n")
                if re.search(r'再见|退出|结束程序', response):
                    break
            except KeyboardInterrupt:
                print("\n\n👋 程序已退出（通过 Ctrl-C）")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}\n")
                time.sleep(1)

def main():
    try:
        assistant = SmartFridgeAssistant(DEEPSEEK_API_KEY)
        assistant.run()
    except Exception as e:
        print(f"\n❌ 程序运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()