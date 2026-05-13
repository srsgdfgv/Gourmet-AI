#!/usr/bin/env python3
# Flask server to serve SPA and bridge to smart_fridge assistant
# Extended API:
# - GET/POST/PATCH/DELETE /ingredients
# - GET/POST/DELETE /preferences
# - GET /api/recipes (builds prompt using inventory, preferences, current season)
# - existing /listen, /listen/stop, /message preserved
from flask import Flask, send_from_directory, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import traceback
import os
import sys
import json
import re
import importlib.util
import threading
import time
from datetime import datetime

# 脚本所在目录即项目根（含 raspberry_pi_client.py、weight_scale.py、recog/ 等）
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 前端静态页所在目录（与仓库中「冰箱助手UI」一致；index 内引用同级的 common.css、app.js）
_FRIDGE_UI_DIR = os.path.join(_PROJECT_ROOT, "冰箱助手UI")


def _static_ui_dir() -> str:
    """
    index.html / common.css / app.js 所在目录。
    - 优先：与 web.ui.py 同目录（板子 CRAIC 文件夹平铺上传时常用）。
    - 否则：使用子目录「冰箱助手UI」（PC 上仓库原有结构）。
    """
    if os.path.isfile(os.path.join(_PROJECT_ROOT, "index.html")):
        return _PROJECT_ROOT
    if os.path.isfile(os.path.join(_FRIDGE_UI_DIR, "index.html")):
        return _FRIDGE_UI_DIR
    return _PROJECT_ROOT


def _load_raspberry_pi_client():
    path = os.path.join(_PROJECT_ROOT, 'raspberry_pi_client.py')
    spec = importlib.util.spec_from_file_location('raspberry_pi_client', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_category_from_name(name: str) -> str:
    """与库存种类一致，用于判断识别结果是否为果蔬类。"""
    if not name or not str(name).strip():
        return '其他'
    for kw in ('奶', '酸奶', '芝士', '奶酪', '黄油'):
        if kw in name:
            return '奶制品类'
    for kw in ('肉', '鸡', '鸭', '猪', '牛', '羊', '鱼', '虾', '蟹', '蛋', '火腿', '培根'):
        if kw in name:
            return '肉蛋类'
    for kw in ('菜', '果', '瓜', '茄', '椒', '葱', '蒜', '姜', '豆', '菇', '笋', '叶', '萝卜', '薯', '梨', '桃', '橙', '莓', '蕉', '柚', '柠檬'):
        if kw in name:
            return '果蔬类'
    return '其他'


try:
    from smart_fridge import SmartFridgeAssistant, DEEPSEEK_API_KEY
except Exception as e:
    raise RuntimeError("无法导入 smart_fridge.py。请确认文件位于同目录且可导入.") from e

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

assistant = SmartFridgeAssistant(DEEPSEEK_API_KEY)

# 前端「点录音 / 点停止」后台线程（树莓派麦克风连续写入直至 request_stop）
manual_record_lock = threading.Lock()
manual_record_thread = None


# favicon route: return local file if present, else inline svg
@app.route('/favicon.ico')
def favicon():
    fav = os.path.join(_PROJECT_ROOT, "favicon.ico")
    if os.path.isfile(fav):
        return send_from_directory(_PROJECT_ROOT, "favicon.ico")
    svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
      <rect fill='#2f8de6' width='64' height='64' rx='12'/>
      <text x='50%' y='50%' font-size='36' text-anchor='middle' fill='white' dy='12'>冰</text>
    </svg>"""
    resp = Response(svg, mimetype='image/svg+xml')
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp


# Serve index：同目录平铺优先，否则 冰箱助手UI/
@app.route('/')
def index():
    return send_from_directory(_static_ui_dir(), "index.html")


# 与 index.html 中相对路径 href="common.css"、src="app.js" 对应
@app.route("/common.css")
def serve_fridge_common_css():
    return send_from_directory(_static_ui_dir(), "common.css")


@app.route("/app.js")
def serve_fridge_app_js():
    return send_from_directory(_static_ui_dir(), "app.js")


# ---------------------
# Ingredients REST API
# ---------------------
@app.route('/ingredients', methods=['GET'])
def get_ingredients():
    try:
        items = assistant.db.get_ingredients()
        return jsonify({"ingredients": items})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/ingredients', methods=['POST'])
def post_ingredient():
    """
    Accepts JSON:
    {
      "name": "西红柿",
      "quantity": 2,
      "unit": "个",
      "category": "蔬菜",
      "expiry_days": 7,
      "expiry_date": "2026-01-20",
      "notes": "无"
    }
    """
    try:
        data = request.get_json(force=True)
        name = (data.get('name') or "").strip()
        if not name:
            return jsonify({"error": "name required"}), 400
        quantity = float(data.get('quantity', 1.0) or 1.0)
        unit = data.get('unit') or "个"
        category = data.get('category') or "其他"
        fridge_area = data.get('fridge_area') or '冷藏区'
        expiry_days = data.get('expiry_days')
        expiry_date = data.get('expiry_date')
        notes = data.get('notes')
        ing = assistant.db.add_or_merge_ingredient(name=name, quantity=quantity, unit=unit,
                                                  category=category, fridge_area=fridge_area,
                                                  expiry_days=expiry_days,
                                                  expiry_date=expiry_date, notes=notes)
        return jsonify({"ingredient": ing})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _ndjson_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + '\n'


@app.route('/ingredients/scan', methods=['POST'])
def post_ingredient_scan():
    """
    NDJSON 流：逐步输出进度。
    顺序：先电子秤至稳定非零克数（称重环节结束）→ 再 K230 摄像头识别。
    仅在「稳定重量 + 识别名称有效 + 推断为果蔬类」齐备时写入数据库一次。
    """
    data = request.get_json(force=True) or {}

    @stream_with_context
    def generate():
        try:
            yield _ndjson_line({
                'step': 'init',
                'message': '开始采集：先称重至稳定非零，再启动摄像头识别；二者齐备后入库',
            })

            k230_host = data.get('k230_host') or None
            fridge_area = data.get('fridge_area') or '冷藏区'
            if fridge_area not in ('冷藏区', '冷冻区'):
                fridge_area = '冷藏区'
            expiry_days = data.get('expiry_days', 7)
            extra_notes = (data.get('notes') or '').strip()

            try:
                from weight_scale import iter_read_stable_weight_progress
            except ImportError as e:
                yield _ndjson_line({
                    'step': 'error',
                    'phase': 'weight_module',
                    'error': f'无法加载称重模块（需在树莓派上运行并安装依赖）: {e}',
                })
                return

            yield _ndjson_line({
                'step': 'weight_phase',
                'message': '步骤 1/3：电子秤 — 先空秤归零再置物；出现稳定非零克数后称重环节结束',
            })

            weight_g = None
            fail_ev = None
            for ev in iter_read_stable_weight_progress():
                yield _ndjson_line({'step': 'weight_progress', 'event': ev})
                ph = ev.get('phase')
                if ph == 'stable_weight':
                    weight_g = ev.get('weight_g')
                elif ph == 'failed':
                    fail_ev = ev

            if weight_g is None:
                msg = (fail_ev or {}).get('message') or '未能读取稳定非零重量'
                yield _ndjson_line({
                    'step': 'error',
                    'phase': 'weight',
                    'error': msg,
                    'detail': fail_ev,
                })
                return

            qty = max(0.1, round(float(weight_g), 2))
            if qty < 0.5:
                yield _ndjson_line({
                    'step': 'error',
                    'phase': 'weight',
                    'error': '重量过小，未写入数据库',
                    'weight_g': qty,
                })
                return

            yield _ndjson_line({
                'step': 'weight_complete',
                'message': '称重已结束（稳定非零 ' + str(qty) + ' g），随后启动 K230 摄像头识别',
                'weight_g': qty,
            })

            yield _ndjson_line({'step': 'recognize', 'message': '步骤 2/3：连接 K230 并识别…'})
            rpc = _load_raspberry_pi_client()
            rec = rpc.request_recognize(host=k230_host)
            yield _ndjson_line({'step': 'recognize_result', 'recognition': rec})

            if not rec or not rec.get('ok'):
                err = (rec or {}).get('error') or '视觉识别失败'
                yield _ndjson_line({
                    'step': 'error',
                    'phase': 'recognize',
                    'error': err,
                    'recognition': rec,
                    'weight_g': qty,
                })
                return

            name = (rec.get('name') or '').strip()
            if not name:
                yield _ndjson_line({
                    'step': 'error',
                    'phase': 'recognize',
                    'error': '识别结果中无食材名称',
                    'recognition': rec,
                    'weight_g': qty,
                })
                return

            category = infer_category_from_name(name)
            yield _ndjson_line({
                'step': 'category',
                'name': name,
                'inferred_category': category,
                'message': f'推断种类：{category}' + ('（将校验是否为果蔬类）' if category == '果蔬类' else ''),
                'weight_g': qty,
            })

            if category != '果蔬类':
                yield _ndjson_line({
                    'step': 'error',
                    'phase': 'category',
                    'error': '识别为非果蔬类食材，未添加库存。请仅放入果蔬后重新识别。',
                    'recognition': rec,
                    'inferred_category': category,
                    'weight_g': qty,
                })
                return

            yield _ndjson_line({
                'step': 'commit',
                'message': '步骤 3/3：重量与种类已齐备，写入数据库…',
                'name': name,
                'weight_g': qty,
            })

            notes_parts = [f'自动采集：HX711 称重 {qty} 克 + K230 识别']
            if extra_notes:
                notes_parts.append(extra_notes)
            notes = ' | '.join(notes_parts)

            ing = assistant.db.add_or_merge_ingredient(
                name=name,
                quantity=qty,
                unit='克',
                category='果蔬类',
                fridge_area=fridge_area,
                expiry_days=expiry_days if expiry_days is not None else 7,
                notes=notes,
            )
            yield _ndjson_line({
                'step': 'done',
                'ingredient': dict(ing) if ing else None,
                'recognition': rec,
                'weight_g': qty,
            })
        except Exception as e:
            traceback.print_exc()
            yield _ndjson_line({'step': 'error', 'phase': 'server', 'error': str(e)})

    return Response(
        generate(),
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        },
    )


@app.route('/ingredients/<int:ingredient_id>', methods=['DELETE'])
def delete_ingredient(ingredient_id):
    try:
        assistant.db.remove_ingredient(ingredient_id)
        return jsonify({"deleted": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/ingredients/<int:ingredient_id>', methods=['PATCH'])
def patch_ingredient(ingredient_id):
    """
    Partial update: accept fields quantity, unit, category, expiry_days, expiry_date, freshness, notes
    """
    try:
        data = request.get_json(force=True)
        # lookup id exists
        res = assistant.db.update_ingredient(ingredient_id,
                                             quantity=data.get('quantity'),
                                             unit=data.get('unit'),
                                             category=data.get('category'),
                                             expiry_date=data.get('expiry_date'),
                                             expiry_days=data.get('expiry_days'),
                                             freshness=data.get('freshness'),
                                             notes=data.get('notes'))
        if not res:
            return jsonify({"error": "not found"}), 404
        return jsonify({"ingredient": res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------
# Preferences API
# ---------------------
@app.route('/preferences', methods=['GET'])
def get_preferences():
    try:
        prefs = assistant.db.get_preferences()
        return jsonify({"preferences": prefs})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/preferences', methods=['POST'])
def post_preference():
    """
    Body: {"pref_type":"饮食偏好", "value":"不吃辣"}
    """
    try:
        data = request.get_json(force=True)
        pref_type = data.get('pref_type')
        value = data.get('value')
        if not pref_type or not value:
            return jsonify({"error": "pref_type and value required"}), 400
        res = assistant.db.set_preference(pref_type, value)
        return jsonify({"preference": res})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/preferences', methods=['DELETE'])
def delete_preference():
    """
    Body: {"pref_type":"口味", "value":"清淡"}
    """
    try:
        data = request.get_json(force=True)
        pref_type = data.get('pref_type')
        value = data.get('value')
        if not pref_type or not value:
            return jsonify({"error": "pref_type and value required"}), 400
        assistant.db.remove_preference(pref_type, value)
        return jsonify({"deleted": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------
# Listen / Message endpoints (existing behavior)
# ---------------------
@app.route('/listen', methods=['POST'])
def listen_once():
    """
    Blocking listen endpoint.
    If recognition result is empty, returns {"recognized": "", "reply": ""} (front-end will ignore empty reply)
    """
    try:
        recognized = assistant.speech.listen_and_recognize()
        if not recognized or not str(recognized).strip():
            return jsonify({"recognized": "", "reply": ""})
        reply = assistant.process_input(recognized)
        return jsonify({"recognized": recognized or "", "reply": reply or ""})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"录音或识别失败: {e}"}), 500


@app.route('/listen/stop', methods=['POST'])
def listen_stop():
    try:
        # 请求停止正在进行的录音和播放
        try:
            assistant.speech.request_stop()
        except Exception as e:
            print(f" request_stop 失败: {e}")
        try:
            # 同时尝试停止任何正在播放的 TTS 音频
            stopped_playback = assistant.speech.stop_playback()
            if stopped_playback:
                print("已停止正在播放的语音。")
        except Exception as e:
            print(f" stop_playback 失败: {e}")
        return jsonify({"stopped": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/listen/manual/start', methods=['POST'])
def listen_manual_start():
    """开始一段手动控制的麦克风录音（与 VAD 无关），直至 /listen/manual/stop。"""
    global manual_record_thread
    with manual_record_lock:
        if manual_record_thread is not None and not manual_record_thread.is_alive():
            manual_record_thread = None
        if manual_record_thread is not None and manual_record_thread.is_alive():
            return jsonify({"error": "已在录音中"}), 409

        def worker():
            global manual_record_thread
            try:
                while assistant.speech.is_tts_busy():
                    time.sleep(0.05)
                assistant.speech.record_until_stop_requested()
            finally:
                with manual_record_lock:
                    if manual_record_thread is threading.current_thread():
                        manual_record_thread = None

        manual_record_thread = threading.Thread(target=worker, daemon=True)
        manual_record_thread.start()
    return jsonify({"started": True})


@app.route('/listen/manual/stop', methods=['POST'])
def listen_manual_stop():
    """停止手动录音 → 百度转写 → process_input 生成回复。"""
    global manual_record_thread

    try:
        try:
            assistant.speech.request_stop()
        except Exception as e:
            print(f"manual stop request_stop: {e}")

        t = None
        with manual_record_lock:
            t = manual_record_thread
        if t is not None:
            t.join(timeout=35.0)
            if t.is_alive():
                return jsonify({"error": "停止录音超时，请重试"}), 500
            with manual_record_lock:
                if manual_record_thread is t:
                    manual_record_thread = None

        recognized = assistant.speech.transcribe_temp_pcm_and_cleanup()
        if not recognized or not str(recognized).strip():
            return jsonify({"recognized": "", "reply": ""})
        reply = assistant.process_input(recognized)
        return jsonify({"recognized": recognized or "", "reply": reply or ""})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"处理失败: {e}"}), 500


@app.route('/listen/gpio_last', methods=['GET'])
def listen_gpio_last():
    """
    Return latest GPIO-triggered listen result for front-end polling.
    """
    try:
        result = getattr(gpio_trigger_callback, "last_result", {"recognized": "", "reply": ""})
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/message', methods=['POST'])
def handle_message():
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or "").strip()
        if not text:
            return jsonify({"reply": "请发送非空文本"}), 400
        # 默认开启语音；前端传 tts / enable_tts 为 false 时可关闭（避免叠音）
        if "tts" in data:
            want_tts = bool(data.get("tts"))
        elif "enable_tts" in data:
            want_tts = bool(data.get("enable_tts"))
        else:
            want_tts = True
        reply = assistant.process_input(text, enable_tts=want_tts)
        return jsonify({"reply": reply})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"处理失败: {e}"}), 500


# ---------------------
# Recipes API
# ---------------------
def month_to_season(month: int) -> str:
    # Northern Hemisphere seasons — adjust if needed
    if month in (12, 1, 2):
        return "冬季"
    if month in (3, 4, 5):
        return "春季"
    if month in (6, 7, 8):
        return "夏季"
    return "秋季"


@app.route('/api/recipes', methods=['GET'])
def api_recipes():
    try:
        ingredients = assistant.db.get_ingredients()
        prefs = assistant.db.get_preferences()
        month = datetime.now().month
        season = month_to_season(month)
        ing_list = []
        for i in ingredients:
            q = i.get('quantity') or ''
            unit = i.get('unit') or ''
            ing_list.append(f"{i.get('name')} {q}{unit}".strip())
        ing_text = ", ".join(ing_list) if ing_list else "无特别食材"

        pref_text_parts = []
        for k, v in prefs.items():
            if v:
                pref_text_parts.append(f"{k}: {', '.join(v)}")
        pref_text = "；".join(pref_text_parts) if pref_text_parts else "无特别偏好"

        prompt = (
            "请根据以下信息推荐最多3个适合的家常菜谱，返回严格的JSON数组，格式："
            '[{"title":"...","desc":"一句话描述","ingredients":["..."],"instructions":"步骤文本（必要时）"}]\n\n'
            f"当前季节：{season}（当前月份：{month}）。\n"
            f"当前冰箱食材：{ing_text}。\n"
            f"用户偏好：{pref_text}。\n"
            "要求：优先使用已有食材；给出简单可执行的做法（每道菜3-8步）；只返回JSON数组，不要有任何其他文字。"
        )

        ai_resp = assistant.ai.chat(prompt, context=None)
        if not ai_resp:
            return jsonify({"recipes": []})

        # 尝试提取 JSON 数组（兼容前后可能有其他文本）
        json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', ai_resp)
        if json_match:
            json_str = json_match.group(0)
            try:
                parsed = json.loads(json_str)
                recipes = []
                for rc in parsed:
                    if isinstance(rc, dict):
                        recipes.append({
                            "title": rc.get("title") or rc.get("name") or "",
                            "desc": rc.get("desc") or rc.get("description") or "",
                            "ingredients": rc.get("ingredients") or [],
                            "instructions": rc.get("instructions") or rc.get("steps") or ""
                        })
                return jsonify({"recipes": recipes})
            except Exception:
                pass

        # 若无法解析，返回空数组
        return jsonify({"recipes": []})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------
# History / Debug APIs (optional)
# ---------------------
@app.route('/api/history', methods=['GET'])
def api_history():
    try:
        hist = {
            "recipe_history": assistant.db.get_recipe_history() if hasattr(assistant.db, 'get_recipe_history') else [],
            "conversation_history": assistant.db.get_conversation() if hasattr(assistant.db, 'get_conversation') else [],
            "ai_messages": assistant.ai.messages[-20:] if hasattr(assistant, 'ai') and getattr(assistant.ai, 'messages', None) else []
        }
        return jsonify(hist)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Static files（必须注册在所有 API 路由之后，否则 /<path:filename> 会抢走 /ingredients 等路径并 404）
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(_PROJECT_ROOT, filename)


import RPi.GPIO as GPIO

# 定义 GPIO 引脚（BCM 编号）
# GPIO4: 会话启动（上升沿）
# GPIO17: 会话停止（上升沿）
ASRPRO_GPIO_PIN = 4
STOP_GPIO_PIN = 17
gpio_trigger_callback_last_default = {"recognized": "", "reply": ""}
GPIO_SESSION_STOP_GRACE_SEC = 1.0

gpio_session_lock = threading.Lock()
gpio_session_active = False
gpio_session_stop_event = threading.Event()
gpio_session_started_at = 0.0


def _end_gpio_session():
    """请求结束 GPIO 会话，并尽快打断当前录音/播放。"""
    gpio_session_stop_event.set()
    try:
        assistant.speech.request_stop()
    except Exception:
        pass
    try:
        assistant.speech.stop_playback()
    except Exception:
        pass


def _gpio_conversation_worker():
    """GPIO 会话线程：在单次会话内循环「听→识别→回复」，直到 GPIO17 上升沿触发结束。"""
    global gpio_session_active, gpio_session_started_at
    with gpio_session_lock:
        gpio_session_active = True
        gpio_session_started_at = time.time()
        gpio_session_stop_event.clear()

    print("🎙️ GPIO 会话已启动：持续对话中，GPIO17 上升沿可结束会话。")
    try:
        while not gpio_session_stop_event.is_set():
            if assistant.speech.is_playing():
                time.sleep(0.1)
                continue

            recognized = assistant.speech.listen_and_recognize()
            if gpio_session_stop_event.is_set():
                break
            if not recognized or not str(recognized).strip():
                continue

            print(f"👤 识别文本: {recognized}")
            reply = assistant.process_input(recognized)
            gpio_trigger_callback.last_result = {"recognized": recognized or "", "reply": reply or ""}
            print(f"🤖 助手回复: {reply}")
    except Exception as e:
        traceback.print_exc()
        gpio_trigger_callback.last_result = {"error": f"GPIO 会话异常: {e}"}
        print(f"❌ GPIO 会话异常: {e}")
    finally:
        with gpio_session_lock:
            gpio_session_active = False
        gpio_session_stop_event.clear()
        print("🛑 GPIO 会话已结束。")

def gpio_trigger_callback(channel):
    """
    GPIO 上升沿回调（在独立线程中执行）
    """
    global gpio_session_active
    now = time.time()
    if hasattr(gpio_trigger_callback, "last_trigger") and (now - gpio_trigger_callback.last_trigger) < 1.0:
        return
    gpio_trigger_callback.last_trigger = now

    print("🔔 检测到 ASRPRO 语音唤醒（GPIO 高电平）")
    with gpio_session_lock:
        already_active = gpio_session_active
    if already_active:
        print("ℹ️ GPIO 会话已在进行中，忽略重复唤醒。")
        return
    threading.Thread(target=_gpio_conversation_worker, daemon=True).start()


def gpio_stop_callback(channel):
    """GPIO17 上升沿回调：作为会话结束信号。"""
    with gpio_session_lock:
        active = gpio_session_active
        started_at = gpio_session_started_at
    if not active:
        return
    # 忽略会话启动后极短时间内的触发，避免误结束。
    if (time.time() - started_at) < GPIO_SESSION_STOP_GRACE_SEC:
        return
    print("🛑 检测到 GPIO17 高脉冲，结束当前持续对话会话。")
    _end_gpio_session()


gpio_trigger_callback.last_result = gpio_trigger_callback_last_default.copy()

def start_gpio_monitor():
    """启动 GPIO 监听线程"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ASRPRO_GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(STOP_GPIO_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    # GPIO4 上升沿开始会话，GPIO17 上升沿结束会话
    GPIO.add_event_detect(ASRPRO_GPIO_PIN, GPIO.RISING, callback=gpio_trigger_callback, bouncetime=200)
    GPIO.add_event_detect(STOP_GPIO_PIN, GPIO.RISING, callback=gpio_stop_callback, bouncetime=200)
    print(f"✅ GPIO 监控已启动，GPIO{ASRPRO_GPIO_PIN} 上升沿启动会话，GPIO{STOP_GPIO_PIN} 上升沿结束会话")
    # 保持线程运行
    while True:
        time.sleep(1)
        

        

if __name__ == '__main__':
    gpio_thread = threading.Thread(target=start_gpio_monitor, daemon=True)
    gpio_thread.start()
    print("启动 Web UI 服务: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)        
    