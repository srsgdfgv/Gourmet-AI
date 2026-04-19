#!/usr/bin/env python3
# Flask server to serve SPA and bridge to smart_fridge assistant
# Extended API:
# - GET/POST/PATCH/DELETE /ingredients
# - GET/POST/DELETE /preferences
# - GET /api/recipes (builds prompt using inventory, preferences, current season)
# - existing /listen, /listen/stop, /message preserved
from flask import Flask, send_from_directory, jsonify, request, Response
from flask_cors import CORS
import traceback
import os
import json
import re
from datetime import datetime

try:
    from smart_fridge import SmartFridgeAssistant, DEEPSEEK_API_KEY
except Exception as e:
    raise RuntimeError("无法导入 smart_fridge.py。请确认文件位于同目录且可导入.") from e

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

assistant = SmartFridgeAssistant(DEEPSEEK_API_KEY)


# favicon route: return local file if present, else inline svg
@app.route('/favicon.ico')
def favicon():
    if os.path.exists('favicon.ico'):
        return send_from_directory('.', 'favicon.ico')
    svg = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>
      <rect fill='#2f8de6' width='64' height='64' rx='12'/>
      <text x='50%' y='50%' font-size='36' text-anchor='middle' fill='white' dy='12'>冰</text>
    </svg>"""
    resp = Response(svg, mimetype='image/svg+xml')
    resp.headers['Cache-Control'] = 'public, max-age=86400'
    return resp


# Serve index
@app.route('/')   # 主页
def index():
    return send_from_directory('.', 'index.html')


# Static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


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
        expiry_days = data.get('expiry_days')
        expiry_date = data.get('expiry_date')
        notes = data.get('notes')
        ing = assistant.db.add_or_merge_ingredient(name=name, quantity=quantity, unit=unit,
                                                   category=category, expiry_days=expiry_days,
                                                   expiry_date=expiry_date, notes=notes)
        return jsonify({"ingredient": ing})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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


@app.route('/message', methods=['POST'])
def handle_message():
    try:
        data = request.get_json(force=True)
        text = (data.get('text') or "").strip()
        if not text:
            return jsonify({"reply": "请发送非空文本"}), 400
        reply = assistant.process_input(text)
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
            "conversation_history": assistant.db.get_conversation() if hasattr(assistant.db,
                                                                               'get_conversation') else [],
            "ai_messages": assistant.ai.messages[-20:] if hasattr(assistant, 'ai') and getattr(assistant.ai, 'messages',
                                                                                               None) else []
        }
        return jsonify(hist)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation', methods=['GET'])
def get_conversation_history():
    """获取对话历史记录"""
    try:
        limit = request.args.get('limit', default=100, type=int)
        role = request.args.get('role', default=None)

        convs = assistant.db.get_conversation(limit)

        # 可选的角色筛选
        if role and role in ['user', 'assistant', 'system']:
            convs = [c for c in convs if c.get('role') == role]

        return jsonify({
            "conversations": convs,
            "total": len(convs)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/conversation', methods=['DELETE'])
def clear_conversation_history():
    """清空对话历史"""
    try:
        c = assistant.db.conn.cursor()
        c.execute("DELETE FROM conversation")
        assistant.db.conn.commit()
        return jsonify({"cleared": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------
# Statistics API (新增)
# ---------------------
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """获取冰箱统计数据"""
    try:
        ingredients = assistant.db.get_ingredients()
        prefs = assistant.db.get_preferences()
        convs = assistant.db.get_conversation(limit=1000)

        # 食材统计
        total_ingredients = len(ingredients)
        now = datetime.now()

        soon_expiry = 0
        expired = 0

        for ing in ingredients:
            expiry = ing.get('expiry_date')
            if expiry:
                try:
                    expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                    diff_days = (expiry_date - now).days
                    if diff_days < 0:
                        expired += 1
                    elif diff_days <= 3:
                        soon_expiry += 1
                except:
                    pass

        # 按种类统计
        categories = {
            "果蔬类": 0,
            "肉蛋类": 0,
            "奶制品类": 0,
            "其他": 0
        }
        for ing in ingredients:
            cat = ing.get('category', '其他')
            if cat in categories:
                categories[cat] += 1
            else:
                categories["其他"] += 1

        # 对话统计
        user_msgs = len([c for c in convs if c.get('role') == 'user'])
        assistant_msgs = len([c for c in convs if c.get('role') == 'assistant'])

        return jsonify({
            "ingredients": {
                "total": total_ingredients,
                "soon_expiry": soon_expiry,
                "expired": expired,
                "categories": categories
            },
            "conversation": {
                "total": len(convs),
                "user": user_msgs,
                "assistant": assistant_msgs
            },
            "preferences": {
                "total_types": len(prefs),
                "total_values": sum(len(v) for v in prefs.values())
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("启动 Web UI 服务: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)