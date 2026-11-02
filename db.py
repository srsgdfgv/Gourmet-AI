# SQLite 持久化数据库模块，用于智能冰箱助手
# 数据库文件: /home/pi/test/smart_fridge.db

import os
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

DEFAULT_DIR = "/home/pi/test"
DEFAULT_DB = os.path.join(DEFAULT_DIR, "smart_fridge.db")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def dict_from_row(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


class SqliteDatabase:
    """
    持久化 DB（支持 CRUD 操作）
    Tables:
    - ingredients
    - preferences
    - recipes
    - conversation
    - shelf_life
    """

    def __init__(self, db_path: str = DEFAULT_DB, ai_service=None):
        self.db_path = db_path
        self.ai_service = ai_service
        ensure_dir(os.path.dirname(self.db_path))
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = lambda cursor, row: dict_from_row(cursor, row)
        self._init_tables()
        self._ensure_recipe_columns()
        self._ensure_shelf_life_table()

    def _init_tables(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 0,
            unit TEXT NOT NULL DEFAULT '个',
            category TEXT DEFAULT '其他',  -- 调整默认值，仅保留四类：果蔬类/肉蛋类/奶制品类/其他
            fridge_area TEXT DEFAULT '冷藏区',  -- 新增：冰箱区域（冷藏区/冷冻区）
            freshness INTEGER DEFAULT 100,
            added_date TEXT,
            expiry_date TEXT,
            notes TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ing_norm ON ingredients(normalized_name)")
        c.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pref_type TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            ingredients_json TEXT,
            instructions TEXT,
            created_at TEXT,
            source TEXT,
            made_count INTEGER DEFAULT 0,
            last_made TEXT
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS conversation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT,
            created_at TEXT
        )""")
        self.conn.commit()

    def _ensure_recipe_columns(self):
        # Already created with columns; keep for compatibility
        self.conn.commit()

    def _ensure_shelf_life_table(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS shelf_life (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            normalized_name TEXT NOT NULL,
            display_name TEXT,
            days INTEGER,
            note TEXT,
            updated_at TEXT
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_shelf_norm ON shelf_life(normalized_name)")
        self.conn.commit()

    # ---------- helpers ----------
    def _normalize(self, name: str) -> str:
        if not name:
            return ""
        return name.strip().lower()

    # ---------- INGREDIENTS CRUD ----------
    def get_ingredients(self) -> List[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM ingredients ORDER BY expiry_date IS NOT NULL, expiry_date")
        return c.fetchall()

    def find_ingredient_by_normalized(self, normalized_name: str) -> Optional[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM ingredients WHERE normalized_name = ?", (normalized_name,))
        return c.fetchone()

    def add_or_merge_ingredient(self, name: str, quantity: float = 1.0, unit: str = "个",
                                category: str = "其他", fridge_area: str = "冷藏区",  # 新增 fridge_area 参数，默认冷藏区
                                expiry_days: Optional[int] = None,
                                expiry_date: Optional[str] = None, freshness: Optional[int] = None,
                                notes: Optional[str] = None) -> Dict:
        name = name.strip()
        normalized = self._normalize(name)
        if expiry_days is not None and expiry_days >= 0:
            expiry = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
        elif expiry_date:
            expiry = expiry_date
        else:
            expiry = None
        if freshness is None:
            freshness = 100
        c = self.conn.cursor()
        c.execute("SELECT * FROM ingredients WHERE normalized_name = ? AND unit = ?", (normalized, unit))
        existing = c.fetchone()
        if existing:
            new_quantity = float(existing["quantity"]) + float(quantity)
            ex_expiry = existing.get("expiry_date")
            if ex_expiry and expiry:
                chosen_expiry = ex_expiry if ex_expiry <= expiry else expiry
            else:
                chosen_expiry = expiry or ex_expiry
            new_freshness = min(int(existing.get("freshness", 100)), int(freshness or 100))
            # 新增：更新 fridge_area（仅当传入非默认值时覆盖，或保留原有逻辑）
            new_category = category if category in ["果蔬类", "肉蛋类", "奶制品类", "其他"] else existing.get("category", "其他")
            new_fridge_area = fridge_area if fridge_area in ["冷藏区", "冷冻区"] else existing.get("fridge_area", "冷藏区")
            c.execute("""
                UPDATE ingredients
                SET quantity = ?, expiry_date = ?, freshness = ?, added_date = ?,
                    category = ?, fridge_area = ?  -- 新增更新 category 和 fridge_area
                WHERE id = ?
            """, (new_quantity, chosen_expiry, new_freshness, datetime.now().strftime("%Y-%m-%d"),
                  new_category, new_fridge_area, existing["id"]))
            self.conn.commit()
            c.execute("SELECT * FROM ingredients WHERE id = ?", (existing["id"],))
            return c.fetchone()
        else:
            # 校验 category 和 fridge_area 合法性
            valid_category = category if category in ["果蔬类", "肉蛋类", "奶制品类", "其他"] else "其他"
            valid_fridge_area = fridge_area if fridge_area in ["冷藏区", "冷冻区"] else "冷藏区"
            c.execute("""
                INSERT INTO ingredients (name, normalized_name, quantity, unit, category, fridge_area, 
                                        freshness, added_date, expiry_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)  -- 新增 fridge_area 字段
            """, (name, normalized, float(quantity), unit, valid_category, valid_fridge_area,
                  freshness, datetime.now().strftime("%Y-%m-%d"), expiry, notes))
            self.conn.commit()
            return self.find_ingredient_by_normalized(normalized)

    def update_ingredient(self, name_or_id: str, quantity: Optional[float] = None, unit: Optional[str] = None,
                          category: Optional[str] = None, fridge_area: Optional[str] = None,  # 新增 fridge_area 参数
                          expiry_date: Optional[str] = None,
                          expiry_days: Optional[int] = None, freshness: Optional[int] = None,
                          notes: Optional[str] = None) -> Optional[Dict]:
        c = self.conn.cursor()
        row = None
        try:
            iid = int(name_or_id)
            c.execute("SELECT * FROM ingredients WHERE id = ?", (iid,))
            row = c.fetchone()
        except:
            normalized = self._normalize(name_or_id)
            c.execute("SELECT * FROM ingredients WHERE normalized_name = ?", (normalized,))
            row = c.fetchone()
        if not row:
            return None
        updates = []
        params = []
        if quantity is not None:
            updates.append("quantity = ?"); params.append(float(quantity))
        if unit is not None:
            updates.append("unit = ?"); params.append(unit)
        if category is not None:
            # 校验 category 合法性
            valid_category = category if category in ["果蔬类", "肉蛋类", "奶制品类", "其他"] else "其他"
            updates.append("category = ?"); params.append(valid_category)
        if fridge_area is not None:  # 新增：处理 fridge_area 更新
            valid_fridge_area = fridge_area if fridge_area in ["冷藏区", "冷冻区"] else "冷藏区"
            updates.append("fridge_area = ?"); params.append(valid_fridge_area)
        if freshness is not None:
            updates.append("freshness = ?"); params.append(int(freshness))
        if notes is not None:
            updates.append("notes = ?"); params.append(notes)
        if expiry_days is not None:
            expiry = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
            updates.append("expiry_date = ?"); params.append(expiry)
        elif expiry_date is not None:
            updates.append("expiry_date = ?"); params.append(expiry_date)
        if not updates:
            return row
        params.append(row["id"])
        sql = f"UPDATE ingredients SET {', '.join(updates)} WHERE id = ?"
        c.execute(sql, params)
        self.conn.commit()
        c.execute("SELECT * FROM ingredients WHERE id = ?", (row["id"],))
        return c.fetchone()

    def remove_ingredient(self, ingredient_id: int):
        c = self.conn.cursor()
        c.execute("DELETE FROM ingredients WHERE id = ?", (ingredient_id,))
        self.conn.commit()

    def clear_inventory(self):
        c = self.conn.cursor()
        c.execute("DELETE FROM ingredients")
        self.conn.commit()

    # ---------- PREFERENCES ----------
    def set_preference(self, pref_type: str, value: str) -> Dict:
        c = self.conn.cursor()
        c.execute("SELECT * FROM preferences WHERE pref_type = ? AND value = ?", (pref_type, value))
        if c.fetchone():
            return {"pref_type": pref_type, "value": value}
        c.execute("INSERT INTO preferences (pref_type, value, created_at) VALUES (?, ?, ?)",
                  (pref_type, value, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()
        return {"pref_type": pref_type, "value": value}

    def get_preferences(self) -> Dict[str, List[str]]:
        c = self.conn.cursor()
        c.execute("SELECT pref_type, value FROM preferences")
        rows = c.fetchall()
        prefs = {}
        for r in rows:
            prefs.setdefault(r["pref_type"], []).append(r["value"])
        return prefs

    def remove_preference(self, pref_type: str, value: str):
        c = self.conn.cursor()
        c.execute("DELETE FROM preferences WHERE pref_type = ? AND value = ?", (pref_type, value))
        self.conn.commit()

    def update_preference(self, old_value: str, new_value: str, pref_type: Optional[str] = None) -> bool:
        c = self.conn.cursor()
        if pref_type:
            c.execute("UPDATE preferences SET value = ? WHERE pref_type = ? AND value = ?", (new_value, pref_type, old_value))
        else:
            c.execute("UPDATE preferences SET value = ? WHERE value = ?", (new_value, old_value))
        self.conn.commit()
        return c.rowcount > 0

    def clear_preferences(self):
        c = self.conn.cursor()
        c.execute("DELETE FROM preferences")
        self.conn.commit()

    # ---------- RECIPES ----------
    def add_recipe(self, title: str, ingredients: List[Dict], instructions: str, source: str = "assistant") -> Dict:
        c = self.conn.cursor()
        ing_json = json.dumps(ingredients, ensure_ascii=False)
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("INSERT INTO recipes (title, ingredients_json, instructions, created_at, source, made_count, last_made) VALUES (?, ?, ?, ?, ?, 0, NULL)",
                  (title, ing_json, instructions, created_at, source))
        self.conn.commit()
        # return a simple record (without id)
        return {"title": title, "ingredients": ingredients, "instructions": instructions, "created_at": created_at}

    def get_recipe_history(self, limit: int = 50) -> List[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM recipes ORDER BY created_at DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        for r in rows:
            try:
                r["ingredients"] = json.loads(r.get("ingredients_json") or "[]")
            except:
                r["ingredients"] = []
        return rows

    def find_recipe_by_title(self, title: str) -> Optional[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM recipes WHERE title = ? ORDER BY created_at DESC LIMIT 1", (title,))
        r = c.fetchone()
        if r:
            try:
                r["ingredients"] = json.loads(r.get("ingredients_json") or "[]")
            except:
                r["ingredients"] = []
        return r

    def update_recipe(self, title: str, new_title: Optional[str] = None, ingredients: Optional[List[Dict]] = None, instructions: Optional[str] = None) -> bool:
        r = self.find_recipe_by_title(title)
        if not r:
            return False
        updates = []
        params = []
        if new_title:
            updates.append("title = ?"); params.append(new_title)
        if ingredients is not None:
            updates.append("ingredients_json = ?"); params.append(json.dumps(ingredients, ensure_ascii=False))
        if instructions is not None:
            updates.append("instructions = ?"); params.append(instructions)
        if not updates:
            return True
        params.append(r["id"])
        sql = f"UPDATE recipes SET {', '.join(updates)} WHERE id = ?"
        c = self.conn.cursor()
        c.execute(sql, params)
        self.conn.commit()
        return c.rowcount > 0

    def delete_recipe(self, title: str) -> bool:
        c = self.conn.cursor()
        c.execute("DELETE FROM recipes WHERE title = ?", (title,))
        self.conn.commit()
        return c.rowcount > 0

    def mark_recipe_made(self, recipe_id: int):
        c = self.conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute("UPDATE recipes SET made_count = COALESCE(made_count,0) + 1, last_made = ? WHERE id = ?", (now, recipe_id))
        self.conn.commit()

    # ---------- CONVERSATION ----------
    def log_conversation(self, role: str, content: str):
        c = self.conn.cursor()
        c.execute("INSERT INTO conversation (role, content, created_at) VALUES (?, ?, ?)",
                  (role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def get_conversation(self, limit: int = 100) -> List[Dict]:
        c = self.conn.cursor()
        c.execute("SELECT * FROM conversation ORDER BY created_at DESC LIMIT ?", (limit,))
        return c.fetchall()

    # ---------- SHELF LIFE ----------
    def set_shelf_life(self, name: str, days: int, note: Optional[str] = None) -> Dict:
        normalized = self._normalize(name)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c = self.conn.cursor()
        c.execute("SELECT * FROM shelf_life WHERE normalized_name = ?", (normalized,))
        existing = c.fetchone()
        if existing:
            c.execute("UPDATE shelf_life SET days = ?, note = ?, updated_at = ? WHERE id = ?",
                      (int(days), note, now, existing["id"]))
        else:
            c.execute("INSERT INTO shelf_life (normalized_name, display_name, days, note, updated_at) VALUES (?, ?, ?, ?, ?)",
                      (normalized, name, int(days), note, now))
        self.conn.commit()
        return {"name": name, "days": int(days), "note": note, "updated_at": now}

    def get_shelf_life(self, name: str) -> Optional[Dict]:
        normalized = self._normalize(name)
        c = self.conn.cursor()
        c.execute("SELECT * FROM shelf_life WHERE normalized_name = ? ORDER BY updated_at DESC LIMIT 1", (normalized,))
        return c.fetchone()

    def delete_shelf_life(self, name: str) -> bool:
        normalized = self._normalize(name)
        c = self.conn.cursor()
        c.execute("DELETE FROM shelf_life WHERE normalized_name = ?", (normalized,))
        self.conn.commit()
        return c.rowcount > 0

    # ---------- AI-backed validation (optional) ----------
    def validate_item_with_ai(self, item_text: str) -> Dict:
        if not self.ai_service:
            simple_food_kw = ["蛋", "鸡蛋", "牛肉", "猪肉", "鱼", "蔬菜", "西红柿", "牛奶", "面包", "豆腐", "米", "面"]
            for kw in simple_food_kw:
                if kw in item_text:
                    return {"is_food": True, "name": item_text.strip(), "quantity": 1.0, "unit": "个", "category": "其他", "note": "基于关键词"}
            return {"is_food": False, "name": item_text.strip(), "quantity": 0, "unit": "", "category": "", "note": "无法判定"}
        prompt = (
            "你是食品解析助手。请判断并以 JSON 返回："
            '{"is_food": true/false, "name":"...", "quantity": 数字, "unit":"...", "category":"肉/蛋/奶/菜/调料/其他", "note":"..."}'
            f" 文本：\"{item_text}\""
        )
        ai_resp = self.ai_service.chat(prompt)
        if not ai_resp:
            return {"is_food": False, "name": item_text.strip(), "quantity": 0, "unit": "", "category": "", "note": "AI 无响应"}
        import re
        m = re.search(r'(\{[\s\S]*\})', ai_resp)
        if not m:
            return {"is_food": False, "name": item_text.strip(), "quantity": 0, "unit": "", "category": "", "note": "AI 返回无法解析"}
        try:
            parsed = json.loads(m.group(1))
            parsed.setdefault("is_food", False)
            parsed.setdefault("name", item_text.strip())
            parsed.setdefault("quantity", 0)
            parsed.setdefault("unit", "")
            parsed.setdefault("category", "其他")
            parsed.setdefault("note", ai_resp[:200])
            return parsed
        except Exception as e:
            return {"is_food": False, "name": item_text.strip(), "quantity": 0, "unit": "", "category": "", "note": f"解析失败: {e}"}

    def close(self):
        try:
            self.conn.close()
        except:
            pass