# -*- coding: utf-8 -*-
"""加载 local_config.py；缺失时给出明确提示。供 CPython / MicroPython 共用。"""


def load():
    try:
        import local_config as cfg
        return cfg
    except ImportError:
        pass

    # 子目录脚本：尝试项目根目录（CPython）
    try:
        import sys
        import os

        here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ""
        if here and here not in sys.path:
            sys.path.insert(0, here)
        parent = os.path.dirname(here) if here else ""
        if parent and parent not in sys.path:
            sys.path.insert(0, parent)
        import local_config as cfg
        return cfg
    except ImportError:
        pass

    raise ImportError(
        "缺少 local_config.py。请复制 local_config.example.py 为 local_config.py 并填写密钥。"
    )
