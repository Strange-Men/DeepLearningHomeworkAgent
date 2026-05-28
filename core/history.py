"""历史记录模块 - SQLite存储检测历史。"""

import json
import sqlite3
import os
from datetime import datetime

import config


class HistoryManager:
    """历史记录管理器。"""

    def __init__(self, db_path=None):
        self.db_path = db_path or config.DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                source_path TEXT,
                result_path TEXT,
                detections TEXT NOT NULL,
                gesture_count INTEGER DEFAULT 0,
                fps REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def add_record(self, source_type, source_path, result_path, detections, gesture_count, fps=None):
        """添加检测记录。"""
        conn = self._get_conn()
        cursor = conn.execute(
            """INSERT INTO detection_history
               (source_type, source_path, result_path, detections, gesture_count, fps, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (source_type, source_path, result_path,
             json.dumps(detections, ensure_ascii=False),
             gesture_count, fps, datetime.now().isoformat())
        )
        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id

    def get_records(self, limit=50):
        """获取历史记录列表。"""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM detection_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_record_by_id(self, record_id):
        """根据ID获取单条记录。"""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM detection_history WHERE id = ?",
            (record_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def delete_record(self, record_id):
        """删除单条记录。"""
        conn = self._get_conn()
        conn.execute("DELETE FROM detection_history WHERE id = ?", (record_id,))
        conn.commit()
        conn.close()
        return True

    def clear_all(self):
        """清空所有记录并重置自增ID。"""
        conn = self._get_conn()
        conn.execute("DELETE FROM detection_history")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='detection_history'")
        conn.commit()
        conn.close()
        return True
