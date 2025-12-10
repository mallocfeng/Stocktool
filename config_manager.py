from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class UserConfig:
    csv_path: str = ""
    formula: str = ""


class ConfigManager:
    def __init__(self, base_dir: Optional[str] = None, filename: str = "user_config.json"):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.path = os.path.join(self.base_dir, filename)

    def load(self) -> UserConfig:
        if not os.path.exists(self.path):
            return UserConfig()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return UserConfig(
                csv_path=data.get("csv_path", ""),
                formula=data.get("formula", ""),
            )
        except Exception:
            return UserConfig()

    def save(self, config: UserConfig) -> None:
        data = {"csv_path": config.csv_path, "formula": config.formula}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
