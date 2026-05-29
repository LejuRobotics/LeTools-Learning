# Copyright (C) 2025-2026 LejuRobotics.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# ---
#
# This project includes code from LeRobot (https://github.com/huggingface/lerobot),
# which is licensed under the Apache License, Version 2.0.

import json
import os
import shutil
import subprocess
import rosbag
from pathlib import Path
from typing import Any

def load_json(fpath: Path) -> Any:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def reindex_rosbag(bag_file)->str:
    bag_file = str(bag_file)
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            return bag_file
    except rosbag.bag.ROSBagException as e:
        print(f"Error reading '{bag_file}': {e}")
    
    # bag is corrupted.
    try:
        print(f"Warning: The bag file '{bag_file}' is corrupted, reindexing...")
        command = [
            "rosbag",
            "reindex",
            bag_file
            ]
        subprocess.run(command, check=True)
        if bag_file.endswith(".bag.active"):
            base_name = bag_file.replace(".bag.active", "")
            rosbag_orig_file = f"{base_name}.bag.orig.active"
        elif bag_file.endswith(".bag"):
            base_name = bag_file.replace(".bag", "")
            rosbag_orig_file = f"{bag_file}.orig.bag"
        if os.path.exists(rosbag_orig_file):
            os.remove(rosbag_orig_file)
        if os.path.exists(bag_file):
            rosbag_file = f"{base_name}.bag"
            shutil.move(bag_file, rosbag_file)
            return rosbag_file
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error reindexing bag file: {e}")
        return None
    
