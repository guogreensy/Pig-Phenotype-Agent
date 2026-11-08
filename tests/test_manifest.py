"""数据集构建逻辑的单元测试（不联网）。

被测函数来自 ``scripts/build_dataset.py``：贪心选图、按图分组划分、分层限量。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_dataset  # noqa: E402


def row(row_id, image_id, class_id):
    from src.schema import DATASET_CLASS_MAP

    return {
        "row_id": row_id,
        "image_id": image_id,
        "_class_id": class_id,
        "_label": DATASET_CLASS_MAP[class_id],
        "_bbox": [0, 0, 200, 200],
    }


class TestSelectImages(unittest.TestCase):
    def test_covers_requested_classes(self):
        rows = []
        for index in range(10):
            rows.append(row("r_s{0}".format(index), "img_stand_{0}".format(index), 3))
            rows.append(row("r_o{0}".format(index), "img_other_{0}".format(index), 4))
        rows.append(row("r_l0", "img_lat_0", 0))
        rows.append(row("r_c0", "img_sit_0", 2))

        chosen = build_dataset.select_images(
            rows, {"standing": 2, "lying_other": 2, "lying_lateral": 1, "sitting": 1}
        )
        labels = {r["_label"] for r in rows if r["image_id"] in set(chosen)}
        self.assertTrue({"standing", "lying_other", "lying_lateral", "sitting"}.issubset(labels))
        # 每个类别配额为 1-2，且这些图基本只含单一类别，因此选图数应接近配额总和
        self.assertLessEqual(len(chosen), 6)

    def test_empty_quota(self):
        rows = [row("r1", "img1", 3)]
        self.assertEqual(build_dataset.select_images(rows, {}), [])

    def test_multi_class_image_counted_once(self):
        rows = [
            row("r1", "img1", 3),
            row("r2", "img1", 4),
            row("r3", "img2", 3),
        ]
        chosen = build_dataset.select_images(rows, {"standing": 1, "lying_other": 1})
        self.assertEqual(len(chosen), 1)  # 一张图同时满足两类


class TestAssignSplits(unittest.TestCase):
    def test_groups_by_image_and_covers_classes(self):
        rows = []
        image_ids = []
        for index in range(20):
            image_id = "img_{0:02d}".format(index)
            image_ids.append(image_id)
            class_id = 3 if index % 2 == 0 else 4
            rows.append(row("r{0}".format(index), image_id, class_id))
        # 加一张坐姿图
        image_ids.append("img_sit")
        rows.append(row("r_sit", "img_sit", 2))

        mapping = build_dataset.assign_splits(image_ids, rows, seed=42, dev_ratio=0.2)
        self.assertEqual(set(mapping), set(image_ids))
        self.assertEqual(set(mapping.values()), {"dev", "test"})
        # 每个类别至少在 dev 中出现一次
        dev_labels = {r["_label"] for r in rows if mapping[r["image_id"]] == "dev"}
        self.assertIn("sitting", dev_labels)

    def test_same_seed_is_reproducible(self):
        rows = [row("r{0}".format(i), "img{0}".format(i), 3) for i in range(10)]
        image_ids = ["img{0}".format(i) for i in range(10)]
        first = build_dataset.assign_splits(image_ids, rows, seed=7)
        second = build_dataset.assign_splits(image_ids, rows, seed=7)
        self.assertEqual(first, second)


class TestStratifiedLimit(unittest.TestCase):
    def test_caps_rows(self):
        rows = []
        for index in range(60):
            rows.append(row("r{0}".format(index), "img{0}".format(index), 3))
        for index in range(30):
            rows.append(row("s{0}".format(index), "img_s{0}".format(index), 4))
        for index in range(10):
            rows.append(row("c{0}".format(index), "img_c{0}".format(index), 2))
        kept = build_dataset.stratified_limit(rows, 40, seed=1)
        self.assertLessEqual(len(kept), 40)
        self.assertTrue(all(0 <= i < len(rows) for i in kept))

    def test_no_limit_needed(self):
        rows = [row("r0", "img0", 3), row("r1", "img1", 4)]
        self.assertEqual(build_dataset.stratified_limit(rows, 10), [0, 1])


if __name__ == "__main__":
    unittest.main()
