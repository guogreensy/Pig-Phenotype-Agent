"""构建猪只姿态数据集：选择性下载 → 逐实例裁剪 → 质量过滤 → 按图分组划分。

数据来源
--------
``anilbhujel/viewpoint-aware-pig-posture-recognition``（HuggingFace，公开，CC BY 4.0）。
只使用官方 **seenVP_test** 划分（300 图 / 2603 实例），不触碰训练划分，
避免与数据集作者的监督训练集重叠。

设计要点
--------
* **选择性下载**：先用 CSV 做贪心选图，只下载被选中的源图（约 41 张 / 8-10 MB），
  不下载整个数据集。
* **逐实例裁剪**：每行标注是一个 bbox + class_id，裁出的图只含一只猪，
  更接近"单只猪姿态识别"的任务定义。
* **质量过滤**：min 边 ≥64px、面积 ≥8192px²、长宽比 ≤4:1，逐条记录排除原因。
* **整组划分**：同一源图的全部裁切只会进入同一个 split，避免近重复泄漏。

用法::

    python scripts/build_dataset.py            # 正式构建
    python scripts/build_dataset.py --dry-run  # 只看选图与统计，不下载
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from PIL import Image  # noqa: E402

import config  # noqa: E402
from src.image_utils import check_bbox_quality, crop_with_margin, save_jpeg  # noqa: E402
from src.schema import DATASET_CLASS_MAP, DATASET_CLASS_NAME, POSTURES  # noqa: E402

REPO = "anilbhujel/viewpoint-aware-pig-posture-recognition"
#: 主站与镜像站。实测 huggingface.co 在国内网络下会间歇性不可达，
#: hf-mirror.com 返回同一文件（字节数一致），因此优先镜像、主站兜底。
HOSTS = ["https://hf-mirror.com", "https://huggingface.co"]
SUBDIR = "viewpoint_aware_pig_posture_recognition"
SPLIT_CSV = "seenVP_test.csv"
SPLIT_IMG_DIR = "seenVP_test_images"

CACHE_DIR = config.DATA_DIR / "datasets" / "_cache"
CSV_CACHE = CACHE_DIR / SPLIT_CSV
IMG_CACHE = CACHE_DIR / SPLIT_IMG_DIR

#: 贪心选图的类别配额（以规范类别计）。
DEFAULT_QUOTA = {
    "standing": 40,
    "lying_other": 40,
    "lying_lateral": 50,
    "sitting": 25,
}
TARGET_TEST_MAX = 200
TARGET_TEST_MIN = 120
SEED = 20261019
DEV_RATIO = 0.2

HEADERS = {"User-Agent": "pig-posture-course-project/1.0"}

#: 记录实际用到的下载主机，写入清单便于复现
USED_HOSTS: Set[str] = set()


# --------------------------------------------------------------------------- io
def remote_urls(relative_path: str) -> List[str]:
    """按优先级生成候选下载地址（镜像优先）。"""
    path = "{0}/{1}".format(SUBDIR, relative_path)
    return ["{0}/datasets/{1}/resolve/main/{2}".format(host, REPO, path) for host in HOSTS]


def download_any(relative_path: str, target: Path, retries: int = 6,
                 timeout: int = 60) -> bytes:
    """依次尝试镜像与主站，成功即缓存到 target。

    镜像站会返回 429 限流，因此对 429 做指数退避（并尊重 Retry-After），
    而不是立刻换主机——主站在本网络下通常不可达。
    """
    if target.is_file() and target.stat().st_size > 0:
        return target.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in remote_urls(relative_path):
        host = url.split("/")[2]
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, headers=HEADERS, timeout=timeout)
                if response.status_code == 200 and response.content:
                    target.write_bytes(response.content)
                    USED_HOSTS.add(host)
                    return response.content
                if response.status_code == 429:
                    delay = min(30.0, 5.0 * attempt) + random.uniform(0, 2.0)
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, min(60.0, float(retry_after)))
                        except ValueError:
                            pass
                    errors.append("{0}: HTTP 429(等待{1:.0f}s)".format(host, delay))
                    time.sleep(delay)
                    continue
                errors.append("{0}: HTTP {1}".format(host, response.status_code))
                break  # 其它状态码换下一个主机
            except requests.RequestException as exc:
                errors.append("{0}: {1}".format(host, exc.__class__.__name__))
                time.sleep(min(10, 2 ** attempt) + random.uniform(0, 1.0))
    raise RuntimeError("下载失败 {0}：{1}".format(relative_path, "；".join(errors[-4:])))


def load_split_rows() -> List[Dict[str, str]]:
    """读取 seenVP_test.csv 并解析 bbox。"""
    content = download_any(SPLIT_CSV, CSV_CACHE)
    text = content.decode("utf-8")
    rows: List[Dict[str, str]] = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            bbox = json.loads(row["bbox"])
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        class_id = int(row["class_id"])
        label = DATASET_CLASS_MAP.get(class_id)
        if label is None:
            continue
        row["_bbox"] = bbox
        row["_class_id"] = class_id
        row["_label"] = label
        rows.append(row)
    return rows


# ------------------------------------------------------------------- selection
def select_images(rows: Sequence[Dict[str, str]], quota: Dict[str, int]) -> List[str]:
    """贪心选择源图：优先覆盖缺口最大的类别。"""
    per_image: Dict[str, Set[str]] = {}
    for row in rows:
        per_image.setdefault(row["image_id"], set()).add(row["_label"])

    need = dict(quota)
    remaining = set(per_image)
    chosen: List[str] = []
    while any(v > 0 for v in need.values()) and remaining:
        best, best_score = None, 0.0
        # 排序遍历：set 的迭代顺序受字符串哈希随机化影响，会导致选图不可复现
        for image_id in sorted(remaining):
            labels = per_image[image_id]
            score = sum(3.0 for label in labels if need.get(label, 0) > 0)
            score += sum(1.0 for label in labels if need.get(label, 0) >= 5)
            score += 0.01 * len(labels)
            if score > best_score:
                best, best_score = image_id, score
        if best is None:
            break
        chosen.append(best)
        remaining.discard(best)
        for label in per_image[best]:
            if need.get(label, 0) > 0:
                need[label] -= 1
    if any(v > 0 for v in need.values()):
        print("    警告：以下类别配额未满足：{0}".format(
            {k: v for k, v in need.items() if v > 0}))
    return chosen


def assign_splits(image_ids: Sequence[str], rows: Sequence[Dict[str, str]],
                  seed: int = SEED, dev_ratio: float = DEV_RATIO) -> Dict[str, str]:
    """按源图分组划分，保证每个类别在 dev 中至少有一张图（若可能）。"""
    per_image: Dict[str, Set[str]] = {}
    for row in rows:
        per_image.setdefault(row["image_id"], set()).add(row["_label"])

    image_list = list(image_ids)
    rng = random.Random(seed)
    rng.shuffle(image_list)

    dev: Set[str] = set()
    # 先保证类别覆盖：每个类别挑 ceil(ratio * 出现该类的图数) 张
    for label in POSTURES:
        containing = [img for img in image_list if label in per_image[img]]
        if not containing:
            continue
        want = max(1, int(math.ceil(dev_ratio * len(containing))))
        for img in containing:
            if len([d for d in dev if label in per_image[d]]) >= want:
                break
            dev.add(img)

    # 再按比例补足总 dev 图数
    target_dev_images = max(len(dev), int(round(dev_ratio * len(image_list))))
    for img in image_list:
        if len(dev) >= target_dev_images:
            break
        dev.add(img)

    return {img: ("dev" if img in dev else "test") for img in image_ids}


def _label_of(row: Dict[str, object]) -> str:
    """兼容两种记录形态：CSV 行用 ``_label``，构建中间记录用 ``label``。"""
    return str(row.get("_label") or row.get("label") or "")


def stratified_limit(rows: Sequence[Dict[str, object]], max_n: int, seed: int = SEED) -> List[int]:
    """按类别比例分层限量，返回要保留的行下标。"""
    if len(rows) <= max_n:
        return list(range(len(rows)))
    by_label: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        by_label.setdefault(_label_of(row), []).append(index)
    rng = random.Random(seed + 1)
    total = len(rows)
    keep: List[int] = []
    for label, indexes in by_label.items():
        rng.shuffle(indexes)
        quota = max(1, int(round(max_n * len(indexes) / total)))
        keep.extend(indexes[:quota])
    # 超出则裁剪（从数量最多的类别里去掉尾部）
    keep = sorted(keep)
    if len(keep) > max_n:
        label_of = {index: _label_of(rows[index]) for index in keep}
        keep.sort(key=lambda i: (label_of[i], i))
        keep = keep[:max_n]
    return sorted(keep)


# ------------------------------------------------------------------------ main
def build(quota: Dict[str, int], target_test_max: int, dry_run: bool) -> int:
    config.ensure_dirs()
    print("[1/6] 读取标注 CSV …")
    rows = load_split_rows()
    print("    实例数 {0}，源图数 {1}".format(len(rows), len({r['image_id'] for r in rows})))

    print("[2/6] 贪心选择源图 …")
    image_ids = select_images(rows, quota)
    selected_rows = [row for row in rows if row["image_id"] in set(image_ids)]
    print("    选中源图 {0} 张，覆盖实例 {1} 个".format(len(image_ids), len(selected_rows)))

    if dry_run:
        counts: Dict[str, int] = {}
        for row in selected_rows:
            counts[row["_label"]] = counts.get(row["_label"], 0) + 1
        print("    实例类别分布：{0}".format(json.dumps(counts, ensure_ascii=False, sort_keys=True)))
        print("    [dry-run] 未下载、未裁剪。")
        return 0

    print("[3/6] 下载源图（已缓存则跳过；镜像限流时自动退避重试）…")
    total_bytes = 0
    failures: List[str] = []
    for index, image_id in enumerate(sorted(image_ids), 1):
        target = IMG_CACHE / image_id
        try:
            content = download_any("{0}/{1}".format(SPLIT_IMG_DIR, image_id), target)
        except RuntimeError as exc:
            failures.append(image_id)
            print("    [{0}/{1}] 失败：{2}".format(index, len(image_ids), str(exc)[:150]))
            continue
        total_bytes += len(content)
        if index % 5 == 0 or index == len(image_ids):
            print("    已获取 {0}/{1}（本次 {2:.1f} MB）".format(
                index, len(image_ids), total_bytes / 1048576))
        time.sleep(0.5)  # 轻微间隔，降低镜像站限流概率
    if failures:
        print("    有 {0} 张源图下载失败；重跑本脚本会自动续传（已缓存的不会重复下载）。".format(
            len(failures)))
        print("    失败清单：{0}".format("、".join(failures[:10])))
        selected_rows = [r for r in selected_rows if r["image_id"] not in set(failures)]
    print("    源图合计 {0:.1f} MB".format(total_bytes / 1048576))

    print("[4/6] 逐实例裁剪与质量过滤 …")
    records: List[Dict[str, object]] = []
    excluded: Dict[str, int] = {}
    for row in selected_rows:
        image_path = IMG_CACHE / row["image_id"]
        try:
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
        except Exception as exc:
            excluded["read_error"] = excluded.get("read_error", 0) + 1
            continue
        bbox = row["_bbox"]
        width = int(round(float(bbox[2])))
        height = int(round(float(bbox[3])))
        reason = check_bbox_quality(width, height)
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        record = {
            "sample_id": row["row_id"],
            "source_image": row["image_id"],
            "row_id": row["row_id"],
            "bbox": json.dumps(bbox),
            "class_id": row["_class_id"],
            "class_name": DATASET_CLASS_NAME.get(row["_class_id"], ""),
            "label": row["_label"],
            "crop_w": width,
            "crop_h": height,
            "filter_reason": reason or "",
            "_bbox_list": bbox,
        }
        records.append(record)

    kept = [r for r in records if not r["filter_reason"]]
    print("    裁切候选 {0}，合格 {1}，排除 {2}".format(len(records), len(kept), len(records) - len(kept)))
    if excluded:
        print("    排除原因统计：{0}".format(json.dumps(excluded, ensure_ascii=False, sort_keys=True)))

    print("[5/6] 按源图分组划分 dev/test …")
    split_map = assign_splits(image_ids, selected_rows)
    for record in kept:
        record["split"] = split_map[record["source_image"]]

    dev_rows = [r for r in kept if r["split"] == "dev"]
    test_rows = [r for r in kept if r["split"] == "test"]
    keep_indexes = stratified_limit(test_rows, target_test_max)
    test_rows = [test_rows[i] for i in keep_indexes]
    print("    dev 图数 {0} / test 图数 {1}".format(
        len({r['source_image'] for r in dev_rows}), len({r['source_image'] for r in test_rows})))

    if len(test_rows) < TARGET_TEST_MIN:
        print("    注意：test 仅 {0} 条，低于目标下限 {1}，报告中需说明样本量限制。".format(
            len(test_rows), TARGET_TEST_MIN))

    final_rows = dev_rows + test_rows

    print("[6/6] 写出图片与清单 …")
    # 清空旧的裁切样本，避免上一次构建（选图可能不同）的残留文件混入新数据集
    if config.SAMPLES_DIR.exists():
        removed = 0
        for old in config.SAMPLES_DIR.rglob("*.jpg"):
            old.unlink()
            removed += 1
        if removed:
            print("    已清理旧裁切样本 {0} 张".format(removed))

    for record in final_rows:
        image_path = IMG_CACHE / record["source_image"]
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            crop = crop_with_margin(image, record["_bbox_list"], margin_ratio=0.08)
        target = config.SAMPLES_DIR / record["split"] / record["label"] / "{0}.jpg".format(
            record["sample_id"])
        save_jpeg(crop, target, quality=92)
        record["image_path"] = str(target.relative_to(ROOT)).replace("\\", "/")
        record["sha256"] = hashlib.sha256(crop.tobytes()).hexdigest()
        record["source_url"] = "{0}/datasets/{1}/resolve/main/{2}/{3}/{4}".format(
            sorted(USED_HOSTS)[0] if USED_HOSTS else HOSTS[0], REPO, SUBDIR, SPLIT_IMG_DIR,
            record["source_image"])

    fields = ["sample_id", "source_image", "row_id", "bbox", "class_id", "class_name",
              "label", "split", "image_path", "crop_w", "crop_h", "filter_reason",
              "sha256", "source_url"]
    with config.MANIFEST_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in final_rows:
            writer.writerow(record)

    with config.LABELS_PATH.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "image_path", "true_label", "split",
                         "reviewer", "review_status", "notes"])
        for record in final_rows:
            writer.writerow([record["sample_id"], record["image_path"], record["label"],
                             record["split"], "", "unreviewed", ""])

    # 汇总
    on_disk = sorted(config.SAMPLES_DIR.rglob("*.jpg"))
    if len(on_disk) != len(final_rows):
        print("    警告：磁盘上有 {0} 张裁切图，与清单 {1} 行不一致，请检查。".format(
            len(on_disk), len(final_rows)))
    else:
        print("    一致性检查通过：裁剪图 {0} 张 = 清单 {1} 行".format(len(on_disk), len(final_rows)))

    summary = {}
    for split in ("dev", "test"):
        subset = [r for r in final_rows if r["split"] == split]
        counts: Dict[str, int] = {}
        for record in subset:
            counts[record["label"]] = counts.get(record["label"], 0) + 1
        summary[split] = {"n": len(subset), "labels": counts,
                          "images": len({r["source_image"] for r in subset})}
    print("\n" + "=" * 60)
    print("数据集构建完成")
    print("=" * 60)
    for split, info in summary.items():
        print("  {0:<5} {1:>3} 条 / {2} 张源图  {3}".format(
            split, info["n"], info["images"], json.dumps(info["labels"], ensure_ascii=False, sort_keys=True)))
    print("  清单: {0}".format(config.MANIFEST_PATH.relative_to(ROOT)))
    print("  标签: {0}".format(config.LABELS_PATH.relative_to(ROOT)))
    print("  许可: CC BY 4.0 —— 使用时须注明数据集出处（见 data/README.md）")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="构建猪只姿态数据集（选择性下载）")
    parser.add_argument("--dry-run", action="store_true", help="只选图与统计，不下载")
    parser.add_argument("--target-test-max", type=int, default=TARGET_TEST_MAX,
                        help="test 集上限（默认 200）")
    for label, default in DEFAULT_QUOTA.items():
        parser.add_argument("--quota-{0}".format(label.replace("_", "-")), type=int,
                            default=default, help="{0} 选图配额".format(label))
    args = parser.parse_args(argv)

    quota = {}
    for label in DEFAULT_QUOTA:
        quota[label] = getattr(args, "quota_{0}".format(label))
    return build(quota, args.target_test_max, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
