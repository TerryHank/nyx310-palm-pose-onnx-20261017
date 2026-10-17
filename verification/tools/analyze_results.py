#!/usr/bin/env python3
"""Analyze JSONL captured by ainn_result_receiver without running local inference."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


COUNT_RE = re.compile(r"count:\s*(\d+)")


def parse_payload(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("payload root is not an object")
    return payload


def decode_detection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    mark = payload.get("mark") or {}
    title = mark.get("title") or {}
    match = COUNT_RE.search(str(title.get("text", "")))
    declared_count = int(match.group(1)) if match else None

    labels = mark.get("label") or []
    lines = mark.get("line") or []
    points = mark.get("point") or []
    classes: list[str] = []
    world_centers: list[str] = []
    boxes: list[list[int | float]] = []

    for index in range(0, len(labels), 2):
        item = labels[index]
        if isinstance(item, dict):
            classes.append(str(item.get("text", "")))
        if index + 1 < len(labels) and isinstance(labels[index + 1], dict):
            world_centers.append(str(labels[index + 1].get("text", "")))

    for index in range(0, len(lines), 4):
        group = lines[index : index + 4]
        if len(group) != 4 or not all(isinstance(line, list) and len(line) == 4 for line in group):
            continue
        xs = [coordinate for line in group for coordinate in (line[0], line[2])]
        ys = [coordinate for line in group for coordinate in (line[1], line[3])]
        boxes.append([min(xs), min(ys), max(xs), max(ys)])

    return {
        "payload_timestamp_ms": payload.get("timestamp"),
        "declared_count": declared_count,
        "classes": classes,
        "world_centers": world_centers,
        "boxes": boxes,
        "points": points,
        "confidence_available": False,
    }


def analyze_records(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    unique = [record for record in records if not record.get("duplicate", False)]
    parsed: list[dict[str, Any]] = []
    parse_errors = 0
    class_counts: Counter[str] = Counter()
    result_timestamps: list[int] = []
    host_elapsed: list[int] = []

    for record in unique:
        try:
            decoded = decode_detection_payload(parse_payload(record.get("payload_text", "")))
        except (json.JSONDecodeError, TypeError, ValueError):
            parse_errors += 1
            continue
        item = {
            "host_received_utc": record.get("host_received_utc"),
            "host_elapsed_ms": record.get("host_elapsed_ms"),
            "result_index": record.get("result_index"),
            "result_timestamp_ms": record.get("result_timestamp_ms"),
            **decoded,
        }
        parsed.append(item)
        class_counts.update(decoded["classes"])
        if isinstance(record.get("result_timestamp_ms"), int):
            result_timestamps.append(record["result_timestamp_ms"])
        if isinstance(record.get("host_elapsed_ms"), int):
            host_elapsed.append(record["host_elapsed_ms"])

    intervals = [later - earlier for earlier, later in zip(host_elapsed, host_elapsed[1:])]
    detection_frames = sum(1 for item in parsed if (item["declared_count"] or 0) > 0)
    signatures = {
        json.dumps([item["classes"], item["boxes"]], separators=(",", ":"), sort_keys=True)
        for item in parsed
    }
    summary = {
        "records_total": len(records),
        "records_unique": len(unique),
        "payloads_parsed": len(parsed),
        "payload_parse_errors": parse_errors,
        "detection_frames": detection_frames,
        "class_counts": dict(sorted(class_counts.items())),
        "distinct_detection_signatures": len(signatures),
        "result_timestamp_strictly_increasing": all(
            later > earlier for earlier, later in zip(result_timestamps, result_timestamps[1:])
        ),
        "host_interval_ms": {
            "samples": len(intervals),
            "min": min(intervals) if intervals else None,
            "median": statistics.median(intervals) if intervals else None,
            "max": max(intervals) if intervals else None,
        },
        "confidence_available": False,
        "confidence_note": (
            "The deployed BoxDetection serializer omits detect_result_t.prop; confidence cannot be "
            "recovered from the AINN JSON result."
        ),
    }
    return summary, parsed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number}: JSON root is not an object")
            records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.results_jsonl)
    summary, parsed = analyze_records(records)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "parsed_results.jsonl").open("w", encoding="utf-8") as stream:
        for item in parsed:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
