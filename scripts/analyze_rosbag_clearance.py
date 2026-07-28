#!/usr/bin/env python3
"""Correlate LaserScan clearance with commands in rosbag2 DB3 without ROS."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sqlite3
from pathlib import Path

from export_rosbag_motion import CdrReader, messages, parse_twist


def parse_laser_scan(data: bytes) -> dict[str, object]:
    reader = CdrReader(data)
    stamp_sec, stamp_nanosec = reader.int32(), reader.uint32()
    frame_id = reader.string()
    values = [reader.unpack("f", 4) for _ in range(7)]
    angle_min, angle_max, angle_increment, time_increment, scan_time, range_min, range_max = values
    count = reader.uint32()
    ranges = [reader.unpack("f", 4) for _ in range(count)]
    intensity_count = reader.uint32()
    intensities = [reader.unpack("f", 4) for _ in range(intensity_count)]
    return {
        "stamp_ns": stamp_sec * 1_000_000_000 + stamp_nanosec, "frame_id": frame_id,
        "angle_min": angle_min, "angle_max": angle_max, "angle_increment": angle_increment,
        "time_increment": time_increment, "scan_time": scan_time,
        "range_min": range_min, "range_max": range_max,
        "ranges": ranges, "intensities": intensities,
    }


def finite_min(values: list[float], lower: float, upper: float) -> float | None:
    valid = [value for value in values if math.isfinite(value) and lower <= value <= upper]
    return min(valid) if valid else None


def proximity_episodes(rows: list[dict[str, object]], threshold_m: float = 0.5,
                       max_gap_s: float = 0.2) -> list[dict[str, float]]:
    close = [row for row in rows if row["front_min_range_m"] is not None
             and row["front_min_range_m"] < threshold_m]
    episodes: list[list[dict[str, object]]] = []
    for row in close:
        if not episodes or (row["recorded_ns"] - episodes[-1][-1]["recorded_ns"]) / 1e9 > max_gap_s:
            episodes.append([row])
        else:
            episodes[-1].append(row)
    result = []
    for episode in episodes:
        closest = min(episode, key=lambda row: row["front_min_range_m"])
        result.append({
            "start_ns": episode[0]["recorded_ns"], "end_ns": episode[-1]["recorded_ns"],
            "duration_s": (episode[-1]["recorded_ns"] - episode[0]["recorded_ns"]) / 1e9,
            "samples": len(episode), "min_front_range_m": closest["front_min_range_m"],
            "command_speed_at_closest_mps": closest["command_speed_mps"],
            "command_age_at_closest_ms": closest["command_age_ms"],
        })
    return result


def analyze(
    db: Path,
    output_csv: Path,
    scan_topic: str = "/scan",
    command_topic: str = "/mobile_base_controller/cmd_vel",
) -> dict[str, object]:
    connection = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        commands = [(timestamp, parse_twist(payload))
                    for timestamp, payload in messages(connection, command_topic)]
        command_times = [row[0] for row in commands]
        rows = []
        for recorded_ns, payload in messages(connection, scan_topic):
            scan = parse_laser_scan(payload)
            nearest = min(bisect.bisect_left(command_times, recorded_ns), len(commands) - 1)
            if nearest and abs(command_times[nearest - 1] - recorded_ns) < abs(command_times[nearest] - recorded_ns):
                nearest -= 1
            command_ns, command = commands[nearest]
            front = [value for index, value in enumerate(scan["ranges"])
                     if abs(scan["angle_min"] + index * scan["angle_increment"]) <= math.pi / 6]
            speed = math.hypot(command["linear_x"], command["linear_y"])
            rows.append({
                "recorded_ns": recorded_ns, "min_range_m": finite_min(scan["ranges"], scan["range_min"], scan["range_max"]),
                "front_min_range_m": finite_min(front, scan["range_min"], scan["range_max"]),
                "command_age_ms": abs(recorded_ns - command_ns) / 1e6,
                "command_speed_mps": speed, "command_yaw_rate_rps": command["angular_z"],
            })
    finally:
        connection.close()
    if not rows:
        raise ValueError("scan topic contains no messages")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["front_min_range_m"] is not None]
    episodes = proximity_episodes(valid)
    paired = [row for row in valid if row["command_age_ms"] <= 200.0]
    low = [row for row in paired if row["command_speed_mps"] < 0.03]
    moving = [row for row in paired if row["command_speed_mps"] >= 0.03]
    mean = lambda group, key: sum(row[key] for row in group) / len(group) if group else None
    return {
        "bag": db.parent.name, "database": str(db),
        "scan_topic": scan_topic, "command_topic": command_topic,
        "scan_samples": len(rows),
        "valid_front_samples": len(valid), "paired_command_samples": len(paired),
        "command_pair_ratio": len(paired) / len(valid),
        "mean_paired_command_age_ms": mean(paired, "command_age_ms"),
        "mean_front_clearance_m": mean(valid, "front_min_range_m"),
        "mean_front_clearance_low_speed_m": mean(low, "front_min_range_m"),
        "mean_front_clearance_moving_m": mean(moving, "front_min_range_m"),
        "front_below_0_5m_ratio": sum(row["front_min_range_m"] < 0.5 for row in valid) / len(valid),
        "front_below_1_0m_ratio": sum(row["front_min_range_m"] < 1.0 for row in valid) / len(valid),
        "low_speed_paired_ratio": len(low) / len(paired) if paired else None,
        "proximity_episode_count": len(episodes),
        "proximity_total_time_s": sum(episode["duration_s"] for episode in episodes),
        "proximity_longest_time_s": max((episode["duration_s"] for episode in episodes), default=0.0),
        "minimum_front_range_m": min(row["front_min_range_m"] for row in valid),
        "closest_command_speed_mps": (
            min(episodes, key=lambda item: item["min_front_range_m"])["command_speed_at_closest_mps"]
            if episodes and min(episodes, key=lambda item: item["min_front_range_m"])["command_age_at_closest_ms"] <= 200.0
            else None
        ),
        "proximity_episodes": episodes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("build/rosbag_clearance"))
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument(
        "--command-topic", default="/mobile_base_controller/cmd_vel"
    )
    args = parser.parse_args()
    reports = []
    for db in args.databases:
        report = analyze(
            db.resolve(),
            args.output_dir / f"{db.parent.name}_scan_commands.csv",
            scan_topic=args.scan_topic,
            command_topic=args.command_topic,
        )
        reports.append(report)
        print(f"{report['bag']}: {report['scan_samples']} scans, mean front {report['mean_front_clearance_m']:.2f} m")
    (args.output_dir / "clearance_summary.json").write_text(json.dumps({"bags": reports}, indent=2) + "\n")
    fields = list(reports[0].keys())
    with (args.output_dir / "clearance_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        csv_fields = [field for field in fields if field != "proximity_episodes"]
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({field: report[field] for field in csv_fields} for report in reports)
    lines = ["# Offline Clearance Comparison", "",
             "A proximity episode is a contiguous period with front clearance below 0.5 m; gaps above 0.2 s split events.", "",
             "| Bag | Mean front (m) | Minimum (m) | Episodes | Total close time (s) | Longest (s) | Speed at closest (m/s) | Command pair coverage |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        speed = report["closest_command_speed_mps"]
        speed_text = f"{speed:.3f}" if speed is not None else "n/a"
        lines.append(f"| {report['bag']} | {report['mean_front_clearance_m']:.2f} | "
                     f"{report['minimum_front_range_m']:.2f} | {report['proximity_episode_count']} | "
                     f"{report['proximity_total_time_s']:.1f} | {report['proximity_longest_time_s']:.1f} | "
                     f"{speed_text} | {report['command_pair_ratio']:.1%} |")
    lines += ["", "Command speed at closest is reported only when the nearest command is within 200 ms.",
              "Values equal to the scanner range minimum are lower-bound/saturation readings, not precise obstacle distances."]
    (args.output_dir / "clearance_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
