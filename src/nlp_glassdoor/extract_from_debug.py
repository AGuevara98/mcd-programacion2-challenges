from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def find_review_objects(text: str) -> list[str]:
    """
    Extract candidate review objects by anchoring on reviewId and known fields.
    """
    pattern = re.compile(
        r'\{'
        r'(?:(?!\{reviewId\}).)*?'
        r'"reviewId":\d+'
        r'(?:(?!\{reviewId\}).)*?'
        r'"summary":"(?:\\.|[^"\\])*"'
        r'(?:(?!\{reviewId\}).)*?'
        r'"pros":"(?:\\.|[^"\\])*"'
        r'(?:(?!\{reviewId\}).)*?'
        r'"cons":"(?:\\.|[^"\\])*"'
        r'(?:(?!\{reviewId\}).)*?'
        r'\}',
        re.DOTALL,
    )
    return pattern.findall(text)


def extract_string(obj: str, field: str) -> str | None:
    m = re.search(rf'"{re.escape(field)}":"((?:\\.|[^"\\])*)"', obj)
    if not m:
        return None
    return bytes(m.group(1), "utf-8").decode("unicode_escape").strip()


def extract_number(obj: str, field: str) -> str | None:
    m = re.search(rf'"{re.escape(field)}":(-?\d+(?:\.\d+)?)', obj)
    return m.group(1) if m else None


def extract_bool(obj: str, field: str) -> str | None:
    m = re.search(rf'"{re.escape(field)}":(true|false)', obj)
    return m.group(1) if m else None


def extract_nested_string(obj: str, parent: str, child: str) -> str | None:
    m = re.search(
        rf'"{re.escape(parent)}":\{{.*?"{re.escape(child)}":"((?:\\.|[^"\\])*)".*?\}}',
        obj,
        re.DOTALL,
    )
    if not m:
        return None
    return bytes(m.group(1), "utf-8").decode("unicode_escape").strip()


def normalize_review(obj: str, fallback_company: str, source_file: str) -> dict | None:
    review_id = extract_number(obj, "reviewId")
    pros = extract_string(obj, "pros")
    cons = extract_string(obj, "cons")

    if not review_id:
        return None
    if not pros and not cons:
        return None

    return {
        "company": extract_nested_string(obj, "employer", "shortName") or fallback_company,
        "review_title": extract_string(obj, "summary"),
        "pros": pros,
        "cons": cons,
        "rating": extract_number(obj, "ratingOverall"),
        "review_date": extract_string(obj, "reviewDateTime"),
        "employment_status": extract_string(obj, "employmentStatus"),
        "is_current_job": extract_bool(obj, "isCurrentJob"),
        "language_id": extract_number(obj, "languageId"),
        "job_title": extract_nested_string(obj, "jobTitle", "text"),
        "location": extract_nested_string(obj, "location", "name"),
        "review_id": review_id,
        "source_file": source_file,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html_path", required=True)
    parser.add_argument("--company", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    html_path = Path(args.html_path)
    text = html_path.read_text(encoding="utf-8", errors="ignore")

    objects = find_review_objects(text)
    print(f"Candidate review objects found: {len(objects)}")

    rows = []
    for obj in objects:
        row = normalize_review(obj, args.company, str(html_path))
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["review_id"])

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"Saved {len(df)} rows to {args.output_csv}")
    if not df.empty:
        print(df[["company", "review_title", "pros", "cons"]].head().to_string())


if __name__ == "__main__":
    main()