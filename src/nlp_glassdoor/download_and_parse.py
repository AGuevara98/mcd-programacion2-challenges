from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def build_driver(headless: bool = False) -> webdriver.Chrome:
    options = Options()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=es-MX")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/146.0.0.0 Safari/537.36"
    )
    if headless:
        options.add_argument("--headless=new")

    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="ignore")


def is_blocked_html(html: str) -> bool:
    lowered = html.lower()
    block_signals = [
        "just a moment",
        "help us protect glassdoor",
        "ayúdanos a proteger glassdoor",
        "verification successful",
        "cf-turnstile",
        "cloudflare",
        "challenge-platform",
        "enable javascript and cookies to continue",
    ]
    return any(signal in lowered for signal in block_signals)


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)

    value = value.replace("\\u0026quot;", '"')
    value = value.replace("\\u0026#x27;", "'")
    value = value.replace("\\u003c", "<")
    value = value.replace("\\u003e", ">")
    value = value.replace("\\/", "/")
    value = value.replace("\\n", " ")
    value = value.replace('\\"', '"')
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_json_candidates(html: str) -> list[str]:
    """
    Look for script blocks or inline payloads that include reviews/pros/cons.
    """
    candidates: list[str] = []

    script_blocks = re.findall(
        r"<script[^>]*>(.*?)</script>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    for block in script_blocks:
        lowered = block.lower()
        if '"reviews"' in lowered or '"pros"' in lowered or '"cons"' in lowered:
            candidates.append(block)

    if '"reviews"' in html or '"pros"' in html or '"cons"' in html:
        candidates.append(html)

    return candidates


def extract_reviews_array_from_text(text: str) -> list[dict]:
    """
    Try to find a JSON array after a "reviews": marker and decode it.
    """
    marker_match = re.search(r'"reviews"\s*:\s*\[', text)
    if not marker_match:
        return []

    start = marker_match.end() - 1  # points to '['
    depth = 0
    in_string = False
    escape = False
    end = None

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end is None:
        return []

    raw_array = text[start : end + 1]

    try:
        data = json.loads(raw_array)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        return []

    return []


def extract_reviews_from_payloads(html: str) -> list[dict]:
    """
    Main parser: scan candidate blocks and try to decode reviews arrays.
    """
    all_reviews: list[dict] = []

    for candidate in extract_json_candidates(html):
        reviews = extract_reviews_array_from_text(candidate)
        if reviews:
            all_reviews.extend(reviews)

    # deduplicate by reviewId when possible
    seen = set()
    deduped = []
    for review in all_reviews:
        key = review.get("reviewId") or (
            review.get("summary"),
            review.get("pros"),
            review.get("cons"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(review)

    return deduped


def normalize_review(review: dict, company_name: str, source_url: str) -> dict | None:
    pros = review.get("pros")
    cons = review.get("cons")

    if not pros and not cons:
        return None

    employer = review.get("employer") or {}
    job_title = review.get("jobTitle") or {}
    location = review.get("location") or {}

    return {
        "company": clean_text(employer.get("shortName")) or company_name,
        "review_title": clean_text(review.get("summary")),
        "pros": clean_text(pros),
        "cons": clean_text(cons),
        "rating": review.get("ratingOverall"),
        "review_date": clean_text(review.get("reviewDateTime")),
        "employment_status": clean_text(review.get("employmentStatus")),
        "is_current_job": review.get("isCurrentJob"),
        "language_id": review.get("languageId"),
        "job_title": clean_text(job_title.get("text")),
        "location": clean_text(location.get("name")),
        "review_id": review.get("reviewId"),
        "source_url": source_url,
    }


def download_html(url: str, html_output: Path, wait_seconds: int, headless: bool) -> str:
    driver = build_driver(headless=headless)

    try:
        driver.get(url)

        print("\n=== MANUAL STEP REQUIRED ===")
        print("1. Solve the Cloudflare / Glassdoor challenge in the browser")
        print("2. Make sure you can SEE actual reviews")
        print("3. Press ENTER here when ready...\n")
        input("Press ENTER after solving the challenge...")

        # Extra buffer after manual solve
        time.sleep(wait_seconds)

        # Scroll to trigger lazy loading
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)

        html = driver.page_source
        save_text(html_output, html)

        print(f"[DEBUG] Final page title: {driver.title}")
        print(f"[DEBUG] Final URL: {driver.current_url}")

        return html

    finally:
        driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Glassdoor page HTML first, then parse it."
    )
    parser.add_argument("--url", required=True, help="Glassdoor reviews URL")
    parser.add_argument("--company", required=True, help="Fallback company name")
    parser.add_argument("--html_output", required=True, help="Path to save raw HTML")
    parser.add_argument("--csv_output", required=True, help="Path to save parsed CSV")
    parser.add_argument("--wait_seconds", type=int, default=8, help="Seconds to wait after load")
    parser.add_argument("--headless", action="store_true", help="Run Chrome headless")
    args = parser.parse_args()

    html_output = Path(args.html_output)
    csv_output = Path(args.csv_output)

    print(f"Downloading HTML from: {args.url}")
    html = download_html(
        url=args.url,
        html_output=html_output,
        wait_seconds=args.wait_seconds,
        headless=args.headless,
    )
    print(f"Saved HTML to: {html_output}")

    if is_blocked_html(html):
        print("Blocked page detected. HTML was saved, but no reviews can be parsed from this session.")
        print("Open the saved HTML and verify whether it is a challenge page.")
        pd.DataFrame(
            columns=[
                "company",
                "review_title",
                "pros",
                "cons",
                "rating",
                "review_date",
                "employment_status",
                "is_current_job",
                "language_id",
                "job_title",
                "location",
                "review_id",
                "source_url",
            ]
        ).to_csv(csv_output, index=False, encoding="utf-8")
        print(f"Saved empty CSV to: {csv_output}")
        return

    raw_reviews = extract_reviews_from_payloads(html)
    print(f"Raw reviews found in payloads: {len(raw_reviews)}")

    rows = []
    for review in raw_reviews:
        row = normalize_review(
            review=review,
            company_name=args.company,
            source_url=args.url,
        )
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        subset = ["company", "review_title", "pros", "cons"]
        if "review_id" in df.columns:
            subset.append("review_id")
        df = df.drop_duplicates(subset=[c for c in subset if c in df.columns])

    csv_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_output, index=False, encoding="utf-8")

    print(f"Normalized review rows: {len(df)}")
    print(f"Saved CSV to: {csv_output}")
    if not df.empty:
        preview_cols = [c for c in ["company", "review_title", "pros", "cons", "location"] if c in df.columns]
        print(df[preview_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()