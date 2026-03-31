from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from src.common.io_utils import save_dataframe_csv


@dataclass
class ScraperConfig:
    delay_seconds: float = 2.0
    timeout_seconds: int = 20
    max_pages: int = 5
    headless: bool = False
    save_debug_html: bool = True
    debug_dir: str = "debug/glassdoor"


EN_MONTHS = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
ES_MONTHS = r"(?:ene|feb|mar|abr|may|jun|jul|ago|sep|oct|nov|dic|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)"

EN_DATE_RE = re.compile(
    rf"\b{EN_MONTHS}\s+\d{{1,2}},\s+\d{{4}}\b",
    flags=re.IGNORECASE,
)
ES_DATE_RE = re.compile(
    rf"\b\d{{1,2}}\s+de\s+{ES_MONTHS}\s+de\s+\d{{4}}\b|\b{ES_MONTHS}\s+\d{{4}}\b",
    flags=re.IGNORECASE,
)
RATING_RE = re.compile(r"\b([1-5](?:\.\d)?)\b")


def clean_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "unknown"


def normalize_glassdoor_url(url: str) -> str:
    """
    Add countryPickerRedirect=true when missing.
    Keep the original path as provided by the seed file:
    - /Reviews/...
    - /Evaluaciones/...
    """
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    if "countryPickerRedirect" not in query:
        query["countryPickerRedirect"] = "true"

    normalized = parsed._replace(query=urlencode(query))
    return urlunparse(normalized)


def build_driver(headless: bool = False) -> webdriver.Chrome:
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--lang=en-US")
    chrome_options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    )

    if headless:
        chrome_options.add_argument("--headless=new")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.set_page_load_timeout(60)
    return driver


def wait_for_page(driver: webdriver.Chrome, timeout_seconds: int) -> None:
    wait = WebDriverWait(driver, timeout_seconds)

    possible_review_locators = [
        (By.XPATH, "//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚ', 'abcdefghijklmnopqrstuvwxyzáéíóú'), 'pros')]"),
        (By.XPATH, "//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚ', 'abcdefghijklmnopqrstuvwxyzáéíóú'), 'cons')]"),
        (By.XPATH, "//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚ', 'abcdefghijklmnopqrstuvwxyzáéíóú'), 'ventajas')]"),
        (By.XPATH, "//*[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚ', 'abcdefghijklmnopqrstuvwxyzáéíóú'), 'desventajas')]"),
        (By.CSS_SELECTOR, 'li[data-test="review"]'),
        (By.CSS_SELECTOR, 'div[data-test="review"]'),
        (By.TAG_NAME, "body"),
    ]

    last_error = None
    for locator in possible_review_locators:
        try:
            wait.until(EC.presence_of_element_located(locator))
            return
        except TimeoutException as exc:
            last_error = exc

    if last_error:
        raise last_error


def close_cookie_or_modal_if_present(driver: webdriver.Chrome) -> None:
    candidates = [
        (By.ID, "onetrust-accept-btn-handler"),
        (By.XPATH, "//button[contains(., 'Accept')]"),
        (By.XPATH, "//button[contains(., 'Aceptar')]"),
        (By.XPATH, "//button[contains(., 'Got it')]"),
        (By.XPATH, "//button[contains(., 'Continue')]"),
        (By.XPATH, "//button[contains(., 'Continuar')]"),
        (By.XPATH, "//button[contains(@aria-label, 'close') or contains(@aria-label, 'Close')]"),
    ]

    for by, value in candidates:
        try:
            elements = driver.find_elements(by, value)
            for el in elements:
                if el.is_displayed() and el.is_enabled():
                    try:
                        el.click()
                        time.sleep(1)
                        return
                    except Exception:
                        continue
        except Exception:
            continue


def save_debug_html(config: ScraperConfig, company_name: str, page_num: int, html: str) -> None:
    if not config.save_debug_html:
        return

    debug_dir = Path(config.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    safe_company = make_safe_name(company_name)
    debug_path = debug_dir / f"{safe_company}_page{page_num}.html"
    debug_path.write_text(html, encoding="utf-8")


def extract_text_after_label(block_text: str, labels: list[str], stop_labels: list[str]) -> str:
    escaped_labels = "|".join(re.escape(lbl) for lbl in labels)
    escaped_stops = "|".join(re.escape(lbl) for lbl in stop_labels)

    pattern = rf"(?:{escaped_labels})\s+(.*?)(?=\s+(?:{escaped_stops})\s+|$)"
    match = re.search(pattern, block_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return clean_text(match.group(1))
    return ""


def parse_rating(text: str) -> float | None:
    for match in RATING_RE.finditer(text):
        try:
            value = float(match.group(1))
            if 1.0 <= value <= 5.0:
                return value
        except ValueError:
            continue
    return None


def parse_review_date(text: str) -> str | None:
    match = EN_DATE_RE.search(text)
    if match:
        return clean_text(match.group(0))

    match = ES_DATE_RE.search(text)
    if match:
        return clean_text(match.group(0))

    return None


def contains_review_markers(text: str) -> bool:
    text_l = text.lower()
    return (
        ("pros" in text_l and "cons" in text_l)
        or ("ventajas" in text_l and "desventajas" in text_l)
    )


def extract_title_from_block(block) -> str:
    preferred_selectors = [
        "h2",
        "h3",
        '[data-test*="review-title"]',
        '[class*="title"]',
        "a",
        "span",
    ]

    for selector in preferred_selectors:
        for node in block.select(selector):
            txt = clean_text(node.get_text(" ", strip=True))
            if not txt:
                continue
            txt_l = txt.lower()
            if txt_l in {"pros", "cons", "ventajas", "desventajas"}:
                continue
            if len(txt) <= 160:
                return txt

    for candidate in block.stripped_strings:
        txt = clean_text(candidate)
        txt_l = txt.lower()
        if not txt:
            continue
        if txt_l in {"pros", "cons", "ventajas", "desventajas"}:
            continue
        if len(txt) <= 160:
            return txt

    return ""


def parse_review_block(block) -> dict | None:
    text = clean_text(block.get_text(" ", strip=True))
    if not text or len(text) < 30:
        return None

    if not contains_review_markers(text):
        return None

    title = extract_title_from_block(block)

    pros = extract_text_after_label(
        text,
        labels=["Pros", "Ventajas"],
        stop_labels=[
            "Cons",
            "Desventajas",
            "Show more",
            "Mostrar más",
            "Helpful",
            "Útil",
            "Share",
            "Compartir",
            "Advice to Management",
            "Recomendación para la gerencia",
        ],
    )

    cons = extract_text_after_label(
        text,
        labels=["Cons", "Desventajas"],
        stop_labels=[
            "Pros",
            "Ventajas",
            "Show more",
            "Mostrar más",
            "Helpful",
            "Útil",
            "Share",
            "Compartir",
            "Advice to Management",
            "Recomendación para la gerencia",
        ],
    )

    rating = parse_rating(text)
    review_date = parse_review_date(text)

    if not pros and not cons:
        return None

    return {
        "review_title": title,
        "pros": pros,
        "cons": cons,
        "rating": rating,
        "review_date": review_date,
        "raw_block_text": text,
    }


def find_review_blocks(soup: BeautifulSoup) -> list:
    selectors = [
        'li[data-test="review"]',
        'div[data-test="review"]',
        "li.empReview",
        "article",
        "li",
        "div",
        "section",
    ]

    seen = set()
    blocks = []

    for selector in selectors:
        for node in soup.select(selector):
            text = clean_text(node.get_text(" ", strip=True))
            if len(text) < 50:
                continue
            if not contains_review_markers(text):
                continue

            key = text[:500]
            if key not in seen:
                seen.add(key)
                blocks.append(node)

    return blocks


def find_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    preferred_texts = {
        "next",
        "next page",
        "siguiente",
        "siguiente página",
    }

    for a in soup.find_all("a", href=True):
        label = clean_text(a.get_text(" ", strip=True)).lower()
        aria = clean_text(a.get("aria-label")).lower()
        href = a["href"]

        if (
            label in preferred_texts
            or "next" in aria
            or "siguiente" in aria
        ):
            return normalize_glassdoor_url(urljoin(current_url, href))

    return None


def debug_page_markers(company_name: str, html: str) -> None:
    print(f"[{company_name}] HTML length: {len(html)}")
    print(f"[{company_name}] Contains 'Ventajas': {'Ventajas' in html}")
    print(f"[{company_name}] Contains 'Desventajas': {'Desventajas' in html}")
    print(f"[{company_name}] Contains 'Pros': {'Pros' in html}")
    print(f"[{company_name}] Contains 'Cons': {'Cons' in html}")


def scrape_glassdoor_reviews(
    start_url: str,
    company_name: str,
    config: ScraperConfig | None = None,
) -> pd.DataFrame:
    config = config or ScraperConfig()
    driver = build_driver(headless=config.headless)

    all_rows: list[dict] = []
    current_url = normalize_glassdoor_url(start_url)
    visited: set[str] = set()

    try:
        for page_num in range(1, config.max_pages + 1):
            if not current_url or current_url in visited:
                break

            visited.add(current_url)
            print(f"[{company_name}] Page {page_num}: {current_url}")

            driver.get(current_url)
            wait_for_page(driver, config.timeout_seconds)
            time.sleep(config.delay_seconds)
            close_cookie_or_modal_if_present(driver)
            time.sleep(1)

            html = driver.page_source
            save_debug_html(config, company_name, page_num, html)
            debug_page_markers(company_name, html)

            soup = BeautifulSoup(html, "lxml")
            blocks = find_review_blocks(soup)
            page_rows = []

            for block in blocks:
                parsed = parse_review_block(block)
                if parsed:
                    parsed["company"] = company_name
                    parsed["source_url"] = current_url
                    parsed["page_number"] = page_num
                    page_rows.append(parsed)

            print(f"[{company_name}] Extracted {len(page_rows)} review rows on this page.")
            all_rows.extend(page_rows)

            next_url = find_next_page_url(soup, current_url)
            current_url = next_url

    finally:
        driver.quit()

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["company", "review_title", "pros", "cons"]).reset_index(drop=True)
    return df


def scrape_many_companies(
    input_csv: str,
    output_csv: str,
    max_pages: int = 5,
    headless: bool = False,
) -> pd.DataFrame:
    companies = pd.read_csv(input_csv)
    required_cols = {"company", "url"}
    missing = required_cols - set(companies.columns)
    if missing:
        raise ValueError(f"Missing required columns in seed file: {sorted(missing)}")

    all_frames = []
    config = ScraperConfig(max_pages=max_pages, headless=headless)

    for _, row in companies.iterrows():
        company = str(row["company"]).strip()
        url = str(row["url"]).strip()
        if not company or not url:
            continue

        print(f"Scraping {company} -> {url}")
        df = scrape_glassdoor_reviews(
            start_url=url,
            company_name=company,
            config=config,
        )
        all_frames.append(df)

    result = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    save_dataframe_csv(result, output_csv)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Glassdoor reviews with Selenium + BeautifulSoup")
    parser.add_argument("--input_csv", required=True, help="CSV with columns: company,url")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--max_pages", type=int, default=3, help="Pages per company")
    parser.add_argument("--headless", action="store_true", help="Run Chrome in headless mode")
    args = parser.parse_args()

    df = scrape_many_companies(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        max_pages=args.max_pages,
        headless=args.headless,
    )
    print(f"Saved {len(df)} rows to {args.output_csv}")