import argparse
import asyncio
import csv
import os
import shutil
import sys
from pathlib import Path
from playwright.async_api import async_playwright
from datetime import datetime

from src.common.config import RAW_DATA_DIR

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

async def scrape_company_pages(page, base_url, company_name, max_pages=500):
    """Scrape all pages for a single company."""
    print(f"\n{'='*70}\nCompany: {company_name}\n{'='*70}")
    
    all_reviews = []
    page_num = 1
    consecutive_empty_pages = 0
    
    while page_num <= max_pages and consecutive_empty_pages < 2:
        # Build pagination URL
        if page_num == 1:
            current_url = base_url
        else:
            # Handle both .htm and .html endings
            if base_url.endswith('.htm'):
                current_url = base_url.replace('.htm', f'_IP{page_num}.htm')
            elif base_url.endswith('.html'):
                current_url = base_url.replace('.html', f'_IP{page_num}.html')
            else:
                current_url = base_url + f'_IP{page_num}.htm'
        
        try:
            print(f"  Page {page_num:3d}: ", end="", flush=True)
            
            # Navigate with longer timeout for pagination
            await page.goto(current_url, wait_until="domcontentloaded", timeout=60000)
            
            # Wait for page to render
            await page.wait_for_timeout(2000)
            
            # Check if articles are present
            article_count = await page.evaluate("document.querySelectorAll('article').length")
            
            if article_count == 0:
                print("No articles found - stopping")
                consecutive_empty_pages += 1
                page_num += 1
                continue
            
            # Reset consecutive empty counter
            consecutive_empty_pages = 0
            
            # Scroll to load dynamic content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1500)
            
            # Extract reviews using optimized JavaScript
            reviews_data = await page.evaluate("""
                () => {
                    const reviews = [];
                    const articles = document.querySelectorAll('article');
                    
                    articles.forEach(article => {
                        // Get title from h3 or alternative selectors
                        let titleElem = article.querySelector('h3');
                        if (!titleElem) {
                            titleElem = article.querySelector('[data-test*="title"]');
                        }
                        const title = titleElem ? titleElem.textContent.trim() : '';
                        
                        if (!title || title.length === 0) return;
                        
                        // Get all paragraph elements
                        const paras = Array.from(article.querySelectorAll('p'));
                        let pros = '';
                        let cons = '';
                        
                        // Extract pros and cons from paragraphs with labels
                        for (let i = 0; i < paras.length; i++) {
                            const text = paras[i].textContent.trim();
                            
                            // Look for "Ventajas" (Spanish) or "Pros" (English)
                            if ((text.includes('Ventajas') || text.includes('Pros')) && i + 1 < paras.length) {
                                pros = paras[i + 1].textContent.trim();
                            }
                            
                            // Look for "Desventajas" (Spanish) or "Cons" (English)
                            if ((text.includes('Desventajas') || text.includes('Cons')) && i + 1 < paras.length) {
                                cons = paras[i + 1].textContent.trim();
                            }
                        }
                        
                        reviews.push({
                            review_title: title,
                            pros: pros,
                            cons: cons
                        });
                    });
                    
                    return reviews;
                }
            """)
            
            if reviews_data:
                all_reviews.extend(reviews_data)
                print(f"✓ Found {len(reviews_data):2d} reviews (Total: {len(all_reviews)})")
            else:
                print(f"✗ No reviews extracted from page (found {article_count} articles)")
                consecutive_empty_pages += 1
            
            page_num += 1
            
        except asyncio.TimeoutError:
            print(f"✗ Timeout - skipping page")
            page_num += 1
            continue
        except Exception as e:
            print(f"✗ Error: {str(e)[:50]}")
            page_num += 1
            continue
    
    print(f"\n  Total reviews for {company_name}: {len(all_reviews)}")
    return all_reviews

async def main():
    parser = argparse.ArgumentParser(description="Scrape Glassdoor reviews with manual login.")
    parser.add_argument(
        "--targets-file",
        default=str(RAW_DATA_DIR / "glassdoor_targets.csv"),
        help="CSV with company names and review URLs.",
    )
    parser.add_argument(
        "--output-file",
        default=str(RAW_DATA_DIR / "glassdoor_reviews.csv"),
        help="Destination CSV for scraped reviews.",
    )
    parser.add_argument(
        "--keep-browser-open",
        action="store_true",
        help="Leave the browser open after scraping finishes.",
    )
    args = parser.parse_args()

    # Read targets
    targets_file = Path(args.targets_file)
    targets = []
    try:
        with open(targets_file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            targets = list(reader)
        print(f"Loaded {len(targets)} companies from {targets_file}")
    except FileNotFoundError:
        print(f"Error: {targets_file} not found")
        return
    
    print(f"\nStarting scrape at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Browser: Chromium (VISIBLE)")
    
    # Check if we have a successful login flag
    login_flag = Path("./login_success.flag")
    has_valid_login = login_flag.exists()
    
    if has_valid_login:
        print("\n✓ Using saved login session")
    else:
        print("\n➜ No valid login session found - will require manual login")
    
    # Launch browser with persistent user data directory ONLY if we have a valid login
    async with async_playwright() as p:
        browser = None
        if has_valid_login:
            # Use persistent context to reuse saved session (returns BrowserContext)
            context = await p.chromium.launch_persistent_context(
                user_data_dir="./browser_data",
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage'
                ]
            )
            page = context.pages[0] if context.pages else await context.new_page()
        else:
            # Use regular browser without persistence (returns Browser, needs new context)
            browser = await p.chromium.launch(
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage'
                ]
            )
            context = await browser.new_context()
            page = await context.new_page()
        
        # Add stealth scripts to hide automation detection
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            
            Object.defineProperty(navigator, 'chrome', {
                get: () => ({
                    runtime: {}
                }),
            });
            
            Object.defineProperty(navigator, 'permissions', {
                get: () => ({
                    query: () => Promise.resolve({ state: Notification.permission })
                }),
            });
        """)
        
        # Set user agent
        await page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
        # Track popup windows and close failed ones
        def handle_popup(popup_page):
            """Handle popup windows that may appear during login"""
            async def close_if_failed():
                try:
                    await popup_page.wait_for_timeout(5000)
                    # If popup is still blank after 5 seconds, close it
                    content = await popup_page.content()
                    if not content or len(content) < 200:
                        await popup_page.close()
                        print("\n⚠ Closed unresponsive popup window")
                except:
                    try:
                        await popup_page.close()
                    except:
                        pass
            
            asyncio.create_task(close_if_failed())
        
        # Listen for popup windows
        context.on("page", handle_popup)
        
        # Navigate to login page FIRST
        print("\nNavigating to https://www.glassdoor.com.mx/index.htm?countryRedirect=true")
        try:
            await page.goto("https://www.glassdoor.com.mx/index.htm?countryRedirect=true", wait_until="domcontentloaded", timeout=60000)
        except Exception as e:
            print(f"⚠ Warning: Navigation timeout (page may still be loading): {str(e)[:50]}")
        
        print("\n" + "="*70)
        print("WAITING FOR YOUR CONFIRMATION")
        print("="*70)
        print("\nPlease log in if needed and confirm you are ready to start scraping.")
        print("The browser window is open - please interact with it as needed.")
        print("If a popup appears and won't load, it will auto-close in 5 seconds.")
        print("-" * 70)
        
        # Wait for user confirmation with timeout
        try:
            await asyncio.wait_for(asyncio.to_thread(lambda: input("\nPress Enter after you have confirmed you're ready to start scraping: ")), timeout=300.0)
        except asyncio.TimeoutError:
            print("\n✓ Continuing (5 minute timeout reached)...")
        except EOFError:
            print("\n✓ Continuing (no input available)...")
        
        # Verify login was successful by checking current URL/page state
        current_url = page.url
        print(f"\nVerifying login... Current URL: {current_url}")
        
        # Check if we're on Glassdoor (successful login redirects to home/jobs)
        if "glassdoor.com" in current_url:
            # Check if there's any indication we're logged in (could check for user menu, etc)
            try:
                # Small wait to ensure page is fully loaded
                await page.wait_for_timeout(1000)
                page_content = await page.content()
                
                # If page has substantial content, assume login successful
                if len(page_content) > 5000:
                    print("✓ Login verification: SUCCESS")
                    # Save the login success flag
                    login_flag.write_text("success")
                    print("✓ Saved login session for future use")
                else:
                    print("⚠ Login verification: UNCERTAIN (page content small)")
            except Exception as e:
                print(f"⚠ Could not verify login: {str(e)[:50]}")
        else:
            print(f"⚠ Not on Glassdoor domain, login may have failed")
        
        print("\n✓ Starting scrape...")
        await page.wait_for_timeout(2000)
        
        all_reviews = []
        
        # Scrape each company
        for idx, target in enumerate(targets, 1):
            company = target["company"].strip()
            url = target["url"].strip()
            
            print(f"\n[{idx}/{len(targets)}] Processing {company}...")
            
            try:
                reviews = await scrape_company_pages(page, url, company, max_pages=500)
                
                for review in reviews:
                    all_reviews.append({
                        "company": company,
                        "review_title": review["review_title"],
                        "pros": review["pros"],
                        "cons": review["cons"],
                        "source_url": url,
                        "page_number": review.get("page_number"),
                    })
                
                print(f"  ✓ Added {len(reviews)} reviews from {company}")
                
            except Exception as e:
                print(f"  ✗ Error processing {company}: {str(e)[:80]}")
                continue
        
        # Save results
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Total reviews scraped: {len(all_reviews)}")
        
        if all_reviews:
            output_file = Path(args.output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "company",
                        "review_title",
                        "pros",
                        "cons",
                        "source_url",
                        "page_number",
                    ],
                )
                writer.writeheader()
                writer.writerows(all_reviews)
            print(f"✓ Saved to {output_file}")
        
        print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}\n")
        
        # Cleanup: if login wasn't successful, remove the browser_data directory
        if not login_flag.exists():
            print("ℹ Cleaning up browser data (login was not successful)...")
            try:
                if Path("./browser_data").exists():
                    shutil.rmtree("./browser_data")
                    print("✓ Removed browser data")
            except Exception as e:
                print(f"⚠ Could not cleanup: {str(e)[:50]}")
        
        if args.keep_browser_open:
            print("Browser is still open. Press Ctrl+C to close it when done.")
            await page.wait_for_timeout(999999999)
        else:
            await context.close()
            if browser is not None:
                await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
