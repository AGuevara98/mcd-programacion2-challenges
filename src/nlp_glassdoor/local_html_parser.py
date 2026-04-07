import os
import csv
import glob
import re
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Enforce consistent language detection results
DetectorFactory.seed = 0

def extract_text_after_label(container, labels):
    """
    Looks for a specific label (like 'Pros' or 'Ventajas') inside the container,
    and attempts to extract the text immediately following it.
    """
    for label in labels:
        # Find the tag that contains the exact label text (case-insensitive)
        label_tag = container.find(string=re.compile(rf'^{label}:?$', re.IGNORECASE))
        
        if label_tag:
            # Usually, the actual text is in the next sibling element, 
            # or wrapped in a span right next to the label.
            parent = label_tag.parent
            
            # Try finding the next sibling of the parent element
            next_element = parent.find_next_sibling()
            if next_element:
                return next_element.get_text(separator=" ", strip=True)
            
            # Fallback: just get all text from the parent's parent and strip the label
            parent_text = parent.parent.get_text(separator=" ", strip=True)
            return parent_text.replace(label, "").strip(" :")
            
    return 'N/A'

def parse_local_glassdoor_files(folder_path, output_csv):
    fieldnames = ['company', 'review_title', 'pros', 'cons', 'rating', 'review_date', 'language']
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        
        search_pattern = os.path.join(folder_path, '*.htm*')
        html_files = glob.glob(search_pattern)
        
        if not html_files:
            print(f"No HTML files found in '{folder_path}'. Please check the path.")
            return
            
        print(f"Found {len(html_files)} files to process.")
        
        for file_path in html_files:
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}...")
            
            # --- DYNAMIC COMPANY NAME EXTRACTION ---
            # 1. Try to extract from the filename first (most reliable for Glassdoor saves)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Check for English filename pattern (e.g., "Airbnb Reviews (2,342)_ Pros...")
            if " Reviews" in name_without_ext:
                company_name = name_without_ext.split(" Reviews")[0].strip()
                
            # Check for Spanish filename pattern (e.g., "Evaluaciones de IBM_ ¿cómo...")
            elif name_without_ext.lower().startswith("evaluaciones de "):
                # Remove the prefix
                cleaned = name_without_ext[16:] 
                # Split at the first underscore or colon and take the first part
                company_name = re.split(r'[_:|]', cleaned)[0].strip()
                
            else:
                # Fallback: Just use the filename without the .htm extension
                company_name = name_without_ext.strip()
            # ----------------------------------------
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                
                # --- NEW STRATEGY: Find containers by content, not classes ---
                # We look for any tag (usually a div or li) that contains BOTH the words Pros and Cons (or Ventajas/Desventajas)
                review_containers = []
                
                # Find all potential text elements
                all_pros = soup.find_all(string=re.compile(r'^(Pros|Ventajas):?$', re.IGNORECASE))
                
                for pro_text in all_pros:
                    # Go up the HTML tree to find the overarching review card (usually 3 to 6 levels up)
                    parent = pro_text.parent
                    for _ in range(6): 
                        if parent and parent.name in ['div', 'li']:
                            parent_text = parent.get_text()
                            # If this container also has "Cons" or "Desventajas", it's the full review card
                            if 'Cons' in parent_text or 'Desventajas' in parent_text:
                                if parent not in review_containers:
                                    review_containers.append(parent)
                                break
                        if parent:
                            parent = parent.parent

                print(f"  -> Found {len(review_containers)} reviews in this file.")

                for review in review_containers:
                    # 1. Extract Pros and Cons using our new text-based extraction
                    pros = extract_text_after_label(review, ['Pros', 'Ventajas'])
                    cons = extract_text_after_label(review, ['Cons', 'Desventajas'])
                    
                    # 2. Extract Title (Usually an <h2> or heavily styled <span> at the top of the card)
                    title_tag = review.find('h2')
                    title = title_tag.get_text(strip=True) if title_tag else 'N/A'
                    
                    # 3. Extract Rating (Usually a number followed by a star, e.g., "4.0")
                    rating_match = re.search(r'([1-5]\.[0-5])', review.get_text())
                    rating = rating_match.group(1) if rating_match else 'N/A'
                    
                    # 4. Extract Date (Dates formats vary wildly, so we grab text that looks like a date format)
                    date_match = re.search(r'(\d{1,2}\s+[a-zA-Z]{3,10}\.?\s+\d{4}|\w{3}\s\d{1,2},\s\d{4})', review.get_text())
                    date = date_match.group(1) if date_match else 'N/A'
                    
                    # --- NLP Language Detection ---
                    text_for_detection = f"{pros} {cons}"
                    language = "Unknown"
                    
                    if text_for_detection.strip() and text_for_detection != "N/A N/A":
                        try:
                            lang_code = detect(text_for_detection)
                            if lang_code == 'en':
                                language = 'English'
                            elif lang_code == 'es':
                                language = 'Spanish'
                            else:
                                language = f"Other ({lang_code})"
                        except LangDetectException:
                            language = "Detection Failed"
                    
                    writer.writerow({
                        'company': company_name,
                        'review_title': title,
                        'pros': pros,
                        'cons': cons,
                        'rating': rating,
                        'review_date': date,
                        'language': language
                    })
                    
    print(f"\nSuccess! All data has been saved to {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse locally saved Glassdoor HTML files into CSV")
    parser.add_argument("--input_folder", required=True, help="Folder containing saved Glassdoor HTML files")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    args = parser.parse_args()

    parse_local_glassdoor_files(args.input_folder, args.output_csv)