import os
import requests
import time
from bs4 import BeautifulSoup
import urllib.parse
import re
from config import LATIN_LIBRARY_AUTHORS
import json

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def scrape_latin_library():
    base_url = "https://www.thelatinlibrary.com/"

    # Call the mainpage
    #response = requests.get(base_url)
    #soup = BeautifulSoup(response.text, 'html.parser')

    for author in LATIN_LIBRARY_AUTHORS :
        print(f"[START] Scraping author: {author} ...")
        author_subpages = scrape_author_subpages(base_url, author)
        if author_subpages is None:
            raise RuntimeError(f"[ERROR] No subpages for {author} found!")
        author_subpages_results = []
        for subpage_url in author_subpages:
            subpage_results = scrape_subpage(subpage_url)
            time.sleep(10) # Waiting 10 secs after each scraped side to spare the server
            if subpage_results:
                author_subpages_results.append(subpage_results)

        print(f"[DONE] Scraping author: {author} ...")
        if len(author_subpages_results) > 0:
            collection = {
                "author": author,
                "results": author_subpages_results
            }
            print(f"[START] Storing authors results...")
            base_dir = "latinlibrary/raw/"
            store_results(collection, base_dir)
            if author.endswith(".html"):
                author = author.removesuffix(".html")
            filename = f"{base_dir}{author}.csv"
            print(f"[DONE] Saved results for {author} to {filename}. \n")
        time.sleep(60)    # Waiting 60 secs after each other to spare the server


def scrape_author_subpages(base_url, author):
    # Define url to author
    if author.endswith(".html"):
        author_url = base_url + author
    else:
        author_url = base_url + author + "/"
    print(f"Author url: {author_url}")
    # Request authors page and fetch all relevant subpages
    page_resp = requests.get(author_url, headers=headers)
    if page_resp.status_code != 200:
        print(f"Access to {author} denied")
        return None
    page_soup = BeautifulSoup(page_resp.text, 'html.parser')
    shtml_links = []
    for div_tag in page_soup.find_all('div', class_="work"):
        for a_tag in div_tag.find_all('a', href=True):
            href = a_tag['href']
            if href.endswith(('.shtml', ".html")):
                full_url = urllib.parse.urljoin(author_url, href)
                shtml_links.append(full_url)
    shtml_links = list(set(shtml_links))
    print(f"Found {len(shtml_links)} subpages for {author}.")
    return shtml_links


def scrape_subpage(subpage_url):
    print(f"Scraping subpage: {subpage_url}")
    page_resp = requests.get(subpage_url, headers=headers)
    if page_resp.status_code != 200:
        print(f"Access to {subpage_url} denied")
    page_resp.encoding = 'ISO-8859-1'
    page_soup = BeautifulSoup(page_resp.text, 'html.parser')
    paragraphs = page_soup.find_all('p')
    clean_texts = []
    ignore_classes = ['pagehead', 'border', 'shortborder', 'internal_navigation']
    for p in paragraphs:
        p_classes = p.get('class', [])
        if any(cls in ignore_classes for cls in p_classes):
            continue
        # Delete <font> tags
        for font_tag in p.find_all('font'):
            font_tag.unwrap()
        # Delete all <a> tags
        for a_tag in p.find_all('a'):
            link_text = a_tag.get_text(strip=True)
            if re.match(r'^\[?\d+\]?$', link_text):
                a_tag.decompose()
            else:
                a_tag.unwrap()
        # Delete all <b> tags
        for tag in p.find_all(['b', 'i']):
            tag.unwrap()
        text = p.get_text(separator=" ", strip=True)
        if "...." in text:
            continue

        text = clean_text(text)
        if text and not word_sign_ratio(text):
            words = text.split()
            words = [
                w for w in words
                if not is_roman_numeral(w) and re.search(r'[a-zA-Z]', w)
            ]
            word_count = len(words)
            if word_count >= 4:
                clean_texts.append({
                    "text": text,
                    "word_count": word_count
                })
    if len(clean_texts) == 0:
        return None

    return {
        "author_subpage": subpage_url,
        "subpage_results": clean_texts
    }


def clean_text(text):
    text = re.sub(r'^\d+\.?\s*', '', text)
    text = re.sub(r'^(?i)(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\.?\s+', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'^[\W_]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '...', text)
    text = text.replace('...', ' ... ')
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    return text


def word_sign_ratio(text, threshold=0.1):
    dot_count = text.count('.')
    if len(text) == 0: return True
    return (dot_count / len(text)) > threshold

def store_results(collection, base_dir="latinlibrary/raw/"):
    os.makedirs(base_dir, exist_ok=True)
    author = collection["author"]
    author_results = collection["results"]
    filename = f"{base_dir}{author}.jsonl"
    with open(filename, mode="w", encoding="utf-8") as f:
        for page_data in author_results:
            subpage_url = page_data["author_subpage"]
            for result in page_data["subpage_results"]:
                json_record = {
                    "text": result["text"],
                    "word_count": result["word_count"],
                    "meta": {
                        "author": author,
                        "source_url": subpage_url
                    }
                }
                json_string = json.dumps(json_record, ensure_ascii=False)
                f.write(json_string + "\n")


def is_roman_numeral(word):
    clean_word = re.sub(r'[\W_]+', '', word)
    if not clean_word:
        return False
    pattern = r'^(?i)(?=[MDCLXVI])M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return bool(re.match(pattern, clean_word))


if __name__ == "__main__":
    scrape_latin_library()
