import os
import requests
import time
from bs4 import BeautifulSoup
import urllib.parse
import re
import pandas as pd
from config import LATIN_LIBRARY_AUTHORS

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
            base_dir = "latinlibrary/"
            store_results(collection, base_dir)
            filename = f"{base_dir}{author}.csv"
            print(f"[DONE] Saved results for {author} to {filename}. \n")
        time.sleep(60)    # Waiting 60 secs after each other to spare the server


def scrape_author_subpages(base_url, author):
    # Define url to author
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
    for p in paragraphs:
        # Delete <font> tags
        for font_tag in p.find_all('font'):
            font_tag.decompose()
        # Delete all <a> tags
        for a_tag in p.find_all('a'):
            a_tag.decompose()
        # Delete all <b> tags
        for b_tag in p.find_all('b'):
            if b_tag.find('a', attrs={'name': True}):
                b_tag.decompose()
        text = p.get_text(strip=True)
        text = clean_text(text)
        if text:
            word_count = len(text.split())
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
    return text


def store_results(collection, base_dir="latinlibrary/"):
    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)
    author = collection["author"]
    author_results = collection["results"]
    df = pd.json_normalize(
        author_results,
        record_path=['subpage_results'],
        meta=['author_subpage']
    )
    df = df.rename(columns={
        'author_subpage': 'subpage',
        'word-cnt': 'word_count'
    })
    df = df[["subpage", "word_count", "text"]]
    filename = f"{base_dir}{author}.csv"
    df.to_csv(filename, index=False, encoding="utf-8")


if __name__ == "__main__":
    scrape_latin_library()
