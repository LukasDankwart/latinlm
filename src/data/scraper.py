import os
import requests
from bs4 import BeautifulSoup
import json
from src.data.preprocessing import split_paragraph_to_sentences,clean_paragraph
import src.config as config
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
import time

headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

def scrape_vatican_news_page(url: str) -> list[dict]:
    """ Scrapes one specific news page specified by function argument """

    latin_text = []
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Connection failed: {e}")
        return latin_text

    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    valid_paragraphs = []
    for i, p in enumerate(paragraphs):
        raw_text = p.get_text(strip=True)
        if not raw_text:
            continue
        sentences = split_paragraph_to_sentences(raw_text)
        cleaned_sentences = clean_paragraph(sentences)
        if cleaned_sentences:
            valid_paragraphs.append(" ".join(cleaned_sentences))
        else:
            pass

    for i, p in enumerate(valid_paragraphs):
        wrd_count = len(p.split())
        latin_text.append({
            "url": url,
            "paragraph": i,
            "wrd_cnt": wrd_count,
            "text": p
        })
    return latin_text


def scrape_vatican(list_urls: list, output_dir: str) -> None:
    """ Scrapes given list of urls separately """

    data = []
    print(f"[START] Scraping given set of urls...")
    for idx, url in enumerate(list_urls):
        if idx % 10 == 0:
            print(f"-- {idx} / {len(list_urls)} scraped")
        subpage_data = scrape_vatican_news_page(url)
        if len(subpage_data) != 0:
            data.extend(subpage_data)
        time.sleep(1)
    filename = "vatican_news.jsonl"
    output_path = os.path.join(output_dir, filename)
    print(f"[END] Finished scraping")
    print(f"-- {len(data)} samples have been aggregated \n")

    wrd_sum = 0
    with open(output_path, mode="w", encoding="utf-8") as f:
        for page in data:
            wrd_sum += page["wrd_cnt"]
            json_record = {
                "url": page["url"],
                "paragraph": page["paragraph"],
                "wrd_cnt": page["wrd_cnt"],
                "text": page["text"]
            }
            json_string = json.dumps(json_record, ensure_ascii=False)
            f.write(json_string + "\n")
    print(f"[STORED] Results at {output_path}")
    print(f"-- Overall number of words: {wrd_sum}")


def get_subpages_links(api_url: str) -> list[str]:
    """ Scrapes specified base url of vatican news to extract all relevant subpages urls """

    params = {
        "queryroute": "vaticannews-search-main",
        "q": "hebdomada papae",
        "fq": "lang_s:it",
        "sort": "editorial_date_dt desc,id asc",
        "qId": "8c78f938-3e32-461a-8cf3-a889875adb25",
        "rows": 50,
        "start": 0
    }
    print(f"[START] Scraping references of subpages...")
    all_urls = []
    while True:
        print(f"-- Starting from index {params['start']}...")
        response = requests.get(base_api_url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        docs = data.get("response", {}).get("docs", [])
        # If no more docs can be found, stop
        if not docs:
            break

        for doc in docs:
            raw_id = doc.get("id", "")
            if raw_id:
                clean_path = raw_id.replace("/content/vaticannews", "")
                full_url = f"https://www.vaticannews.va{clean_path}.html"
                all_urls.append(full_url)

        params["start"] += params["rows"]

        time.sleep(1)
    final_urls = [url for url in all_urls if "hebdomadae-papae" in url or "hebdomada-papae" in url]
    print(f"[END] Finished scraping references to subpages")
    print(f"-- {len(final_urls)} references to hebdomadae-papae have been extracted")
    return final_urls


if __name__ == "__main__":

    request_url = "https://www.vaticannews.va/bin/servlet/solr/search?queryroute=vaticannews-search-main&q=hebdomada%20AXNXD%20papae&fq=lang_s:it&sort=editorial_date_dt%20desc,id%20asc&rows=18&qId=8c78f938-3e32-461a-8cf3-a889875adb25"
    base_api_url = "https://www.vaticannews.va/bin/servlet/solr/search"

    console = Console()

    with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
    ) as progress:
        """ 1. Find all URLs to subpages that are related to 'hebdomadae-papae' """
        task1 = progress.add_task(f"[cyan] Scraping subpages of vatican news...", total=None)
        subpages_urls = get_subpages_links(base_api_url)
        progress.update(task1, description=f"[green]✓ {len(subpages_urls)} URLs to subpages have been found!", total=1, completed=1)

        """ 2. Scrape each of the found subpages for latin data """
        task2 = progress.add_task(f"[yellow] Scraping subpages of vatican news...", total=None)
        output_dir = config.RAW_DATA_DIR
        scrape_vatican(subpages_urls, output_dir)
        progress.update(task2, description=f"[green]✓ Data has been stored in {output_dir} ", total=1, completed=1)

