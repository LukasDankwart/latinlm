import src.config as config
import src.data.scraper as scraper

""" 
    This scripts orchestrates which websites are scraped.
"""

if __name__ == "__main__":
    output_dir = config.RAW_DATA_DIR

    print(f"[START] Start scrapers for all implemented websites...")

    """ 1. Find all URLs to subpages that are related to 'hebdomadae-papae' """
    vatican_subpages = scraper.get_subpages_links(config.VATICAN_BASE_API_URL)
    """ 2. Scrape each of the found subpages of vatican news for latin data """
    scraper.scrape_vatican(vatican_subpages, output_dir)


    """ 3. Find all URLS of blog post from 'Nuntii Latini' """
    nuntii_subpages = scraper.scrape_nuntii_latini_subpages(config.NUNTII_BASE_URL)
    """ 4. Scrape all subpages of 'Nuntii Latini' """
    scraper.scrape_nuntii_latini(nuntii_subpages, output_dir)

    print(f"[END] Done scraping websites!")

