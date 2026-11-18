import json
import re
from pathlib import Path
from urllib.parse import urlparse

CACHE_DIR = Path("cache")


def cache_path(website_url):
    """Builds the cache file path for a website, e.g. cache/huggingface.com.json."""
    parsed = urlparse(website_url if "://" in website_url else f"https://{website_url}")
    name = re.sub(r"[^A-Za-z0-9.-]", "_", parsed.netloc.removeprefix("www."))
    return CACHE_DIR / f"{name}.json"


def get_pages(website_url, fetch_pages, refresh=False):
    """Returns the cached pages if present, otherwise fetches them and saves them."""
    path = cache_path(website_url)

    if path.exists() and not refresh:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    pages = fetch_pages(website_url)
    CACHE_DIR.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    return pages