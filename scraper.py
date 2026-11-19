
from collections import Counter
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import requests
from tokens import get_truncated_text
from config import MAX_PAGE_TOKENS, RAG_ENABLED, REPEATED_LINE_MIN_SHARE

# Tags that never hold brochure content, plus the navigation menu and the footer.
# `header` and elements marked hidden/aria-hidden are deliberately kept: on real sites they
# hold the page heading, customer stories and testimonials (carousels, collapsed panels).
IRRELEVANT_TAGS = ["script", "style", "img", "input", "nav", "footer"]
IRRELEVANT_LANDMARKS = '[role="navigation"], [role="contentinfo"]'


def is_web_link(href):
    """
    Tells whether a link points to a web page: not an in-page anchor (#...), mailto:, tel:, javascript: and so on.
    Relative links such as "/about" count as web links.
    """
    href = href.strip()
    if not href or href.startswith("#"):
        return False
    return urlparse(href).scheme in ("", "http", "https")


def is_decorative_line(line):
    """
    Tells whether a line of page text is decoration and not content: it has no letters or digits at all
    (a lone "•" or "→"), or it is a long run of mostly symbols (text art such as "✳ ⟡ · ◎ ✦").
    Prices like "$2.50" and short labels stay, because they contain enough letters or digits.
    """
    line = line.strip()
    if not line:
        return True
    alphanumeric = sum(char.isalnum() for char in line)
    return alphanumeric == 0 or (len(line) >= 8 and alphanumeric / len(line) < 0.2)


def remove_decorative_lines(text):
    """Removes the decorative lines (see is_decorative_line) from the text of a page."""
    return "\n".join(line for line in text.split("\n") if not is_decorative_line(line))


def remove_repeated_lines(pages, min_share=REPEATED_LINE_MIN_SHARE, min_pages=4, min_count=3):
    """
    Removes the lines that repeat across the pages of one site, such as the menu that
    <nav> removal misses when a site builds its menu from plain <div> elements.

    Args:
        pages (list): Page dicts with a "content" key.
        min_share (float): A line is removed when it appears on more than this share of the pages.
        min_pages (int): With fewer pages than this, nothing is removed, because the share means little.
        min_count (int): A line must also appear on at least this many pages, so a line shared by two
            pages (for example a card that is on the landing page and on a product page) is kept.

    Returns:
        list: New page dicts with the repeated lines removed. The input is not changed.
    """
    if len(pages) < min_pages:
        return pages

    pages_with_line = Counter()
    for page in pages:
        pages_with_line.update({line.strip() for line in page["content"].split("\n") if line.strip()})

    repeated = {line for line, count in pages_with_line.items() if count >= min_count and count > min_share * len(pages)}
    return [
        {**page, "content": "\n".join(line for line in page["content"].split("\n") if line.strip() not in repeated)}
        for page in pages
    ]


def fetch_website_links(url):
    """
    Fetches all the web links from the given website URL.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        list: The links found on the website, without in-page anchors, mailto:, tel: and similar links.
    """

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Parse the raw bytes: response.text assumes ISO-8859-1 when the server sends no charset, which garbles UTF-8 text
        soup = BeautifulSoup(response.content, 'html.parser')
        links = [link['href'] for link in soup.find_all('a', href=True) if is_web_link(link['href'])]
        return links
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []


def fetch_website_content(url):
    """
    Fetches the content of the given website URL.

    Args:
        url (str): The URL of the website to scrape.

    Returns:
        str: The content of the website, truncated to MAX_PAGE_TOKENS tokens.
    """

    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')  # raw bytes, so the encoding is detected correctly
        title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevant in soup.body(IRRELEVANT_TAGS) + soup.body.select(IRRELEVANT_LANDMARKS):
                irrelevant.decompose()
            text = remove_decorative_lines(soup.body.get_text(separator="\n", strip=True))
        else:
            text = ""

        if RAG_ENABLED:
            return title + "\n\n" + text 
        return get_truncated_text(title + "\n\n" + text, MAX_PAGE_TOKENS)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""


