import re
from typing import List
import requests
from bs4 import BeautifulSoup

YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|embed/|shorts/|live/)|youtu\.be/)([A-Za-z0-9_-]{11})"
)

def is_youtube_like(url: str) -> bool:
    return "youtube.com" in (url or "").lower() or "youtu.be" in (url or "").lower()

def extract_youtube_urls_from_educatieonline(url: str) -> List[str]:
    """
    Extrage toate ID-urile unice de YouTube de pe o pagina educatieonline.md
    folosind strategia solicitata (<a> tags, <iframe> si text brut).
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        html = response.text
    except Exception as e:
        raise RuntimeError(f"Nu s-a putut accesa educatieonline.md: {e}")

    soup = BeautifulSoup(html, "html.parser")
    urls: List[str] = []
    seen = set()

    def add_vid(vid: str):
        if not vid or vid in seen:
            return
        seen.add(vid)
        urls.append(f"https://www.youtube.com/watch?v={vid}")

    # 1. <a> tags
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if is_youtube_like(href):
            match = YOUTUBE_ID_RE.search(href)
            if match:
                add_vid(match.group(1))

    # 2. <iframe> tags
    for iframe in soup.find_all("iframe", src=True):
        src = iframe.get("src", "").strip()
        if is_youtube_like(src):
            match = YOUTUBE_ID_RE.search(src)
            if match:
                add_vid(match.group(1))

    # 3. Raw Regex on the entire HTML source
    for match in YOUTUBE_ID_RE.finditer(html):
        add_vid(match.group(1))

    return urls
