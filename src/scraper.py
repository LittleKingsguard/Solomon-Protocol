import httpx
import trafilatura
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

async def fetch_page(url: str) -> Optional[str]:
    """Fetches the HTML of a page."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return None

import re

def extract_main_content(html: str) -> tuple[Optional[str], List[str]]:
    """Uses trafilatura to extract the main article/content text from HTML, separating images."""
    try:
        # include_links=True, include_images=True because our embedder is multimodal
        result = trafilatura.extract(html, include_links=True, include_images=True)
        if not result:
            return None, []
            
        # Extract markdown images
        image_pattern = r'!\[.*?\]\(.*?\)'
        images = re.findall(image_pattern, result)
        
        # Remove images from main text
        clean_text = re.sub(image_pattern, '', result)
        return clean_text.strip(), images
    except Exception as e:
        logger.error(f"Failed to extract content: {e}")
        return None, []

async def scrape_pages(urls: List[str]) -> dict:
    """Scrapes multiple pages and returns a dictionary of URL -> (Extracted Text, Extracted Images)."""
    results = {}
    for url in urls:
        html = await fetch_page(url)
        if html:
            text, images = extract_main_content(html)
            if text or images:
                results[url] = (text, images)
            else:
                logger.warning(f"Could not extract content from {url}")
    return results
