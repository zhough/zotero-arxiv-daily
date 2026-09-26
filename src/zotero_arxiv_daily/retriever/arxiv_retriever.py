from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


def _entry_to_arxiv_result(entry: Any) -> ArxivResult:
    """Convert a parsed arXiv Atom entry back into the library's native Result object."""
    title = getattr(entry, "title", None) or ""
    summary = getattr(entry, "summary", None) or ""
    authors = getattr(entry, "authors", []) or []
    entry_id = getattr(entry, "id", None) or ""
    pdf_url = None
    for link in getattr(entry, "links", []) or []:
        href = getattr(link, "href", None)
        if href and "/pdf/" in href:
            pdf_url = href
            break

    # Prefer the library's own conversion method if it exists.
    if hasattr(arxiv.Result, "from_entry"):
        return arxiv.Result.from_entry(entry)

    # Fallback for older / alternative library shapes.
    result = type("FallbackResult", (), {})()
    result.title = title
    result.summary = summary
    result.authors = authors
    result.entry_id = entry_id
    result.pdf_url = pdf_url
    result.source_url = lambda: entry_id.replace("/abs/", "/src/") if "/abs/" in entry_id else None
    return result


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Keep requests small and slow enough to avoid arXiv API throttling.
        query = "+".join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if "Feed error for query" in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")

        raw_papers: list[ArxivResult] = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = [
            i.id.removeprefix("oai:arXiv.org:")
            for i in feed.entries
            if i.get("arxiv_announce_type", "new") in allowed_announce_types
        ]
        all_paper_ids = [paper_id.strip() for paper_id in all_paper_ids if paper_id and paper_id.strip()]
        if not all_paper_ids:
            logger.warning("No valid arXiv paper IDs found; skipping the API request.")
            return raw_papers
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 10
        batch_retry_delay = 30
        batch_size = 10
        retryable_statuses = {403, 406, 429, 500, 502, 503, 504}

        try:
            for i in range(0, len(all_paper_ids), batch_size):
                batch_ids = all_paper_ids[i : i + batch_size]
                batch_number = i // batch_size

                for attempt in range(max_batch_retries):
                    try:
                        response = requests.get(
                            "https://export.arxiv.org/api/query",
                            params={"id_list": ",".join(batch_ids)},
                            headers={"User-Agent": "zotero-arxiv-daily/1.0"},
                            timeout=(10, 60),
                        )
                        response.raise_for_status()
                        parsed = feedparser.parse(response.content)
                        batch = [_entry_to_arxiv_result(entry) for entry in parsed.entries]
                        logger.info(
                            f"Batch {batch_number}: fetched {len(parsed.entries)} entries, converted {len(batch)} results"
                        )
                        bar.update(len(batch))
                        raw_papers.extend(batch)
                        break
                    except requests.RequestException as exc:
                        status = getattr(exc.response, "status_code", None)
                        if status not in retryable_statuses:
                            raise
                        if attempt == max_batch_retries - 1:
                            logger.error(
                                f"arXiv API failed for batch {batch_number} after "
                                f"{max_batch_retries} attempts: HTTP {status}"
                            )
                            raise

                        wait = min(300, batch_retry_delay * (2**attempt))
                        logger.warning(
                            f"arXiv API returned HTTP {status} on batch {batch_number}; "
                            f"retry {attempt + 1}/{max_batch_retries} in {wait}s"
                        )
                        sleep(wait)
                    except Exception as exc:
                        logger.warning(f"Skipping arXiv batch {batch_number}: {type(exc).__name__}: {exc}")
                        break

                if i + batch_size < len(all_paper_ids):
                    sleep(10)
        finally:
            bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
