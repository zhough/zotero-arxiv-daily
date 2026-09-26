"""Tests for ArxivRetriever."""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever, _run_with_hard_timeout
import zotero_arxiv_daily.retriever.arxiv_retriever as arxiv_retriever


def _sleep_and_return(value: str, delay_seconds: float) -> str:
    time.sleep(delay_seconds)
    return value


def _raise_runtime_error() -> None:
    raise RuntimeError("boom")


def test_arxiv_retriever(config, mock_feedparser, monkeypatch):
    monkeypatch.setattr("zotero_arxiv_daily.retriever.base.sleep", lambda _: None)
    rss_xml = Path("tests/retriever/arxiv_rss_example.xml").read_bytes()
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        return SimpleNamespace(content=rss_xml, raise_for_status=lambda: None)

    monkeypatch.setattr(arxiv_retriever.requests, "get", fake_get)
    new_entries = [
        e for e in mock_feedparser.entries
        if e.get("arxiv_announce_type", "new") == "new"
    ]

    # Skip file downloads in convert_to_paper
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_html", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_pdf", lambda paper: None)
    monkeypatch.setattr(arxiv_retriever, "extract_text_from_tar", lambda paper: None)

    retriever = ArxivRetriever(config)
    papers = retriever.retrieve_papers()

    assert len(papers) == len(new_entries)
    assert set(p.title for p in papers) == set(e.title for e in new_entries)
    assert papers[0].authors == ["Alice Smith", "Bob Jones"]
    assert papers[0].abstract == "We propose a neural architecture search method for efficient transformers."
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2508.14001v1"
    assert calls == ["https://rss.arxiv.org/atom/cs.AI+cs.CV"]
    result = arxiv_retriever._entry_to_arxiv_result(new_entries[0])
    assert isinstance(result, arxiv_retriever.arxiv.Result)
    assert result.source_url() == "https://arxiv.org/src/2508.14001v1"


def test_arxiv_retriever_empty_feed(config, monkeypatch):
    empty_feed = b'<feed xmlns="http://www.w3.org/2005/Atom"><title>cs.AI updates</title></feed>'
    monkeypatch.setattr(
        arxiv_retriever.requests,
        "get",
        lambda url, **kwargs: SimpleNamespace(content=empty_feed, raise_for_status=lambda: None),
    )
    assert ArxivRetriever(config)._retrieve_raw_papers() == []


def test_arxiv_retriever_rejects_failed_rss_request(config, monkeypatch):
    def failed_get(url, **kwargs):
        raise requests.HTTPError("RSS unavailable")

    monkeypatch.setattr(arxiv_retriever.requests, "get", failed_get)
    with pytest.raises(requests.HTTPError, match="RSS unavailable"):
        ArxivRetriever(config)._retrieve_raw_papers()


def test_run_with_hard_timeout_returns_value():
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 0.01), timeout=1, operation="test op", paper_title="paper"
    )
    assert result == "done"


def test_run_with_hard_timeout_returns_none_on_timeout(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _sleep_and_return, ("done", 1.0), timeout=0.01, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "timed out" in warnings[0]


def test_run_with_hard_timeout_returns_none_on_failure(monkeypatch):
    warnings: list[str] = []
    monkeypatch.setattr(arxiv_retriever, "logger", SimpleNamespace(warning=warnings.append))
    result = _run_with_hard_timeout(
        _raise_runtime_error, (), timeout=1, operation="test op", paper_title="paper"
    )
    assert result is None
    assert "boom" in warnings[0]
