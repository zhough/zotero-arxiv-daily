"""One-off replay of the 2026-09-23 cs.CV/cs.RO announcements."""

import json
from pathlib import Path
from time import sleep
import xml.etree.ElementTree as ET

import arxiv
from loguru import logger
import requests

from zotero_arxiv_daily.retriever.arxiv_retriever import ArxivRetriever


DATE = "2026-09-23"
BASE = "https://oaipmh.arxiv.org/oai"
IDS = json.loads(Path(__file__).with_name("sep23_ids.json").read_text())
assert len(IDS) == len(set(IDS)) == 212


def child(node, name):
    return node.find(f"{{*}}{name}")


def value(node, name):
    element = child(node, name)
    return " ".join(element.itertext()).strip() if element is not None else ""


def load_records():
    """Collect bulk OAI metadata; retrieve records individually if updated since DATE."""
    session = requests.Session()
    session.headers.update({"User-Agent": "zotero-arxiv-daily/1.0 (historical replay)"})
    records = {}
    params = {"verb": "ListRecords", "from": DATE, "until": DATE, "metadataPrefix": "arXiv"}
    ids = set(IDS)

    for page in range(20):
        response = session.get(BASE, params=params, timeout=(15, 90))
        response.raise_for_status()
        root = ET.fromstring(response.content)
        error = root.find("{*}error")
        if error is not None:
            raise RuntimeError(f"OAI ListRecords error: {error.attrib.get('code')}: {error.text}")
        for record in root.findall(".//{*}record"):
            metadata = record.find("{*}metadata/{*}arXiv")
            if metadata is not None and value(metadata, "id") in ids:
                records[value(metadata, "id")] = metadata
        token = root.find(".//{*}resumptionToken")
        if token is None or not token.text or not token.text.strip():
            break
        params = {"verb": "ListRecords", "resumptionToken": token.text.strip()}
        logger.info(f"OAI page {page + 1}: matched {len(records)} of {len(IDS)}")
        sleep(3)
    else:
        raise RuntimeError("OAI pagination exceeded 20 pages")

    logger.info(f"OAI bulk metadata matched {len(records)} of {len(IDS)} original announcements")
    for index, paper_id in enumerate(IDS):
        if paper_id in records:
            continue
        sleep(3)
        response = session.get(
            BASE,
            params={"verb": "GetRecord", "identifier": f"oai:arXiv.org:{paper_id}", "metadataPrefix": "arXiv"},
            timeout=(15, 90),
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        metadata = root.find(".//{*}metadata/{*}arXiv")
        if metadata is None:
            raise RuntimeError(f"Missing OAI metadata for {paper_id}")
        records[paper_id] = metadata
        if index % 25 == 0:
            logger.info(f"OAI metadata: {len(records)} of {len(IDS)}")

    papers = []
    for paper_id in IDS:
        metadata = records[paper_id]
        author_nodes = metadata.findall("{*}authors/{*}author")
        authors = [
            arxiv.Result.Author(" ".join(filter(None, [value(a, "forenames"), value(a, "keyname")])))
            for a in author_nodes
        ]
        title = value(metadata, "title")
        abstract = value(metadata, "abstract")
        if not title or not abstract or not authors:
            raise RuntimeError(f"Incomplete OAI metadata for {paper_id}")
        papers.append(arxiv.Result(
            entry_id=f"https://arxiv.org/abs/{paper_id}",
            title=title,
            authors=authors,
            summary=abstract,
            links=[arxiv.Result.Link(href=f"https://arxiv.org/pdf/{paper_id}", title="pdf")],
        ))
    logger.info(f"Replaying {len(papers)} papers announced on {DATE}")
    return papers


ArxivRetriever._retrieve_raw_papers = lambda self: load_records()

if __name__ == "__main__":
    import runpy

    runpy.run_path("src/zotero_arxiv_daily/main.py", run_name="__main__")
