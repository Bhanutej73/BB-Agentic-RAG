import re
import requests
import logging
import sys
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    Language
)
from src.exception.exception import AgenticException
from src.logger.logger import logging
from src.pinecone_setup.data import LANGCHAIN_DATA_SOURCE

class DocumentExtractor:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (docs-ingestion)"})

    def _load_pages(self, urls, timeout=30, retries=2):
        docs = []
        try:
            for url in urls:
                md_url = url if url.endswith((".md", ".mdx")) else url.rstrip("/") + ".md"
            
                text = None
                for attempt in range(retries + 1):
                    try:
                        r = self.session.get(md_url, timeout=timeout)
                        r.raise_for_status()
                        text = r.text
                        break
                    except requests.RequestException as e:
                        if attempt == retries:
                            logging.warning(f"Skipping {md_url}: {e}")
            
                if not text:
                     continue
            
                        # Guard: some sites return an HTML page when no .md version exists
                if text.lstrip().lower().startswith(("<!doctype", "<html")):
                    logging.warning(f"Got HTML instead of markdown, skipping {md_url}")
                    continue
            
                docs.append(Document(page_content=text, metadata={"source": url}))
            
            logging.info(f"Loaded {len(docs)}/{len(urls)} pages")
            return docs
        except Exception as e:
            logging.error("Erro occurred while leading the pages")
            raise AgenticException(e,sys)
        
    
    @staticmethod
    def _clean(md: str) -> str:
        try:
            md = re.sub(r"\A---.*?---\s*", "", md, flags=re.S)            # frontmatter
            md = re.sub(r"^(import|export)\s.+$", "", md, flags=re.M)      # MDX imports/exports
            md = re.sub(
                    r"</?(Note|Tip|Warning|Info|Check|Tabs|Tab|Card|CardGroup|Steps|Step|Accordion|AccordionGroup)[^>]*>",
                    "", md,
                )
            return re.sub(r"\n\s*\n+", "\n\n", md).strip()
        except Exception as e:
            logging.error("An error occurred while cleaning the documents")
            raise AgenticException(e,sys)

    # ---------- 1. ingestion: full coverage, chunked ----------
    def extract_for_ingestion(self, urls, min_chars=500, chunk_size=1500, chunk_overlap=150):
        try:
            header_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
                        strip_headers=False,
                    )
            size_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.MARKDOWN,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
            
            chunks = []
            for d in self._load_pages(urls):
                text = self._clean(d.page_content)
                if len(text) < min_chars:          # skip stub/redirect/index pages
                    continue
            
                sections = header_splitter.split_text(text)
                for c in size_splitter.split_documents(sections):
                    path = " > ".join(c.metadata[h] for h in ("h1", "h2", "h3") if h in c.metadata)
                    c.metadata.update({**d.metadata, "section": path})
                    if path:                       # heading path helps embeddings a lot
                        c.page_content = f"{path}\n\n{c.page_content}"
                    chunks.append(c)
            
            total = sum(len(c.page_content) for c in chunks)
            logging.info(f"{len(chunks)} chunks | {total} chars | ~{total // 4} tokens to embed")
            return chunks
        except Exception as e:
            logging.error("An error occurred in the ingestion pipeline while extracting documents")
            raise AgenticException(e,sys)
        

    # ---------- 2. golden dataset: capped whole pages ----------
    def extract_for_golden_dataset(self, urls, max_docs=50, max_chars=3000, min_chars=800):
        docs, seen = [], set()
        try:
            for d in self._load_pages(urls):
                text = self._clean(d.page_content)
                if len(text) < min_chars or text in seen:
                    continue
                seen.add(text)

                if len(text) > max_chars:          # cut at a paragraph boundary, not mid-sentence
                    cut = text.rfind("\n\n", 0, max_chars)
                    text = text[: cut if cut > min_chars else max_chars]

                docs.append(Document(page_content=text, metadata=d.metadata))
                if len(docs) >= max_docs:
                    break

            total = sum(len(d.page_content) for d in docs)
            logging.info(f"{len(docs)} docs | {total} chars | ~{total // 4} tokens per pass")
            return docs
        except Exception as e:
            raise AgenticException(e,sys)

if __name__=="__main__":
    extractor=DocumentExtractor()
    documents=extractor.extract_for_ingestion(urls=LANGCHAIN_DATA_SOURCE)
    print(documents)