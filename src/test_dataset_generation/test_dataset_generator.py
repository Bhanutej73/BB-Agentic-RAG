import sys, types

# Stub for the ChatVertexAI import that ragas 0.4.x needs but langchain-community 0.4.x no longer ships
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI:  # placeholder, never used
    pass
_stub.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub

import inspect
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator

from src.exception.exception import AgenticException
from src.logger.logger import logging
from src.pinecone_setup.data import LANGCHAIN_DATA_SOURCE, LANGGRAPH_DATA_SOURCE
from src.pinecone_setup.document_extractor.document_extractor import DocumentExtractor

load_dotenv(override=True)

# ---------- settings ----------
BATCH_DIR = Path("src/test_dataset/batches")
DRAFT_PATH = Path("src/test_dataset/golden_v1_draft.csv")
PAGES_PER_BATCH = 3             # pages sent to ragas per batch
MAX_CHARS = 3000                # per page, keeps a batch around 3k tokens
MIN_CHARS = 500                 # skip near-empty pages
QUESTIONS_PER_BATCH = 8
PAUSE_SECONDS = 90              # lets the per-minute token window reset
MAX_BATCHES_PER_SOURCE = 1      # per run; raise it (or set None for all) once one batch works
MAX_CONSECUTIVE_EMPTY = 2       # stop early if this many batches in a row return 0 rows
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "openai/gpt-oss-120b")

BATCH_DIR.mkdir(parents=True, exist_ok=True)

# ---------- models ----------
embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
)
generator_llm = LangchainLLMWrapper(
    ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model=GENERATOR_MODEL)
)
run_config = RunConfig(max_workers=1, max_retries=15, max_wait=90, timeout=240)


# ---------- helpers ----------
def remove_empty_batches():
    """Delete batch CSVs with no data rows so those batches rerun instead of being skipped."""
    for p in sorted(BATCH_DIR.glob("*.csv")):
        try:
            rows = len(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            rows = 0
        if rows == 0:
            logging.warning(f"Removing empty batch file {p}")
            p.unlink()


def generate_batch(generator, batch):
    """Call ragas; ask it to raise on failures if this version supports it."""
    fn = generator.generate_with_langchain_docs
    kwargs = {"testset_size": QUESTIONS_PER_BATCH, "run_config": run_config}
    if "raise_exceptions" in inspect.signature(fn).parameters:
        kwargs["raise_exceptions"] = True
    return fn(batch, **kwargs)


def run_source(name, docs, generator):
    """Generate in small batches. Returns True if every planned batch is done."""
    batches = [docs[i:i + PAGES_PER_BATCH] for i in range(0, len(docs), PAGES_PER_BATCH)]
    if MAX_BATCHES_PER_SOURCE:
        batches = batches[:MAX_BATCHES_PER_SOURCE]
    logging.info(f"{name}: {len(docs)} usable pages -> {len(batches)} batches planned")

    consecutive_empty = 0
    for n, batch in enumerate(batches):
        path = BATCH_DIR / f"{name}_batch{n:02d}.csv"
        if path.exists():
            continue  # finished in an earlier run
        est_tokens = sum(len(d.page_content) for d in batch) // 4
        logging.info(f"{name} batch {n}: {len(batch)} pages, ~{est_tokens} tokens per pass")
        try:
            dataset = generate_batch(generator, batch)
        except Exception as e:
            # Most likely a rate limit or a parsing failure. Stop here; rerun later to resume.
            logging.error(f"Stopped at {name} batch {n}: {e}")
            return False

        df = dataset.to_pandas()
        if df.empty:
            consecutive_empty += 1
            logging.warning(
                f"{name} batch {n} returned 0 rows (likely LLM failures); not saved, will retry next run"
            )
            if consecutive_empty >= MAX_CONSECUTIVE_EMPTY:
                logging.error("Too many empty batches in a row; stopping to protect quota.")
                return False
            continue

        consecutive_empty = 0
        df["source"] = name
        df["batch"] = n
        df.to_csv(path, index=False)
        logging.info(f"Saved {path} ({len(df)} rows)")
        time.sleep(PAUSE_SECONDS)
    return True


def merge_batches():
    files = sorted(BATCH_DIR.glob("*.csv"))
    frames = []
    for p in files:
        try:
            frame = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        logging.info("No non-empty batch files to merge.")
        return
    df = pd.concat(frames, ignore_index=True)
    if "user_input" in df.columns:
        df = df.drop_duplicates(subset="user_input").reset_index(drop=True)
    df.insert(0, "id", [f"q{i:03d}" for i in range(len(df))])
    df["type"] = ""   # lookup / multi-hop / unanswerable
    df["keep"] = ""   # y / n after manual review
    df.to_csv(DRAFT_PATH, index=False)
    logging.info(f"Merged {len(frames)} batch files into {DRAFT_PATH} ({len(df)} rows)")


# ---------- main ----------
if __name__ == "__main__":
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "merge":
            merge_batches()
            sys.exit(0)

        remove_empty_batches()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        doc_extractor = DocumentExtractor(LANGCHAIN_DATA_SOURCE, LANGGRAPH_DATA_SOURCE, text_splitter)
        generator = TestsetGenerator(llm=generator_llm, embedding_model=embeddings)

        sources = {
            "langchain": doc_extractor.extract_langchain_data_without_splitting,
           # "langgraph": doc_extractor.extract_langgraph_data_without_splitting,
        }
        for name, loader in sources.items():
            docs = loader(max_chars=MAX_CHARS, min_chars=MIN_CHARS)
            if not run_source(name, docs, generator):
                logging.info("Stopped early. Rerun later to resume from the next batch.")
                break
        else:
            logging.info("Planned batches finished. Run with the 'merge' argument to build the draft CSV.")
    except SystemExit:
        raise
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        raise AgenticException(e, sys)