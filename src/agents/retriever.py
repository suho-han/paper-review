"""Retriever agent that queries ChromaDB for similar reviews."""

import os
from typing import Any, Dict, List, Tuple

import chromadb

from .state import AgentState


class ReviewRetriever:
    def __init__(self, db_path: str = "./chromadb", collection_names: List[str] | None = None):
        self.client = chromadb.PersistentClient(path=db_path)
        if collection_names:
            self.collections = [self.client.get_collection(name=name) for name in collection_names]
        else:
            self.collections = self.client.list_collections()
            names = [collection.name for collection in self.collections]
            print(f"ReviewRetriever initialized with {len(names)} collections: {names}")

    def retrieve_similar_reviews(
        self, query_text: str, k: int = 3
    ) -> Tuple[List[str], List[Dict[str, Any]], List[str], List[str], List[str], List[str]]:
        all_results: List[Dict[str, Any]] = []
        for collection in self.collections:
            try:
                results = collection.query(
                    query_texts=[query_text],
                    n_results=k,
                    where={"type": "review"},
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"Error querying collection {collection.name}: {exc}")
                continue

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0] or [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                meta_with_extras = dict(meta or {})
                meta_with_extras["collection_name"] = getattr(collection, "name", "")
                meta_with_extras["distance"] = dist
                all_results.append({
                    "document": doc,
                    "metadata": meta_with_extras,
                    "distance": dist,
                    "collection": collection,
                })

        all_results.sort(key=lambda item: item["distance"])
        top_results = all_results[:k]

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        abstracts: List[str] = []
        paper_urls: List[str] = []
        pdf_urls: List[str] = []
        pdf_paths: List[str] = []

        for result in top_results:
            documents.append(result["document"])
            metadatas.append(result["metadata"])
            collection = result["collection"]
            metadata = result["metadata"]
            forum_id = metadata.get("forum_id")

            if forum_id:
                abs_id = f"abs_{forum_id}"
                abs_result = collection.get(ids=[abs_id])
                abstracts.append(abs_result.get("documents", [""])[0] or "")
                paper_urls.append(f"https://openreview.net/forum?id={forum_id}")
                pdf_urls.append(f"https://openreview.net/pdf?id={forum_id}")
                pdf_paths.append(self._download_pdf(pdf_urls[-1], forum_id))
            else:
                abstracts.append("")
                paper_urls.append("")
                pdf_urls.append("")
                pdf_paths.append("")

        return documents, metadatas, abstracts, paper_urls, pdf_urls, pdf_paths

    def _download_pdf(self, pdf_url: str, forum_id: str) -> str:
        output_dir = "outputs/pdfs"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"{forum_id}.pdf")

        if os.path.exists(file_path):
            return file_path

        try:
            import requests

            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                with open(file_path, "wb") as file:
                    file.write(response.content)
                return file_path
            print(f"Failed to download PDF. Status code: {response.status_code}")
        except Exception as exc:  # pragma: no cover - network issues
            print(f"Failed to download PDF {pdf_url}: {exc}")

        return ""


def retrieve_node(state: AgentState) -> AgentState:
    print("--- RETRIEVE SIMILAR REVIEWS ---")
    paper_text = state.get("paper_text") or ""
    retriever = ReviewRetriever()
    reviews, metadatas, abstracts, paper_urls, pdf_urls, pdf_paths = retriever.retrieve_similar_reviews(paper_text, k=3)

    progress = list(state.get("progress_log", []))
    progress.append("RAG Agent: retrieved similar human reviews.")

    traces = dict(state.get("agent_traces") or {})
    traces.setdefault("retriever", {})
    traces["retriever"].update({
        "query_preview": paper_text[:600],
        "top_results": [
            {
                "forum_id": (meta or {}).get("forum_id"),
                "collection": (meta or {}).get("collection_name"),
                "distance": (meta or {}).get("distance"),
                "rating": (meta or {}).get("rating"),
            }
            for meta in (metadatas or [])
        ],
    })

    return AgentState(
        retrieved_reviews=reviews,
        retrieved_metadatas=metadatas,
        retrieved_abstracts=abstracts,
        retrieved_paper_urls=paper_urls,
        retrieved_pdf_urls=pdf_urls,
        retrieved_pdf_paths=pdf_paths,
        agent_traces=traces,
        progress_log=progress,
    )
