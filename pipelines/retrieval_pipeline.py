'''
“질의 리스트(여러 개) → ScienceON 검색 → 결과 합치기 → 중복제거 → BGE 리랭크 → 상위 topk 반환”을 담당하는 공통 파이프라인.”

ScienceONRetriever: LangChain BaseRetriever 그대로 옮김
BGERerankerCompressor LangChain DocumentCompressor 그대로 옮김
run_retrieval_for_queries(question, queries, …) 함수:
    각 질의별 ScienceONRetriever.get_relevant_documents() 호출
    Document들을 하나의 리스트로 합치고, CN/제목 기준 중복 제거
    BGE 리랭커로 내림차순 상위 topk 선택
    최종 Document 리스트 반환
'''

# pipelines/retrieval_pipeline.py
import logging
import time
import random
from typing import Any, Dict, List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from pipelines.scienceon_api_example import ScienceONAPIClient

# 기존 코드에서 가져온 클래스들(수정 없이 재사용)
# ScienceONRetriever: ScienceON → Document 변환
class ScienceONRetriever(BaseRetriever):
    client: ScienceONAPIClient
    k: int = 50
    fields: List[str] = ['title', 'abstract', 'author', 'link', 'publisher', 'journal', 'year', 'CN']
    class Config: arbitrary_types_allowed = True
    def _get_relevant_documents(self, query: str) -> List[Document]:
        logging.info(f"Executing search on ScienceON with query: '{query}', k={self.k}")

        for attempt in range(2):  # 최대 2번 시도
            try:
                api_results = self.client.search_articles(query=query, row_count=self.k, fields=self.fields)
                if api_results:  # 정상 결과 있으면 바로 반환
                    logging.info(f"API returned {len(api_results)} results for query: '{query}'")
                    return [Document(page_content=f"{res.get('title', '')}\n\n{res.get('abstract', '')}", metadata={k: v for k, v in res.items() if k not in ['title', 'abstract']}) for res in api_results]
                else:
                    logging.warning(f"API returned 0 results for query: '{query}' (attempt {attempt+1})")
            except Exception as e:
                logging.warning(f"Error on attempt {attempt+1} for query '{query}': {e}")

        # 모두 실패했으면 빈 리스트 반환
        logging.error(f"Failed to get results for query: '{query}' after 2 attempts.")
        return []

# BGERerankerCompressor: CrossEncoder로 rerank 
class BGERerankerCompressor(BaseDocumentCompressor):
    model: CrossEncoder
    top_k: int = 50
    class Config: arbitrary_types_allowed = True
    def compress_documents(self, documents: List[Document], query: str, callbacks: Any = None) -> List[Document]:
        if not documents: return []
        logging.info(f"Reranking {len(documents)} documents for query: '{query}'")
        doc_contents = [doc.page_content for doc in documents]
        pairs = [[query, content] for content in doc_contents]
        scores = self.model.predict(pairs, show_progress_bar=False)
        doc_score_pairs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc, score in doc_score_pairs[:self.top_k]:
            doc.metadata['rerank_score'] = float(score)
            reranked_docs.append(doc)
        return reranked_docs

def _norm(s: str) -> str:
    return (s or "").strip()

def dedupe_preserve_order_documents(docs: List[Document]) -> List[Document]:
    seen_cn = set()
    seen_title = set()
    out = []
    for d in docs:
        # title은 page_content의 첫 줄
        title = _norm((d.page_content.split("\n", 1)[0] if d.page_content else "")).lower()
        cn = _norm(d.metadata.get("CN", ""))
        if cn and cn in seen_cn:
            continue
        if title and title in seen_title:
            continue
        out.append(d)
        if cn: seen_cn.add(cn)
        if title: seen_title.add(title)
    return out

# run_retrieval_for_queries(question, queries, …): 질의 리스트 받아 검색→중복제거→리랭크→상위 topk 문서 반환
def run_retrieval_for_queries(
    question: str,
    queries: List[str],
    *,
    scienceon_client: Any,
    per_query_topk: int = 50,
    reranker: CrossEncoder,
    rerank_topk: int = 50,
    fields: List[str] = None,
    ) -> List[Document]:

    if fields is None:
        fields = ['title', 'abstract', 'author', 'link', 'publisher', 'journal', 'year', 'CN']

    retriever = ScienceONRetriever(client=scienceon_client, k=per_query_topk, fields=fields)
    all_docs: List[Document] = []
    for q in queries:
        docs = retriever.get_relevant_documents(q)
        all_docs.extend(docs)

    # 중복 제거
    de_duplicated = dedupe_preserve_order_documents(all_docs)

    # 리랭킹
    compressor = BGERerankerCompressor(model=reranker, top_k=rerank_topk)
    final_docs = compressor.compress_documents(de_duplicated, question)
    return final_docs
