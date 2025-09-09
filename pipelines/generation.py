import re
import asyncio
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm.asyncio import tqdm_asyncio

def format_document_for_context(doc: Document) -> str:
    """LLM 컨텍스트 및 결과 저장을 위해 Document 객체를 문자열로 변환합니다."""
    title = (doc.page_content.split("\n\n", 1)[0] if doc.page_content else "").replace("\n", " ").strip()
    abstract = (doc.page_content.split("\n\n", 1)[1] if "\n\n" in doc.page_content else "").replace("\n", " ").strip()
    link = doc.metadata.get("link", 'N/A') or ""
    parts = []
    if title:    parts.append(f"Title: {title}")
    if abstract: parts.append(f"Abstract: {abstract}")
    if link:     parts.append(f"Source: {link}")
    return "\n".join(parts)


async def _find_source_for_sentence(sentence: str, formatted_docs_with_num: str, llm: ChatOpenAI) -> int:
    """하나의 문장에 대해 가장 적합한 출처 문서 번호를 찾는 내부 비동기 작업"""
    citation_sys_prompt = """You are a meticulous assistant that verifies if a given sentence is directly supported by a list of source documents.
    [TASK]
    Your task is to find the SINGLE source document that provides a DIRECT and VERIFIABLE origin for the sentence.

    [CRITICAL INSTRUCTIONS]
    1.  **Prioritize finding a citation.** Your main goal is to ground the text. You should cite any sentence that contains specific information (names, numbers, key concepts, factual claims) that can be reasonably traced back to a single source document.
    2.  **A citation MUST be added if the sentence contains a key factual claim.** Do not be overly strict; if the information plausibly comes from a source, cite it.
    3.  Only respond with 0 for sentences that are purely generic, introductory, or transitional and lack specific, verifiable information from the sources.

    [RESPONSE FORMAT]
    - If you find a verifiable source, respond with ONLY the number of that source. (e.g., 3)
    - If the sentence absolutely cannot be sourced according to the rules above, respond with ONLY the number 0.
    """
    citation_usr_template = "Sentence: \"{sentence}\"\n\nNumbered Sources:\n{sources}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", citation_sys_prompt),
        ("human", citation_usr_template),
    ])
    chain = prompt | llm | StrOutputParser()

    try:
        response = await chain.ainvoke({"sentence": sentence, "sources": formatted_docs_with_num})
        match = re.search(r'\d+', response)
        if match:
            return int(match.group(0))
        return 0
    except Exception:
        return 0


async def _process_citations_stage2(raw_prediction: str, retrieved_docs: List[Document], llm: ChatOpenAI) -> Tuple[str, str]:
    """생성된 답변 텍스트를 기반으로 인용을 추가하는 2단계 프로세스"""
    sentences = re.split(r'(?<=[.!?])\s+', raw_prediction.strip())
    sentences = [s for s in sentences if s]

    if not sentences or not retrieved_docs:
        return raw_prediction, ""

    valid_docs = [doc for doc in retrieved_docs if doc.metadata.get("link")]
    if not valid_docs:
        return raw_prediction, ""

    formatted_docs_with_num = "\n\n".join(
        f"[{i+1}] {format_document_for_context(doc)}" for i, doc in enumerate(valid_docs)
    )

    tasks = [_find_source_for_sentence(s, formatted_docs_with_num, llm) for s in sentences]
    source_indices = await tqdm_asyncio.gather(*tasks, desc="   - Grounding citations", leave=False)

    final_citation_map = {}
    doc_to_citation_marker = {}
    sentence_to_marker = {}
    current_marker = 1

    for i, doc_index_plus_one in enumerate(source_indices):
        if 0 < doc_index_plus_one <= len(valid_docs):
            original_doc_index = doc_index_plus_one - 1
            source_link = valid_docs[original_doc_index].metadata.get("link")
            if original_doc_index not in doc_to_citation_marker:
                doc_to_citation_marker[original_doc_index] = current_marker
                final_citation_map[current_marker] = source_link
                current_marker += 1
            sentence_to_marker[i] = doc_to_citation_marker[original_doc_index]

    final_sentences = []
    for i, sentence in enumerate(sentences):
        marker = sentence_to_marker.get(i)
        final_sentences.append(f"{sentence} [{marker}]" if marker else sentence)
    final_prediction_text = " ".join(final_sentences)

    if not final_citation_map:
        return final_prediction_text, ""

    citation_lines = []
    for marker in sorted(final_citation_map.keys()):
        link = final_citation_map[marker]
        citation_lines.append(f"[{marker}] {link}")
    final_citation_block = "\n".join(citation_lines)

    return final_prediction_text, final_citation_block


async def generate_answer_with_citations(
    question: str,
    retrieved_docs: List[Document],
    llm: ChatOpenAI
) -> Tuple[str, str]:
    if not retrieved_docs:
        return "ERROR: No documents found", ""

    # --- 1단계: 초기 답변 생성 ---
    gen_sys_prompt_stage1 = """
    You are a meticulous Question & Answering chatbot in the scientific domain.
    Your task is to answer the user's question strictly based on the provided documents.

    [CRITICAL Instructions]
    1.  Read the user's question and the retrieved documents carefully.
    2.  Formulate a clear and concise answer using the information present in the documents.
    3.  Language policy: Generate the entire output in the SAME language as the user's question (English → English, Korean → Korean).

    [Example Output Structure]
    ##Title##
    <Concise, document-grounded title>

    ##Introduction##
    <2–3 sentences based only on the documents> 

    ##Main Body##
    <2–3 sentences explaining the key points based on the documents>

    ##Conclusion##
    <2–3 sentences summarizing the main takeaways based on the documents> 
    """
    gen_usr_prompt_template_stage1 = "Please answer the question based on the retrieved documents.\nQuestion: {input}\n\nRetrieved documents:\n{retrieved_docs}"
    generation_prompt_stage1 = ChatPromptTemplate.from_messages([("system", gen_sys_prompt_stage1), ("human", gen_usr_prompt_template_stage1)])
    generation_chain_stage1 = generation_prompt_stage1 | llm | StrOutputParser()

    # 컨텍스트 생성을 위해 상위 10개 문서만 사용
    docs_for_context = retrieved_docs[:10]
    retrieved_docs_str = "\n\n".join([format_document_for_context(doc) for doc in docs_for_context])

    if not retrieved_docs_str.strip():
        return "ERROR: No documents could be fitted into the context.", ""
        
    raw_prediction = await generation_chain_stage1.ainvoke({
        "input": question,
        "retrieved_docs": retrieved_docs_str
    })

    # --- 2단계: 인용 추가 ---
    # 1단계 컨텍스트 생성에 사용된 문서를 인용 검증 단계로 전달
    final_prediction, final_citation = await _process_citations_stage2(raw_prediction, docs_for_context, llm)

    return final_prediction, final_citation