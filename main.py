# python main.py --device 5

import argparse, time, asyncio
from pathlib import Path
import torch
import logging
from tqdm import tqdm
import pandas as pd
from sentence_transformers import CrossEncoder

# LangChain
from langchain_openai import ChatOpenAI

# Local module
from pipelines.scienceon_api_example import ScienceONAPIClient
from pipelines.utils import save_results, set_global_device, wait_for_vllm, launch_vllm_server
from pipelines.planners import MainPlanner
from pipelines.retrieval_pipeline import run_retrieval_for_queries
from pipelines.generation import generate_answer_with_citations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def amain():
    # --- 인수 파싱 ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./outputs") 
    ap.add_argument("--input_csv", default="./data/test.csv")
    ap.add_argument("--credentials_json", default="./configs/scienceon_api_credentials.json")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--device", type=int, default=0, help="GPU device id (예: 0, 1, 2...)")
    ap.add_argument("--per_query_topk", type=int, default=30)
    ap.add_argument("--query_topk", type=int, default=50)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lc_api_key", default="not-needed")
    ap.add_argument("--lc_temperature", type=float, default=0.1)
    ap.add_argument("--lc_max_tokens", type=int, default=2048)
    ap.add_argument("--max_llm_retries", type=int, default=10)
    args = ap.parse_args()
    
    set_global_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- vLLM 서버 자동 실행 ---
    proc = launch_vllm_server(
        model="Qwen/Qwen2.5-14B-Instruct", port=8001
    )
    wait_for_vllm(8001)

    # --- LLM 클라이언트 초기화 ---
    llm = ChatOpenAI(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        openai_api_base="http://localhost:8001/v1",
        openai_api_key=args.lc_api_key,
        temperature=args.lc_temperature,
        max_tokens=args.lc_max_tokens
    )

    planner = MainPlanner(llm, max_llm_retries=args.max_llm_retries)

    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device=('cuda' if torch.cuda.is_available() else 'cpu'))
    client = ScienceONAPIClient(credentials_path=Path(args.credentials_json))

    # --- 데이터 로딩 ---
    df = pd.read_csv(args.input_csv)
    if "id" not in df.columns: raise ValueError("input_csv must include 'id'")
    if "Question" not in df.columns: raise ValueError("input_csv must include 'Question'")
    df_to_process = df.head(args.max_rows).copy() if args.max_rows > 0 else df.copy()

    # --- 메인 루프 ---
    final_predictions = []
    final_citations = []
    all_retrieved_docs = {}
    elapsed_times_list = []

    try:
        for index in tqdm(range(len(df_to_process)), desc="전체 파이프라인 처리 중"):
            row = df_to_process.iloc[index]
            sid = str(row["id"])
            prompt_q = str(row.get("Question", "") or "").strip()
            t0 = time.perf_counter()

            # --- 1. 계획 및 검색 단계 ---
            plan = planner.generate(prompt_q, args.k)
            queries = plan.get("selected_queries", []) or []
            
            if not queries:
                logging.warning(f"[{sid}] no queries generated; skip")
                final_predictions.append("ERROR: No queries generated")
                final_citations.append("")
                all_retrieved_docs[sid] = []
                continue

            docs = run_retrieval_for_queries(question=prompt_q, queries=queries, scienceon_client=client, per_query_topk=args.per_query_topk, reranker=reranker, rerank_topk=args.query_topk)
            all_retrieved_docs[sid] = docs
            retrieval_elapsed = time.perf_counter() - t0
            logging.info(f"[{sid}] 검색 완료. {len(docs)}개 문서 검색됨. (소요 시간: {retrieval_elapsed:.2f}초)")

            # --- 2. 생성 단계 (Qwen 모델) ---
            try:
                pred, cit = await generate_answer_with_citations(prompt_q, docs, llm)
                final_predictions.append(pred.strip())
                final_citations.append(cit.strip())

            except Exception as e:
                error_message = f"ERROR during generation for query '{prompt_q[:30]}...': {e}"
                logging.error(f"\n{error_message}")
                final_predictions.append(error_message)
                final_citations.append("")

            total_elapsed = time.perf_counter() - t0
            elapsed_times_list.append(total_elapsed)
            logging.info(f"✅ ID {sid} 전체 처리 완료. (총 소요 시간: {total_elapsed:.2f}초)")

    finally:
        if 'client' in locals() and hasattr(client, 'session') and client.session:
            client.close_session()
    
    logging.info("모든 질문 처리 완료. 최종 결과 파일을 생성합니다...")
    save_results(df_to_process, final_predictions, final_citations, all_retrieved_docs, elapsed_times_list, out_dir, args.version)
    logging.info(f"\n✨ 모든 작업 완료. 최종 결과가 '{out_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(amain())