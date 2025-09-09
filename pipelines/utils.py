import logging
import requests, time, os, subprocess, atexit
from pathlib import Path
import pandas as pd
from typing import Dict, List
from langchain_core.documents import Document
from pipelines.generation import format_document_for_context

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_results(
    df: pd.DataFrame, 
    predictions: List[str], 
    citations: List[str], 
    all_retrieved_docs: Dict[str, List[Document]], 
    elapsed_times: List[float],  # <--- [수정] elapsed_times 파라미터 추가
    out_dir: Path, 
    version: str
):
    """
    - 원본 CSV의 모든 열을 유지합니다.
    - 새로 검색된 문서는 'prediction_retrieved_article_name_*' 열에 저장합니다.
    - Prediction, Citation, elapsed_times 열을 새로 추가합니다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path_csv = out_dir / f"final_submit_{version}.csv"

    # 원본 DataFrame의 복사본을 만들어 작업 수행
    output_df = df.copy()

    # 새로 생성된 결과들을 새 열로 추가
    output_df['Prediction'] = predictions
    output_df['Citation'] = citations
    output_df['elapsed_times'] = elapsed_times

    # 이번 실행에서 새로 검색된 문서들을 'prediction_retrieved_article_name_*' 열에 저장
    max_docs = 0
    if all_retrieved_docs:
        max_docs = max((len(docs) for docs in all_retrieved_docs.values()), default=0)

    # 최대 50개 문서까지 저장
    num_docs_to_save = min(max_docs, 50)
    for i in range(num_docs_to_save):
        col_name = f"prediction_retrieved_article_name_{i+1}"
        output_df[col_name] = output_df['id'].astype(str).map(
            {sid: format_document_for_context(docs[i]) if i < len(docs) else "" for sid, docs in all_retrieved_docs.items()}
        )
    
    # 만약 검색된 문서가 50개 미만일 경우, 나머지 열을 빈 문자열로 채워 스키마를 통일합니다.
    for i in range(num_docs_to_save, 50):
         col_name = f"prediction_retrieved_article_name_{i+1}"
         output_df[col_name] = ""

    # --- CSV 파일 저장 ---
    # 이제 output_df는 원본 열 + 신규 열을 모두 포함합니다.
    try:
        output_df.to_csv(output_path_csv, index=False, encoding='utf-8-sig')
        logging.info(f"CSV 결과 저장이 완료되었습니다: {output_path_csv}")
    except Exception as e:
        logging.error(f"오류: CSV 파일 저장 중 문제가 발생했습니다 - {e}")

def set_global_device(device_id: int):
    """모든 하위 프로세스와 torch/pipeline이 지정된 GPU만 보도록 환경 고정"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    logging.info(f"✅ CUDA_VISIBLE_DEVICES={device_id} 설정 완료 (모든 모듈은 이 GPU만 사용)")

def wait_for_vllm(port: int, timeout: int = 600):
    url = f"http://localhost:{port}/v1/models"
    t0 = time.time()
    while True:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                logging.info(f"✅ vLLM 서버 준비 완료: {url}")
                break
        except Exception:
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError(f"vLLM 서버가 {timeout}초 안에 준비되지 않음.")
        time.sleep(5)

def launch_vllm_server(model: str, port: int, mem_util: float = 0.8):
    """vllm 서버를 백그라운드로 실행하고 종료 시 자동으로 kill"""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(mem_util),
    ]
    proc = subprocess.Popen(cmd, env=os.environ)  
    atexit.register(proc.terminate)
    logging.info(f"🚀 vLLM 서버 실행: {model} (port={port}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    return proc