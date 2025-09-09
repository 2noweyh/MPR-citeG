# pipelines/planners.py
import re, json, time, logging
from typing import List, Dict, Any, Optional, Tuple, Optional, Tuple
from dataclasses import dataclass, field
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.base import LLM

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)
NOISE_TOKENS_DEFAULT = ["!image", "!photo", "!poster", "!ppt", "!review", "!patent", "!dataset", "!press"]
STOPWORDS = {"요약", "간략", "간략한 요약", "summary", "overview"}

class BasePlanner:
    def generate(self, question: str, k:int, sample_id: str = "NOID") -> Dict[str, List[str]]:
        """
        반환 스키마(통일):
        {
          "primary_keywords": [...],
          "expanded_keywords": [...],
          "query_strings_raw": [...],   # 문자열 쿼리들
          "selected_queries": [...]     # 실제로 검색에 사용할 최종 쿼리들
        }
        """
        raise NotImplementedError

# class HFWrapperLLM(LLM):
#     """HuggingFace 모델을 LangChain LLM처럼 쓰기 위한 래퍼"""
#     def __init__(self, model_name: str, device: str = "cuda"):
#         super().__init__()
#         logging.info(f"Loading HuggingFace KONI model: {model_name}")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype="auto",
#             device_map="auto"
#         )
#         self.pipe = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device=0 if device == "cuda" else -1,
#         )

#     def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
#         outputs = self.pipe(prompt, max_new_tokens=256, do_sample=False)
#         text = outputs[0]["generated_text"]
#         if stop:
#             for s in stop:
#                 text = text.split(s)[0]
#         return text

#     @property
#     def _identifying_params(self) -> dict:
#         return {"name": "hf-wrapper-koni"}

#     @property
#     def _llm_type(self) -> str:
#         return "huggingface"

@dataclass
class KwItem: # 키워드 하나를 표현하는 구조체.
    text: str
    lang: str = "en"           # "en" or "ko"
    syn: List[str] = field(default_factory=list)

def _is_ascii(s: str) -> bool: # 문자열이 ASCII로만 구성되었는지 판별
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False

def _choose_lang(s: str) -> str: # 비-ASCII 문자가 있으면 ko, 없으면 en
    # return "en" if _is_ascii(re.sub(r"[^\x00-\x7F]+", "", s)) else "ko"
    if re.search(r"[가-힣]", s):
        return "ko"
    return "en" if _is_ascii(s) else "ko"

def _normalize_query_string_relaxed(q: str) -> str: # and/or → 표준 연산자로 변환, 공백 정리
    """
    1) and/or → 표준화
    2) 공백 정리
    3) 괄호 주변 공백 제거
    4) 너무 많은 OR는 상위 5개까지만 남기기
    5) 따옴표가 있으면 풀어서 OR에 원본도 포함시키기
    """
    if not q or not isinstance(q, str):
        return ""

    # 괄호 주변 공백 최소화
    q = re.sub(r"\(\s+", "(", q)
    q = re.sub(r"\s+\)", ")", q)
    q = re.sub(r"\|\s+\|", "|", q)

    # OR 그룹 단순화 → 상위 5개까지만
    def _truncate_or_group(match):
        terms = [t.strip() for t in match.group(1).split("|")]
        if len(terms) > 5:
            terms = terms[:5]
        return "(" + " | ".join(terms) + ")"

    q = re.sub(r"\(([^()]+)\)", _truncate_or_group, q)

    # 따옴표 단어 완화: "A B" → ("A B" | A B)
    q = re.sub(r"\"([^\"]+)\"", r"(\"\1\" | \1)", q)

    # 공백 정리
    q = re.sub(r"\s{2,}", " ", q).strip()
    # 최종 escape 정리
    q = q.replace("\\\"", "\"").replace("\\\\", "\\")
    return q

def _dedupe_preserve_order(seq: List[str]) -> List[str]: # 리스트 중복 제거(순서 유지)
    seen = set()
    out = []
    for s in seq:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out

class MainPlanner(BasePlanner):
    """
    Mk/Hw 통합: 키워드 계층화 → 포트폴리오(A/B/C) 생성 → 정규화/노이즈 → 중복 제거 
    """
    def __init__(self,
                 llm: BaseLanguageModel,
                 max_llm_retries: int = 5,
                 noise_tokens: Optional[List[str]] = None):
        # self.llm = HFWrapperLLM("KISTI-KONI/KONI-Llama3.1-8B-Instruct-20241024")
        self.llm = llm
        self.max_llm_retries = max_llm_retries
        self.noise_tokens = noise_tokens or NOISE_TOKENS_DEFAULT
        self._sp = StrOutputParser()

        # 키워드 전용 JSON 프롬프트(동의어 배열 포함)
        RULES = """Role: You are a research assistant who designs search keywords for scientific questions.
Follow the rules below strictly to generate keywords and search queries.

Rules:
1) Primary keywords (5): Must be terms that appear in the question or very close synonyms.  
   - Rank them in order of importance (most important first).
   - Maintain scientific accuracy, use noun phrases, and avoid duplicates.
2) Expanded keywords (3): May not appear in the question but must be relevant to the intent,
   and broaden the scope with higher-level concepts, related phenomena, standard terms, or synonyms.
3) Output must be JSON only. Do not add any explanations."""

        SCHEMA = """{
"primary_keywords": [
    {"text": "<most important keyword>", "syn": "<English synonym or technical term if possible>"},
    {"text": "<second most important>", "syn": "<…>"},
    {"text": "<third>", "syn": "<…>"},
    {"text": "<fourth>", "syn": "<…>"},
    {"text": "<fifth>", "syn": "<…>"}
],
"expanded_keywords": [
    {"text": "<expanded_keyword1>", "syn": "<…>", "why": "<reason why it relates to the question's intent>"},
    {"text": "<expanded_keyword2>", "syn": "<…>", "why": "<…>"},
    {"text": "<expanded_keyword2>", "syn": "<…>", "why": "<…>"}
],
        }""".strip()


        self._kw_prompt = (
            PromptTemplate.from_template(
                "{rules}\n\nOutput schema (JSON):\n{schema}\n\nQuestion: {question}\nJSON only."
            ).partial(rules=RULES, schema=SCHEMA)
        )

        # self.translator = pipeline(
        #                             "translation", 
        #                             model="Darong/BlueT", 
        #                             tokenizer=T5TokenizerFast.from_pretrained("paust/pko-t5-base"), 
        #                             max_length=255
        #                             )
        # self.translator_ko2en = pipeline("translation", model="facebook/m2m100_418M", src_lang="ko", tgt_lang="en")
        # self.translator_ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko") 
        # self.translator_en2ko = pipeline(
        #                                 "translation",
        #                                 model="facebook/nllb-200-distilled-600M",
        #                                 src_lang="eng_Latn",
        #                                 tgt_lang="kor_Hang",
        #                                 device=0
        #                             )
        # self.translator_ko2en = pipeline(
        #                                 "translation",
        #                                 model="facebook/nllb-200-distilled-600M",
        #                                 src_lang="kor_Hang",
        #                                 tgt_lang="eng_Latn",
        #                                 device=0
        #                             )
        self.translator = pipeline(
                                        "translation",
                                        model="facebook/nllb-200-distilled-600M",
                                        device=0
                                    )
    # ---------- LLM 호출/파싱 ----------
    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            return {}
        # 1차 시도
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2차: 첫 { ... } 블록 추출
        m = _JSON_BLOCK_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}

    def _call_llm_keywords(self, question: str) -> Tuple[List[KwItem], List[KwItem], Dict[str, Any], bool]:
        retries = 0
        raw = ""
        while True:
            try:
                raw = (self._kw_prompt | self.llm | self._sp).invoke({"question": question}).strip()
                break
            except Exception as e:
                retries += 1
                print(f"LLM ERROR: {e} → retry {retries}/{self.max_llm_retries}")
                if retries >= self.max_llm_retries:
                    break
                time.sleep(2)

        data = self._safe_json_loads(raw)
        primary = []
        expanded = []

        # primary_keywords
        for x in (data.get("primary_keywords") or []):
            if isinstance(x, dict) and x.get("text"):
                syn = x.get("syn") or []
                if isinstance(syn, str):
                    syn = [syn]
                primary.append(
                    KwItem(
                        text=x["text"].strip(),
                        lang=_choose_lang(x["text"]),
                        syn=[s.strip() for s in syn if isinstance(s, str) and s.strip()],
                    )
                )

        # expanded_keywords
        for x in (data.get("expanded_keywords") or []):
            if isinstance(x, dict) and x.get("text"):
                syn = x.get("syn") or []
                if isinstance(syn, str):
                    syn = [syn]
                expanded.append(
                    KwItem(
                        text=x["text"].strip(),
                        lang=_choose_lang(x["text"]),
                        syn=[s.strip() for s in syn if isinstance(s, str) and s.strip()],
                    )
                )

        return primary, expanded, data

    # ---------- 쿼리 빌더 ----------
    def _append_noise(self, q: str) -> str: # 노이즈 억제 붙이기
        if not self.noise_tokens:
            return q
        return f"{q} " + " ".join(self.noise_tokens)

    def _translate_keyword(self, text: str, target_lang: str) -> str:
        if target_lang == "ko":
            return self.translator(text, src_lang="eng_Latn",tgt_lang="kor_Hang")[0]['translation_text']
        elif target_lang == "en":
            return self.translator(text,src_lang="kor_Hang",tgt_lang="eng_Latn")[0]['translation_text']
        return text

    def _filter_keywords(self, items: List[KwItem]) -> List[KwItem]:
        out = []
        for kw in items:
            if kw.text not in STOPWORDS:
                kw.syn = [s for s in kw.syn if s not in STOPWORDS]
                out.append(kw)
        return out
    
    def _ngram_split(self, text: str) -> str:
        tokens = re.split(r"\s+", text)
        if len(tokens) > 1:
            return "(" + " | ".join(tokens) + ")"
        return text

    # ---------- 포트폴리오 빌더 ----------
    def _build_portfolio(self, primary: List[KwItem], expanded: List[KwItem]) -> Dict[str, Dict[str, Dict[str, str]]]:
        # Primary: main1, main2만 사용
        main1 = primary[0] if len(primary) > 0 else KwItem(text="", lang="en", syn=[])
        main2 = primary[1] if len(primary) > 1 else None
        main3 = primary[2] if len(primary) > 2 else None
        main4 = primary[3] if len(primary) > 3 else None
        main5 = primary[4] if len(primary) > 4 else None
        exp1 = expanded[0] if len(expanded) > 0 else None
        exp2 = expanded[1] if len(expanded) > 1 else None
        exp3 = expanded[2] if len(expanded) > 2 else None

        # ---------- Q 블록 ----------
        q_list = []
        for m in [main1, main2, main3]:
            if m and m.text:
                terms = [f"({e.text})" for e in [exp1, exp2, exp3] if e and e.text]
                if terms:
                    q_list.append(f"({m.text}) ({' | '.join(terms)})")
                else:
                    q_list.append(f"({m.text})")

        # ---------- S 블록 ----------
        s_list = []
        for m in [main1, main2, main3]:
            if m and m.text:
                syns = m.syn[:3]  # 동의어 최대 3개
                if syns:
                    terms = [f"({e.text})*" for e in [exp1, exp2, exp3] if e and e.text]
                    query = f"((({m.text}) | ({' | '.join(syns)}))"
                    if terms:
                        query += f" ({' | '.join(terms)})"
                    query += ")"
                    s_list.append(self._append_noise(query))

        # ---------- F 블록 ----------
        f_list = []
        if len(primary) >= 4:
            main_group = " | ".join([f"({kw.text})" for kw in primary[:3] if kw and kw.text])
            main4_F = primary[3].text if primary[3] and primary[3].text else None
            if main_group and main4_F:
                f_list.append(f"(({main_group}) {main4_F})")
        if len(primary) >= 5:
            main_group = " | ".join([f"({kw.text})" for kw in primary[:3] if kw and kw.text])
            main5_F = primary[4].text if primary[4] and primary[4].text else None
            if main_group and main5_F:
                f_list.append(f"(({main_group}) {main5_F})")

        # ---------- G 블록 ----------
        g_list = []

        main_terms = []
        for m in [main1, main2, main3]:
            if m and m.text:
                main_trans = self._translate_keyword(m.text, "en" if m.lang == "ko" else "ko")
                term = f"{re.sub(r'[.,;:]+$', '', main_trans.strip())}"
                main_terms.append(f"({term})")

        exp_terms = []
        for e in [exp1, exp2, exp3]:
            if e and e.text:  
                exp_trans = self._translate_keyword(e.text, "en" if e.lang == "ko" else "ko")
                term = f"{re.sub(r'[.,;:]+$', '', exp_trans.strip())}"
                exp_terms.append(f"({term})")

        # (기본 키워드1 | ... | 5)
        if main_terms:
            g_list.append(f"({' | '.join(main_terms)})")

        # (기본 키워드...) (확장 키워드...)
        if main_terms and exp_terms:
            g_list.append(f"({' | '.join(main_terms)}) ({' | '.join(exp_terms)})")


        portfolio = {
            "Q": {f"Q-{i+1}": {"q": q, "rationale": "메인 + 확장 OR 조합"} for i, q in enumerate(q_list)},
            "S": {f"S-{i+1}": {"q": q, "rationale": "메인 + 동의어 OR"} for i, q in enumerate(s_list)},
            "F": {f"F-{i+1}": {"q": q, "rationale": "메인 그룹 + 추가 Primary 결합"} for i, q in enumerate(f_list)},
            "G": {f"G-{i+1}": {"q": q, "rationale": "메인+확장 번역"} for i, q in enumerate(g_list)},
            # "N": {f"N-{i+1}": {"q": q, "rationale": "확장 와일드카드 + 노이즈 제거"} for i, q in enumerate(n_list)},
            }
        return portfolio

    # ---------- 키워드 생성 빌더 ----------
    def generate(self, question: str, k: int, sample_id: str = "NOID") -> dict:
        # 1) 키워드 수집
        primary, expanded, raw_json = self._call_llm_keywords(question)
        primary = self._filter_keywords(primary)
        expanded = self._filter_keywords(expanded)

        # 2) 포트폴리오 생성
        portfolio = self._build_portfolio(primary, expanded)

        # 3) 쿼리 수집 (새 포트폴리오 구조 기반)
        ordered = []
        for block in ["Q", "S", "F", "G", "N"]:
            for _, item in portfolio.get(block, {}).items():
                ordered.append(item["q"])

        # 4) 정규화 + 중복 제거
        normalized = [_normalize_query_string_relaxed(q) for q in ordered if q and q.strip()]
        normalized = _dedupe_preserve_order(normalized)

        # 5) 최종 top-K
        selected = normalized

        # 6) 결과 구성
        primary_list = [kw.text for kw in primary]
        expanded_list = [kw.text for kw in expanded]

        result = {
            "selected_queries": selected,
            "query_strings_raw": normalized,
            "primary_keywords":primary_list,
            "expanded_keywords":expanded_list,
            "artifacts": {
                "sample_id": sample_id,
                "question": question,
                "queries_by_level": portfolio,
                "queries_ordered": ordered,
                "queries_normalized": normalized,
                "raw_json": raw_json,
            },
        }
        return result