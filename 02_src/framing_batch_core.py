"""Shared OpenAI Batch helpers: L1-first pipeline, then L2 conditioned on L1."""
import hashlib
import json
import logging
import random
import re
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional

import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)


def _merge_annotation_quality_report(run_dir: Path, section_key: str, section: dict) -> None:
    """Merge one section into run_dir/_annotation_quality_report.json (retrieve diagnostics)."""
    path = run_dir / "_annotation_quality_report.json"
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    data[section_key] = section
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class TaiwanBatchManager:
    def __init__(self, api_key: str, model: str = "gpt-5.1", timeout: float = 600.0):
        if not api_key:
            raise ValueError("API Key is missing.")
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.l1_guide = pd.DataFrame()
        self.l2_guide = pd.DataFrame()
        self.l1_fewshot = pd.DataFrame()
        self.l2_fewshot = pd.DataFrame()
        self.hard_pool_df = pd.DataFrame()
        self.hard_pool_enabled = False
        self.hard_pool_n_l1 = 4
        self.hard_pool_n_l2 = 5
        self.hard_pool_seed: Optional[int] = None

    def load_hard_pool(
        self,
        path: Optional[str] = None,
        n_l1: int = 4,
        n_l2: int = 5,
        seed: Optional[int] = None,
        enabled: bool = True,
    ):
        """Load CSV pool; stratified subsamples are drawn per request (see _rng_for_request_id)."""
        self.hard_pool_n_l1 = max(0, int(n_l1))
        self.hard_pool_n_l2 = max(0, int(n_l2))
        self.hard_pool_seed = seed
        self.hard_pool_df = pd.DataFrame()
        self.hard_pool_enabled = False
        if not enabled:
            return
        if not path or not Path(path).exists():
            return
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:
            logger.warning("hard pool read failed %s: %s", path, e)
            return
        need = {"layer", "stratum", "sentence", "gold_L1"}
        if not need.issubset(df.columns):
            logger.warning("hard pool missing columns %s — need %s", list(df.columns), need)
            return
        self.hard_pool_df = df
        self.hard_pool_enabled = True
        logger.info("Hard-case pool enabled rows=%s (n_l1=%s n_l2=%s)", len(df), n_l1, n_l2)

    @staticmethod
    def _rng_for_request_id(global_seed: Optional[int], request_id: str) -> random.Random:
        if global_seed is not None:
            h = int(
                hashlib.md5(f"{global_seed}:{request_id}".encode("utf-8")).hexdigest()[:12],
                16,
            )
            return random.Random(h)
        return random.Random()

    @staticmethod
    def stratified_sample_rows(df: pd.DataFrame, stratum_col: str, n: int, rng: random.Random) -> pd.DataFrame:
        """Roughly balanced draw across strata; fill shortfall from remainder; last resort with replacement."""
        if df.empty or n <= 0:
            return df.iloc[0:0].copy()
        work = df.copy()
        work["_s"] = work[stratum_col].fillna("").astype(str).str.strip()
        work.loc[work["_s"] == "", "_s"] = "_other"
        strata = work["_s"].unique().tolist()
        rng.shuffle(strata)
        k = len(strata)
        base, rem = n // k, n % k
        quotas = {s: base + (1 if i < rem else 0) for i, s in enumerate(strata)}
        picked_parts: List[pd.DataFrame] = []
        picked_idx = set()
        for s in strata:
            sub = work[work["_s"] == s].drop(columns=["_s"], errors="ignore")
            q = quotas.get(s, 0)
            if sub.empty or q <= 0:
                continue
            rs = rng.randint(0, 2**31 - 1)
            take = min(q, len(sub))
            sm = sub.sample(n=take, random_state=rs, replace=False)
            picked_parts.append(sm)
            picked_idx.update(sm.index.tolist())
        got = (
            pd.concat(picked_parts, axis=0)
            if picked_parts
            else work.drop(columns=["_s"], errors="ignore").iloc[0:0]
        )
        if len(got) < n:
            rest = (
                work.drop(index=list(picked_idx), errors="ignore")
                .drop(columns=["_s"], errors="ignore")
            )
            need = n - len(got)
            if not rest.empty:
                rs = rng.randint(0, 2**31 - 1)
                extra = rest.sample(n=min(need, len(rest)), random_state=rs, replace=False)
                got = pd.concat([got, extra], axis=0)
        if len(got) < n and len(work) > 0:
            need = n - len(got)
            pool = work.drop(columns=["_s"], errors="ignore")
            rs = rng.randint(0, 2**31 - 1)
            fill = pool.sample(n=need, random_state=rs, replace=True)
            got = pd.concat([got, fill], axis=0)
        return got.head(n).reset_index(drop=True)

    def _sample_hard_l1(self, rng: random.Random) -> pd.DataFrame:
        df = self.hard_pool_df[self.hard_pool_df["layer"].astype(str).isin(["L1", "BOTH"])].copy()
        if df.empty:
            return df
        return self.stratified_sample_rows(df, "stratum", self.hard_pool_n_l1, rng)

    def _sample_hard_l2(self, rng: random.Random) -> pd.DataFrame:
        df = self.hard_pool_df[self.hard_pool_df["layer"].astype(str).isin(["L2", "BOTH"])].copy()
        if "gold_L2" in df.columns:
            g = df["gold_L2"].fillna("").astype(str).str.strip()
            df = df[g != ""]
        if df.empty:
            return df
        return self.stratified_sample_rows(df, "stratum", self.hard_pool_n_l2, rng)

    def _format_rotating_hard_cases_l1(self, rng: random.Random) -> str:
        samp = self._sample_hard_l1(rng)
        if samp.empty:
            return ""
        lines = [
            "\n**【本輪動態 Hard-case 金標示例（L1）】**\n",
            "下列條目來自審閱共識之歧義/錯誤案例庫；每條請意會邊界判準，**禁止**機械模仿用詞或句式。\n\n",
        ]
        for i, (_, r) in enumerate(samp.iterrows(), 1):
            sent = str(r["sentence"]).strip()
            sent_j = json.dumps(sent, ensure_ascii=False)
            l1 = self._normalize_l1_code(str(r["gold_L1"]).strip())
            cot = str(r.get("gold_L1_cot") or "").strip() or "人工複審共識"
            ex = json.dumps({"L1_cot": cot, "L1": l1}, ensure_ascii=False)
            lines.append(f"[HC-L1-{i}] 文本：{sent_j}\n參考輸出：{ex}\n")
            note = str(r.get("notes") or "").strip()
            if note:
                lines.append(f"類型：{note}\n")
            lines.append("\n")
        return "".join(lines)

    def _format_rotating_hard_cases_l2(self, rng: random.Random) -> str:
        samp = self._sample_hard_l2(rng)
        if samp.empty:
            return ""
        lines = [
            "\n**【本輪動態 Hard-case 金標示例（L2，含 L1 前提）】**\n",
            "下列條目與 L1 先行情境一致；參考輸出為複審共識之多標籤結果。請意會證據與邊界，**禁止**抄襲字面。\n\n",
        ]
        for i, (_, r) in enumerate(samp.iterrows(), 1):
            sent = str(r["sentence"]).strip()
            sent_j = json.dumps(sent, ensure_ascii=False)
            l1 = self._normalize_l1_code(str(r["gold_L1"]).strip())
            raw_l2 = str(r.get("gold_L2") or "").strip()
            l2_list = [x.strip() for x in raw_l2.split("|") if x.strip()]
            l2_cot = str(r.get("gold_L2_cot") or "").strip() or "複審共識多標籤"
            ex = json.dumps({"L2_cot": l2_cot, "L2": l2_list}, ensure_ascii=False)
            lines.append(
                f"[HC-L2-{i}] 已知L1：{l1}\n文本：{sent_j}\n參考輸出：{ex}\n"
            )
            note = str(r.get("notes") or "").strip()
            if note:
                lines.append(f"類型：{note}\n")
            lines.append("\n")
        return "".join(lines)

    def load_guides(
        self,
        l1_path: Optional[str] = None,
        l2_path: Optional[str] = None,
        l1_fewshot_path: Optional[str] = None,
        l2_fewshot_path: Optional[str] = None,
    ):
        try:
            self.l1_guide = (
                pd.read_csv(l1_path) if l1_path and Path(l1_path).exists() else pd.DataFrame()
            )
            self.l2_guide = (
                pd.read_csv(l2_path) if l2_path and Path(l2_path).exists() else pd.DataFrame()
            )
            logger.info("Codebooks L1=%s L2=%s", len(self.l1_guide), len(self.l2_guide))
        except Exception as e:
            logger.error("load codebooks: %s", e)

        for path, attr in [(l1_fewshot_path, "l1_fewshot"), (l2_fewshot_path, "l2_fewshot")]:
            if path and Path(path).exists():
                try:
                    setattr(self, attr, pd.read_csv(path))
                    logger.info("Few-shot %s rows=%s", attr, len(getattr(self, attr)))
                except Exception as e:
                    logger.warning("few-shot %s: %s", path, e)

        for attr, expected_cols in [
            ("l2_fewshot", ["label_id", "Fewshot_v1", "Fewshot_v2", "Fewshot_contrast"]),
            ("l1_fewshot", ["label_id", "Fewshot_v1", "Fewshot_v2", "Fewshot_contrast"]),
        ]:
            df = getattr(self, attr)
            if not df.empty:
                missing = [c for c in expected_cols if c not in df.columns]
                if missing:
                    logger.warning("[%s] missing columns %s (skipped)", attr, missing)

    def _format_criteria(self, val, prefix_symbol):
        if pd.notna(val) and str(val).strip() not in ["", "nan", "N/A", "None"]:
            return f" {prefix_symbol} {str(val).strip()}\n"
        return ""

    def _format_fewshots(self, df_fewshot, label_id):
        if df_fewshot is None or df_fewshot.empty:
            return ""
        row = df_fewshot[df_fewshot["label_id"].astype(str) == str(label_id)]
        if row.empty:
            return ""
        fs_str = ""
        r0 = row.iloc[0]
        for v, prefix in [(r0.get("Fewshot_v1"), "[例1]"), (r0.get("Fewshot_v2"), "[例2]")]:
            if pd.notna(v) and str(v).strip() not in ["", "nan", "N/A", "None"]:
                fs_str += f" {prefix} {str(v).strip()}\n"
        c = r0.get("Fewshot_contrast")
        if pd.notna(c) and str(c).strip() not in ["", "nan", "N/A", "None"]:
            fs_str += f" [對比] {str(c).strip()}\n"
        return fs_str

    def _build_l2_system_prompt(self, rng: Optional[random.Random] = None) -> str:
        prompt = (
            "角色:台灣政治文本分析專家。任務:標註L2(國族建構,多選)。\n"
            "符號說明: 「+」後文字來自 codebook 之 inclusion_criteria；「-」來自 exclusion_criteria；"
            "「⚖」後為 boundary_test（邊界操作化測試）。"
            "[例1][例2]為 fewshot 正例；若列有「對比」則為與他標之邊界對照（非額外正例）。\n\n"
            "分析步驟(CoT):\n"
            "1. L2_cot: 執行【多標籤窮盡掃描】。L2為多選，絕對不能只抓最強烈的單一訊號！"
            "找到主要標籤後強制反問：『還有其他面向嗎？』"
            "L2_cot請控制在50字以內，僅列標籤編號與關鍵證據詞。\n"
            "2. L2: 填寫L2標籤陣列，盡可能窮盡所有符合的標籤，無則填[]。\n\n"
            "**【寧多勿漏原則】**\n"
            "寧可多標後在排除規則中剔除，不可因不確定就直接輸出[]。"
            "只有文本完全不含任何國族建構信號（無主權、無威脅、無認同、無動員、無價值論述）時才輸出[]。\n\n"
            "**【召回優先】**\n"
            "當文本可能同時屬於多個 L2 類別時，傾向於標註而非遺漏。"
            "多標籤任務中召回率優於精確率：漏標比誤標的研究代價更高——誤標可在後續人工審核中修正，漏標則直接丟失資料。\n\n"
            "特別注意：經濟政見中的願景號召（打造、讓台灣成為、一起加油）仍可觸發L2-07；"
            "社會議題中的資源分配訴求仍可觸發L2-05或L2-07。不可因文本看似「純經濟」「純民生」就跳過L2掃描。\n\n"
            "**【L2 國族建構】定義**\n"
        )
        if not self.l2_guide.empty:
            for _, row in self.l2_guide.iterrows():
                label_id = row.get("label_id")
                prompt += f"[{label_id}_{row.get('label_cn')}] 定義:{row.get('core_definition', '')}\n"
                prompt += self._format_criteria(row.get("inclusion_criteria"), "+")
                prompt += self._format_criteria(row.get("exclusion_criteria"), "-")
                prompt += self._format_criteria(row.get("boundary_test"), "⚖")
                prompt += self._format_fewshots(
                    self.l2_fewshot if not self.l2_fewshot.empty else None, label_id
                )

        if self.hard_pool_enabled and rng is not None and self.hard_pool_n_l2 > 0:
            prompt += self._format_rotating_hard_cases_l2(rng)

        valid_l2_ids = (
            ", ".join(self.l2_guide["label_id"].astype(str).tolist()) if not self.l2_guide.empty else ""
        )

        prompt += (
            "\n**【完整分析範例示範】**\n"
            "範例一（多標籤，主體性+差異化）：\n"
            "輸入：「柯文哲表示，台灣有自己的政府和軍隊，台灣絕對不會是香港。」\n"
            '輸出：{"L2_cot": "政府軍隊→L2-01；台港對比→L2-02", "L2": ["L2-01", "L2-02"]}\n\n'
            "範例二（多標籤窮盡掃描）：\n"
            "輸入：「蔡英文強調，台灣是民主國家，民眾是國家主人。"
            "我們遵守憲法，也積極爭取國際社會支持加入CPTPP。」\n"
            '輸出：{"L2_cot": "民主國家→L2-08；憲法主人→L2-01；CPTPP→L2-03", '
            '"L2": ["L2-01", "L2-03", "L2-08"]}\n'
        )

        prompt += (
            "\n**【嚴格輸出規範】**\n"
            "絕對禁止發明未定義的標籤。你的輸出必須嚴格選自以下列表：\n"
            f"合法 L2: [{valid_l2_ids}]\n\n"
            "**輸出JSON:**\n"
            "{\n"
            '  "L2_cot": "關鍵詞→標籤（≤50字）",\n'
            '  "L2": ["必須嚴格選自合法L2列表，盡可能窮盡所有符合的標籤，無則填[]"]\n'
            "}"
        )
        prompt += (
            "\n若提供前文/後文，僅作為理解本句語境的輔助，標註對象嚴格限定為【本句】。"
        )
        return prompt

    def _build_l1_system_prompt(self, rng: Optional[random.Random] = None) -> str:
        prompt = (
            "角色:台灣政治文本分析專家。任務:標註L1(具體語境,單選)。\n"
            "符號說明: 「+」後文字來自 inclusion_criteria；「-」來自 exclusion_criteria；"
            "「⚖」後為 boundary_test（邊界操作化測試）。"
            "[例1][例2]為 fewshot 正例；若有「對比」則為邊界對照。\n\n"
            "分析步驟(CoT):\n"
            "1. L1_cot: ≤20字，列選定標籤+排除最易混淆項。\n"
            "2. L1: 填單一標籤。\n\n"
            "**【L1 具體語境】定義**\n"
        )
        if not self.l1_guide.empty:
            for _, row in self.l1_guide.iterrows():
                label_id = row.get("label_id")
                prompt += f"[{label_id}_{row.get('label_cn')}] 定義:{row.get('core_definition', '')}\n"
                prompt += self._format_criteria(row.get("inclusion_criteria"), "+")
                prompt += self._format_criteria(row.get("exclusion_criteria"), "-")
                prompt += self._format_criteria(row.get("boundary_test"), "⚖")
                prompt += self._format_fewshots(
                    self.l1_fewshot if not self.l1_fewshot.empty else None, label_id
                )

        valid_l1_ids = (
            ", ".join(self.l1_guide["label_id"].astype(str).tolist()) if not self.l1_guide.empty else ""
        )

        prompt += (
            "\n**【跨標籤強制檢查】**\n"
            "【L1-03 vs L1-06 強制檢查】標註前必須執行：①掃描安全化語法三要素（存在性威脅/不可逆時點/超常規措施）→"
            "符合兩項以上→L1-03，無論是否出現政策語言。②僅符合零至一項→檢查處方建議性質：對抗性→L1-03，制度性→L1-06。\n"
            "【L1-02 vs L1-07 三步測試】①文本是否明示或隱含一條道德原則？→是則傾向L1-02。"
            "②移除情緒語言後，道德主張是否仍可辨識？→是則L1-02。"
            "③移除道德語言後，情緒訴求是否仍為文本主功能？→是則L1-07。\n\n"
        )

        if self.hard_pool_enabled and rng is not None and self.hard_pool_n_l1 > 0:
            prompt += self._format_rotating_hard_cases_l1(rng)

        prompt += (
            "\n**【嚴格輸出規範】**\n"
            "絕對禁止發明未定義的標籤。你的輸出必須嚴格選自以下列表：\n"
            f"合法 L1: [{valid_l1_ids}]\n\n"
            "L1 欄位必須僅為編號（例如 L1-03），不得附加中文標籤名或其他後綴。\n\n"
            "**輸出JSON:**\n"
            "{\n"
            '  "L1_cot": "標籤+排除項（≤20字）",\n'
            '  "L1": "必須嚴格選自合法L1列表"\n'
            "}"
        )
        prompt += (
            "\n若提供前文/後文，僅作為理解本句語境的輔助，標註對象嚴格限定為【本句】。"
        )
        return prompt

    @staticmethod
    def _normalize_l1_code(raw: str) -> str:
        """Map values like 'L1-03_衝突與安全' -> 'L1-03' when prefix matches codebook pattern."""
        s = str(raw).strip()
        m = re.match(r"^(L1-\d+)", s)
        return m.group(1) if m else s

    @staticmethod
    def _context_field_strip(row: pd.Series, key: str) -> str:
        v = row.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v).strip()

    @staticmethod
    def _build_user_text(row: pd.Series, plain_instruction: bool = True) -> str:
        """User-facing text block: optional prev/next context; target is always 本句.

        If plain_instruction is True (L1 / parallel user message), empty context uses
        「請標註以下文本」+ sentence. If False (e.g. embedded under L2 task header), empty
        context is sentence only.
        """
        sentence = TaiwanBatchManager._context_field_strip(row, "sentence")
        prev = TaiwanBatchManager._context_field_strip(row, "prev")
        next_ = TaiwanBatchManager._context_field_strip(row, "next")
        if not prev and not next_:
            if plain_instruction:
                return f"請標註以下文本：\n\n{sentence}"
            return sentence
        return (
            f"【前文（僅供語境參考，不標註）】{prev}\n"
            f"【本句（請標註）】{sentence}\n"
            f"【後文（僅供語境參考，不標註）】{next_}"
        )

    def prepare_l1_batch_from_input(self, input_folder: str, output_jsonl: str, reference_csv: str) -> int:
        """All non-empty sentences → L1 batch (same discovery rules as legacy L2-from-input)."""
        input_path = Path(input_folder).resolve()
        if not input_path.exists():
            logger.error("Input missing: %s", input_path)
            return 0
        if input_path.is_file() and input_path.suffix.lower() == ".csv":
            csv_files = [input_path]
        elif input_path.is_dir():
            csv_files = list(input_path.glob("**/*.csv"))
        else:
            csv_files = []
        if not csv_files:
            logger.warning("No CSV under %s", input_path)
            return 0

        count = 0

        with open(output_jsonl, "w", encoding="utf-8") as f_jsonl:
            ref_rows = []
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    if "sentence" not in df.columns:
                        logger.warning("Skip %s: no sentence column", file_path.name)
                        continue
                    if "id" not in df.columns:
                        df["id"] = df.index

                    for _, row in df.iterrows():
                        if pd.isna(row.get("sentence")) or str(row["sentence"]).strip() == "":
                            continue
                        unique_id = f"req_{uuid.uuid4().hex}"
                        rng = TaiwanBatchManager._rng_for_request_id(self.hard_pool_seed, unique_id)
                        system_prompt = self._build_l1_system_prompt(rng)
                        user_content = self._build_user_text(row)
                        request_body = {
                            "custom_id": unique_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.model,
                                "messages": [
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_content},
                                ],
                                "temperature": 0,
                                "response_format": {"type": "json_object"},
                            },
                        }
                        f_jsonl.write(json.dumps(request_body, ensure_ascii=False) + "\n")
                        row_data = row.to_dict()
                        row_data["custom_id"] = unique_id
                        row_data["source_file"] = file_path.name
                        ref_rows.append(row_data)
                        count += 1
                except Exception as e:
                    logger.error("read %s: %s", file_path.name, e)

        if ref_rows:
            pd.DataFrame(ref_rows).to_csv(reference_csv, index=False, encoding="utf-8-sig")
            logger.info("L1 batch: %s requests -> %s", count, output_jsonl)
            return count
        logger.warning("L1 batch: no rows (check sentence column and non-empty cells)")
        return 0

    def prepare_l2_batch_after_l1(
        self, results_l1_csv: str, output_jsonl: str, reference_csv: str
    ) -> int:
        """L2 batch: user message includes prior L1 label as conditioning context."""
        df = pd.read_csv(results_l1_csv)
        if "custom_id" not in df.columns:
            logger.error("results_l1 CSV missing custom_id column")
            return 0

        count = 0

        with open(output_jsonl, "w", encoding="utf-8") as f_jsonl:
            ref_rows = []
            for _, row in df.iterrows():
                if pd.isna(row.get("sentence")) or str(row["sentence"]).strip() == "":
                    continue
                cid = row["custom_id"]
                rng = TaiwanBatchManager._rng_for_request_id(self.hard_pool_seed, str(cid))
                system_prompt = self._build_l2_system_prompt(rng)
                l1_val = row.get("L1_label")
                if pd.notna(l1_val) and str(l1_val).strip() != "":
                    l1_display = str(l1_val).strip()
                else:
                    l1_display = "（前階段未取得有效 L1，請仍對全文掃描 L2 信號）"
                body_text = self._build_user_text(row, plain_instruction=False)
                user_block = (
                    f"【已知條件】本句經前階段標註之 L1 具體語境為：{l1_display}\n\n"
                    f"【任務】在以上 L1 語境前提下，請對下列文本進行 L2（國族建構）多選標註。\n\n"
                    f"{body_text}"
                )
                request_body = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_block},
                        ],
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"},
                    },
                }
                f_jsonl.write(json.dumps(request_body, ensure_ascii=False) + "\n")
                ref_rows.append(row.to_dict())
                count += 1

        if ref_rows:
            pd.DataFrame(ref_rows).to_csv(reference_csv, index=False, encoding="utf-8-sig")
            logger.info("L2 batch (L1-conditioned): %s requests -> %s", count, output_jsonl)
            return count
        logger.warning("L2 batch: no rows")
        return 0

    def prepare_parallel_l1_l2(
        self, input_folder: str, l1_jsonl: str, l2_jsonl: str, reference_csv: str
    ) -> int:
        """
        Build L1 + L2 batch JSONLs with the same custom_id per row (no L1→L2 dependency).
        L2 user message is sentence-only (no L1 conditioning), same as legacy L2-first API shape.
        """
        input_path = Path(input_folder).resolve()
        if not input_path.exists():
            logger.error("Input missing: %s", input_path)
            return 0
        if input_path.is_file() and input_path.suffix.lower() == ".csv":
            csv_files = [input_path]
        elif input_path.is_dir():
            csv_files = sorted(input_path.glob("**/*.csv"), key=lambda p: str(p).lower())
        else:
            csv_files = []
        if not csv_files:
            logger.warning("No CSV under %s", input_path)
            return 0

        count = 0
        ref_rows = []

        logger.info(
            "prepare_parallel_l1_l2: input=%s (%s CSV file(s)); each row rebuilds full L1/L2 system prompts (CPU-heavy, can take many minutes on large CSVs)",
            input_path,
            len(csv_files),
        )

        with open(l1_jsonl, "w", encoding="utf-8") as f1, open(l2_jsonl, "w", encoding="utf-8") as f2:
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    if "sentence" not in df.columns:
                        logger.warning("Skip %s: no sentence column", file_path.name)
                        continue
                    if "id" not in df.columns:
                        df["id"] = df.index
                    logger.info(
                        "prepare_parallel_l1_l2: processing %s (%s rows, non-empty sentence TBD)",
                        file_path.name,
                        len(df),
                    )

                    for _, row in df.iterrows():
                        if pd.isna(row.get("sentence")) or str(row["sentence"]).strip() == "":
                            continue
                        cid = f"req_{uuid.uuid4().hex}"
                        rng_l1 = TaiwanBatchManager._rng_for_request_id(self.hard_pool_seed, cid + ":L1")
                        rng_l2 = TaiwanBatchManager._rng_for_request_id(self.hard_pool_seed, cid + ":L2")
                        sys_l1 = self._build_l1_system_prompt(rng_l1)
                        sys_l2 = self._build_l2_system_prompt(rng_l2)
                        user_plain = self._build_user_text(row)

                        body_l1 = {
                            "custom_id": cid,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.model,
                                "messages": [
                                    {"role": "system", "content": sys_l1},
                                    {"role": "user", "content": user_plain},
                                ],
                                "temperature": 0,
                                "response_format": {"type": "json_object"},
                            },
                        }
                        body_l2 = {
                            "custom_id": cid,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": self.model,
                                "messages": [
                                    {"role": "system", "content": sys_l2},
                                    {"role": "user", "content": user_plain},
                                ],
                                "temperature": 0.2,
                                "response_format": {"type": "json_object"},
                            },
                        }
                        f1.write(json.dumps(body_l1, ensure_ascii=False) + "\n")
                        f2.write(json.dumps(body_l2, ensure_ascii=False) + "\n")
                        row_data = row.to_dict()
                        row_data["custom_id"] = cid
                        row_data["source_file"] = file_path.name
                        if input_path.is_dir():
                            try:
                                row_data["source_relpath"] = str(
                                    file_path.relative_to(input_path)
                                ).replace("\\", "/")
                            except ValueError:
                                row_data["source_relpath"] = file_path.name
                        else:
                            row_data["source_relpath"] = file_path.name
                        ref_rows.append(row_data)
                        count += 1
                        if count % 2000 == 0:
                            logger.info(
                                "prepare_parallel_l1_l2: built %s paired JSONL lines so far (%s)",
                                count,
                                file_path.name,
                            )
                except Exception as e:
                    logger.error("read %s: %s", file_path.name, e)

        if ref_rows:
            pd.DataFrame(ref_rows).to_csv(reference_csv, index=False, encoding="utf-8-sig")
            logger.info(
                "Parallel L1+L2: %s paired requests -> %s + %s",
                count,
                l1_jsonl,
                l2_jsonl,
            )
            return count
        logger.warning("Parallel: no rows")
        return 0

    def _split_jsonl(self, jsonl_path: str, lines_per_chunk: int = 50) -> List[str]:
        p = Path(jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        total = len(lines)
        if total == 0:
            raise ValueError(f"{jsonl_path} is empty — no batch requests to submit.")

        split_files = []
        for i in range(0, total, lines_per_chunk):
            chunk = lines[i : i + lines_per_chunk]
            part_num = (i // lines_per_chunk) + 1
            chunk_path = p.parent / f"{p.stem}_part{part_num:03d}.jsonl"
            with open(chunk_path, "w", encoding="utf-8") as f_out:
                f_out.writelines(chunk)
            split_files.append(str(chunk_path))
        logger.info("split %s lines -> %s files", total, len(split_files))
        return split_files

    def _poll_batch_after_create(self, batch_id: str, max_wait_sec: int = 180) -> str:
        """
        batches.create() often returns HTTP 200 while validation/enqueue fails asynchronously.
        Poll until we know the job left the pre-flight queue or hit failed.

        Returns:
            "ok" — safe to record batch_id (in_progress / finalizing / completed, or timed out validating).
            "retry_enqueued" — failed due to org enqueued-token cap; caller should sleep and resubmit chunk.
            "fatal" — failed for other reasons, or cancelled/expired.
        """
        interval = 5
        deadline = time.monotonic() + max_wait_sec
        while time.monotonic() < deadline:
            b = self.client.batches.retrieve(batch_id)
            st = (b.status or "").lower()
            if st == "failed":
                err_parts: List[str] = []
                if getattr(b, "errors", None) and getattr(b.errors, "data", None):
                    for err in b.errors.data:
                        err_parts.append(str(getattr(err, "message", "") or ""))
                err_blob = " ".join(err_parts).lower()
                if "enqueued" in err_blob and ("limit" in err_blob or "token" in err_blob):
                    logger.warning(
                        "batch %s failed during validation (enqueued token cap): %s",
                        batch_id,
                        (err_blob[:400] if err_blob else st),
                    )
                    return "retry_enqueued"
                logger.error("batch %s failed: %s", batch_id, err_blob[:500] if err_blob else st)
                return "fatal"
            if st in ("in_progress", "finalizing", "completed"):
                return "ok"
            if st in ("cancelled", "expired"):
                logger.error("batch %s terminal status=%s", batch_id, st)
                return "fatal"
            time.sleep(interval)
        logger.warning(
            "batch %s still not in_progress after %ss (last seen may be validating); "
            "recording id — verify on dashboard if it later fails",
            batch_id,
            max_wait_sec,
        )
        return "ok"

    def _wait_for_batch_slots(
        self,
        active_ids: List[str],
        max_concurrent: int,
        poll_interval: int = 120,
    ) -> List[str]:
        """Block until len(still-active) < max_concurrent. Returns list of still-active IDs."""
        while True:
            still_active = []
            for bid in active_ids:
                try:
                    b = self.client.batches.retrieve(bid)
                    if b.status in ("validating", "in_progress", "finalizing"):
                        still_active.append(bid)
                except Exception as e:
                    logger.warning("poll %s failed: %s — treating as active", bid, e)
                    still_active.append(bid)
            if len(still_active) < max_concurrent:
                return still_active
            logger.info(
                "Waiting for batch slots: %d active >= %d max_concurrent, sleeping %ds",
                len(still_active), max_concurrent, poll_interval,
            )
            time.sleep(poll_interval)

    def submit_files(
        self,
        jsonl_path: str,
        phase_desc: str = "Taiwan framing",
        lines_per_chunk: int = 50,
        max_concurrent: int = 3,
    ) -> List[str]:
        file_path = Path(jsonl_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{jsonl_path} does not exist.")

        files_to_submit = self._split_jsonl(jsonl_path, lines_per_chunk=lines_per_chunk)
        batch_ids = []
        n = len(files_to_submit)
        if n > 200:
            logger.warning(
                "Many batch parts (%s). Org-wide enqueued-token caps (e.g. gpt-5.1 ~1.35M) can fail mid-submit; "
                "use --jsonl-chunk-lines to reduce part count, or pause until dashboard batches finish.",
                n,
            )

        abort_submit = False
        active_ids: List[str] = []
        for i, f_path in enumerate(files_to_submit):
            if abort_submit:
                break
            # --- throttle: wait until we have a free slot ---
            active_ids = self._wait_for_batch_slots(active_ids, max_concurrent)
            while True:
                try:
                    logger.info("[%s/%s] upload %s", i + 1, n, os.path.basename(f_path))
                    with open(f_path, "rb") as f:
                        file_obj = self.client.files.create(file=f, purpose="batch")
                    batch_job = self.client.batches.create(
                        input_file_id=file_obj.id,
                        endpoint="/v1/chat/completions",
                        completion_window="24h",
                        metadata={"description": f"{phase_desc} part {i+1}"},
                    )
                    outcome = self._poll_batch_after_create(batch_job.id)
                    if outcome == "retry_enqueued":
                        logger.warning(
                            "Enqueued-token queue full; sleeping 10m then retrying same part %s/%s",
                            i + 1,
                            n,
                        )
                        time.sleep(600)
                        continue
                    if outcome == "fatal":
                        abort_submit = True
                        break
                    logger.info("batch id %s", batch_job.id)
                    batch_ids.append(batch_job.id)
                    active_ids.append(batch_job.id)
                    time.sleep(2)
                    break
                except Exception as e:
                    err_msg = str(e).lower()
                    enqueued_cap = (
                        "enqueued token limit" in err_msg
                        or ("enqueued" in err_msg and "token" in err_msg and "limit" in err_msg)
                    )
                    if enqueued_cap:
                        logger.warning(
                            "Hit OpenAI enqueued-token limit (org quota for pending batch work). "
                            "Sleeping 10m; wait for in_progress batches to complete, or raise limits. %s",
                            e,
                        )
                        time.sleep(600)
                        continue
                    elif "rate limit" in err_msg:
                        logger.warning("rate limit, sleep 5m — %s", e)
                        time.sleep(300)
                        continue
                    else:
                        logger.error("submit %s: %s", f_path, e)
                        abort_submit = True
                        break
        return batch_ids

    def retrieve_results(self, batch_ids: List[str]) -> dict:
        all_results = {}
        for b_id in batch_ids:
            try:
                batch = self.client.batches.retrieve(b_id)

                if batch.status == "completed":
                    if not batch.output_file_id:
                        logger.warning("batch %s no output_file_id", b_id)
                        continue
                    content = self.client.files.content(batch.output_file_id).text
                    for line in content.splitlines():
                        try:
                            item = json.loads(line)
                            cid = item["custom_id"]
                            resp = (
                                item.get("response", {})
                                .get("body", {})
                                .get("choices", [{}])[0]
                                .get("message", {})
                                .get("content")
                            )
                            if resp:
                                all_results[cid] = json.loads(resp)
                            else:
                                all_results[cid] = {"error": "Empty content"}
                        except json.JSONDecodeError:
                            all_results[item.get("custom_id", "unknown")] = {"error": "JSON Parse Error"}
                        except Exception as e:
                            all_results[item.get("custom_id", "unknown")] = {"error": str(e)}

                elif batch.status in ("failed", "expired", "cancelled"):
                    logger.error("batch %s %s", b_id, batch.status)
                    if hasattr(batch, "errors") and batch.errors and hasattr(batch.errors, "data"):
                        for err in batch.errors.data:
                            logger.error(
                                "err code=%s msg=%s line=%s",
                                getattr(err, "code", ""),
                                getattr(err, "message", ""),
                                getattr(err, "line", ""),
                            )
                    if batch.error_file_id:
                        try:
                            err_text = self.client.files.content(batch.error_file_id).text
                            logger.error("error file:\n%s", err_text[:2000])
                        except Exception:
                            pass
                else:
                    logger.info("batch %s status=%s", b_id, batch.status)

            except Exception as e:
                logger.error("retrieve %s: %s", b_id, e)
        return all_results

    def process_l1_results(self, batch_ids: List[str], ref_csv: str, output_csv: str):
        results = self.retrieve_results(batch_ids)
        if not results:
            logger.error("no L1 results")
            return

        df = pd.read_csv(ref_csv)
        valid_l1_set = (
            set(self.l1_guide["label_id"].astype(str).tolist()) if not self.l1_guide.empty else set()
        )

        def parse_l1(cid):
            if cid not in results:
                return None, None, "missing response"
            res = results.get(cid, {})
            if not isinstance(res, dict) or res.get("error"):
                return None, None, res.get("error", "error") if isinstance(res, dict) else "bad response"
            l1_pred = self._normalize_l1_code(str(res.get("L1", "")).strip())
            if valid_l1_set and l1_pred not in valid_l1_set:
                if l1_pred:
                    logger.warning("[%s] invalid L1 dropped %s", cid, l1_pred)
                return None, res.get("L1_cot", ""), "invalid L1 code"
            return (l1_pred or None), res.get("L1_cot", ""), ""

        parsed = df["custom_id"].apply(
            lambda cid: pd.Series(parse_l1(cid), index=["L1_label", "L1_reasoning", "L1_error"])
        )
        for col in parsed.columns:
            df[col] = parsed[col]

        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        l1_n = len(df[df["L1_label"].notna() & (df["L1_label"] != "")])
        logger.info("L1 out %s rows=%s l1=%s", output_csv, len(df), l1_n)

        err_s = df["L1_error"].fillna("").astype(str).str.strip()
        has_err = err_s != ""
        l1_err_rows = int(has_err.sum())
        dist_raw = err_s[has_err].value_counts().to_dict() if l1_err_rows else {}
        err_dist = {str(k): int(v) for k, v in dist_raw.items()}
        logger.info(
            "L1 quality: L1_error non-empty rows=%s distribution=%s",
            l1_err_rows,
            err_dist,
        )
        run_dir = Path(output_csv).parent
        _merge_annotation_quality_report(
            run_dir,
            "l1",
            {
                "rows": len(df),
                "l1_valid_label_rows": int(l1_n),
                "l1_error_non_empty_rows": l1_err_rows,
                "l1_error_type_distribution": err_dist,
            },
        )

    def process_l2_results(self, batch_ids: List[str], ref_csv: str, final_csv: str):
        results = self.retrieve_results(batch_ids)
        if not results:
            logger.error("no L2 results")
            return

        df = pd.read_csv(ref_csv)
        valid_l2_set = (
            set(self.l2_guide["label_id"].astype(str).tolist()) if not self.l2_guide.empty else set()
        )

        def parse_l2(cid):
            res = results.get(cid, {})
            if not isinstance(res, dict):
                return "", "", "non-dict response", 0
            if res.get("error"):
                return "", "", res.get("error", "unknown"), 0
            l2_pred = res.get("L2", [])
            if isinstance(l2_pred, list):
                clean = [str(x).strip() for x in l2_pred if str(x).strip()]
                dropped_n = 0
                if valid_l2_set:
                    invalid = [x for x in clean if x not in valid_l2_set]
                    dropped_n = len(invalid)
                    if invalid:
                        logger.warning("[%s] invalid L2 dropped %s", cid, invalid)
                    clean = [x for x in clean if x in valid_l2_set]
                return "|".join(clean), res.get("L2_cot", ""), "", dropped_n
            return "", res.get("L2_cot", ""), "L2 field not a list", 0

        parsed = df["custom_id"].apply(
            lambda cid: pd.Series(
                parse_l2(cid),
                index=["L2_labels", "L2_reasoning", "L2_error", "_l2_invalid_label_drops"],
            )
        )
        for col in parsed.columns:
            df[col] = parsed[col]

        lab = df["L2_labels"].fillna("").astype(str).str.strip()
        l2_empty_rows = int((lab == "").sum())
        rows_invalid_dropped = int((df["_l2_invalid_label_drops"] > 0).sum())
        logger.info(
            "L2 quality: L2_labels empty rows=%s rows_with_invalid_L2_dropped=%s",
            l2_empty_rows,
            rows_invalid_dropped,
        )

        Path(final_csv).parent.mkdir(parents=True, exist_ok=True)
        df_out = df.drop(columns=["_l2_invalid_label_drops"])
        df_out.to_csv(final_csv, index=False, encoding="utf-8-sig")
        l2_pos = len(df_out[df_out["L2_labels"].notna() & (df_out["L2_labels"] != "")])
        logger.info("final %s rows=%s l2_pos=%s", final_csv, len(df_out), l2_pos)
        run_dir = Path(final_csv).parent
        _merge_annotation_quality_report(
            run_dir,
            "l2",
            {
                "rows": len(df_out),
                "l2_non_empty_label_rows": int(l2_pos),
                "l2_empty_label_rows": l2_empty_rows,
                "l2_rows_with_invalid_label_dropped": rows_invalid_dropped,
                "l2_invalid_label_drop_count_total": int(df["_l2_invalid_label_drops"].sum()),
            },
        )


VALIDATION_OUTPUT_BASE = "03_outputs/01_results_labelings/03_validation_output"
CORPUS_RESULTS_BASE = "03_outputs/01_results_labelings/01_results_datasets"
_LATEST_RUN_FILE = "latest_run.txt"
_LATEST_CORPUS_RUN_FILE = "latest_corpus_run.txt"
# 16_corpus_merged_deduped_async.py：仅跑 merged_deduped.csv，与 05 的 latest 指针分离
_LATEST_MERGED_DEDUPED_RUN_FILE = "latest_merged_deduped_run.txt"


def create_run_dir(base_dir: Path) -> Path:
    from datetime import datetime

    output_base = base_dir / VALIDATION_OUTPUT_BASE
    output_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = output_base / f"Run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (output_base / _LATEST_RUN_FILE).write_text(str(run_dir), encoding="utf-8")
    logger.info("run_dir %s", run_dir)
    return run_dir


def create_corpus_run_dir(base_dir: Path) -> Path:
    """New timestamped folder under 01_results_datasets; updates latest_corpus_run.txt."""
    from datetime import datetime

    output_base = base_dir / CORPUS_RESULTS_BASE
    output_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"Run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (output_base / _LATEST_CORPUS_RUN_FILE).write_text(str(run_dir.resolve()), encoding="utf-8")
    logger.info("corpus run_dir %s", run_dir)
    return run_dir


def resolve_corpus_run_dir(base_dir: Path) -> Path:
    output_base = base_dir / CORPUS_RESULTS_BASE
    pointer = output_base / _LATEST_CORPUS_RUN_FILE
    if not pointer.exists():
        raise FileNotFoundError(
            f"{pointer} missing; run 02_src/05_corpus_parallel_async.py submit first."
        )
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"corpus run_dir gone: {run_dir}")
    logger.info("corpus run_dir %s", run_dir)
    return run_dir


def create_merged_deduped_corpus_run_dir(base_dir: Path) -> Path:
    """与 create_corpus_run_dir 相同目录结构，但写入 latest_merged_deduped_run.txt。"""
    from datetime import datetime

    output_base = base_dir / CORPUS_RESULTS_BASE
    output_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_base / f"Run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (output_base / _LATEST_MERGED_DEDUPED_RUN_FILE).write_text(
        str(run_dir.resolve()), encoding="utf-8"
    )
    logger.info("merged_deduped corpus run_dir %s", run_dir)
    return run_dir


def resolve_merged_deduped_corpus_run_dir(base_dir: Path) -> Path:
    output_base = base_dir / CORPUS_RESULTS_BASE
    pointer = output_base / _LATEST_MERGED_DEDUPED_RUN_FILE
    if not pointer.exists():
        raise FileNotFoundError(
            f"{pointer} missing; run 02_src/16_corpus_merged_deduped_async.py submit first."
        )
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"merged_deduped corpus run_dir gone: {run_dir}")
    logger.info("merged_deduped corpus run_dir %s", run_dir)
    return run_dir


def resolve_run_dir(base_dir: Path) -> Path:
    output_base = base_dir / VALIDATION_OUTPUT_BASE
    pointer = output_base / _LATEST_RUN_FILE
    if not pointer.exists():
        raise FileNotFoundError(
            f"{pointer} missing; run submit first "
            f"(02_src/05_validation_l1_async.py submit or 02_src/05_validation_parallel_async.py submit)."
        )
    run_dir = Path(pointer.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir gone: {run_dir}")
    logger.info("run_dir %s", run_dir)
    return run_dir
