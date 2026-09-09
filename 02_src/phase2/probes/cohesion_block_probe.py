"""凝聚簇 (L2-04/06/07/08) Layer A/B 前置探针.

**主规格：L1 取 GPT 标注，L2 取 DeepSeek 标注。** GPT 的 L2 仅作稳健性对照。
理由：`07_l207_arbitration` 人工金标准下 L2-07 上 GPT κ=-0.20 / DS κ=0.44；
而 DS 那次是纯 L2 重标（labeled_units 的 L1 列为空），L1 只能取 GPT，
且 gold 148 句上 GPT L1 (73.6%) 优于 DS L1 (68.2%)。

对应 04_documents/l1_l2_combination_plan.md：
  §2 凝聚 = L2-04 集體敘事再造 / L2-06 民族自豪感 / L2-07 凝聚與願景 / L2-08 民主價值
  §4 Layer B = 这四码之间的六条边

分析层次
  1. 分布      —— 四码边际、纯度、句级联合模式
  2. Layer A   —— L1 x L2 同句共标：计数/行%/列%/lift/调整后标准化残差 + cluster bootstrap
  3. Layer B   —— 六条边 NPMI + doc cluster bootstrap；窗口大小敏感性；文档集中度
  4. 分解      —— 同句捆绑 vs 句间序列共现（逐标签文档内置换零模型）
  5. 混杂      —— 句长、boilerplate 抓取残留
  6. 稳健性    —— DS(主) vs GPT(对照) 两套 L2
  7. 可行性    —— L1 x camp 分层内的最小共现量

零模型说明：在分层 S 内对每个标签的存在向量独立置换，保留每层每码的计数，
摧毁层内的句内捆绑与句间邻接。当层内 n<2 或某码在层内全 0/全 1 时置换等于
恒等，该层观测值必须同时计入 null（否则细分层下 null 被系统性压低）。

用法: python 02_src/phase2/probes/cohesion_block_probe.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import boilerplate  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "03_outputs/01_results_labelings"
OUT = BASE / "08_cohesion_probe"
OUT.mkdir(parents=True, exist_ok=True)

COH = ["L2-04", "L2-06", "L2-07", "L2-08"]
BND = ["L2-01", "L2-02", "L2-03", "L2-05"]
ALL8 = BND + COH
PAIRS = list(itertools.combinations(COH, 2))
BPAIRS = list(itertools.combinations(BND, 2))
PAIRS28 = list(itertools.combinations(ALL8, 2))

# L2 标注源。DS = 主规格；GPT = 对照。L1 恒取 GPT。
L2_DIR = {"DS": BASE / "06_phase2_coarticulation_deepseek",
          "GPT": BASE / "06_phase2_coarticulation"}
L1_DIR = BASE / "06_phase2_coarticulation"
PRIMARY = "DS"

R = 600            # 置换次数
NBOOT = 1000       # bootstrap 次数
MAXLAG = 2         # 与 n=3 窗口的可达距离一致
SEED = 20260804
RNG = np.random.default_rng(SEED)

#: 抓取残留判定统一走 utils.boilerplate（p2_00 用的同一套模式）。p2_00 已按
#: drop_residue 删掉 drop 层，所以这里 `boiler` 实际只剩 prefix 层（真句 + 抓取前缀），
#: 第 6 节的"去 boiler"控制因此是对前缀污染的敏感性检验，不再是对残留行的删除。

TRIG_07 = ["一起", "讓我們", "让我们", "站出來", "站出来", "共同守護", "共同守护",
           "打勝仗", "打胜仗", "必須贏", "必须赢", "加油", "為台灣", "为台湾",
           "目標是", "目标是", "打造", "團結", "团结", "支持", "投票", "選擇",
           "选择", "拜託", "拜托", "努力", "攜手", "携手", "共同"]
TRIG_06 = ["第一", "最高", "紀錄", "纪录", "成長", "成长", "排名", "全球", "世界",
           "驕傲", "骄傲", "光榮", "光荣", "肯定", "成就", "奇蹟", "奇迹", "%", "％"]


def _as_set(x):
    return set(map(str, x)) if x is not None and len(x) else set()


_L1 = None


#: 合并键。多党句在 p2_00 会按 camp 展开，故 (doc_id, sent_idx) 非主键，须带 camp。
KEY = ["doc_id", "sent_idx", "camp"]


def gpt_l1() -> pd.DataFrame:
    """GPT 的 L1（DS 那次重标未产出 L1，其 labeled_units 的 L1 列为空）。"""
    global _L1
    if _L1 is None:
        d = pd.read_parquet(L1_DIR / "labeled_units.parquet")
        _L1 = d[KEY + ["L1"]].sort_values(KEY).reset_index(drop=True)
        assert not _L1.duplicated(KEY).any(), "GPT L1 在 (doc_id, sent_idx, camp) 上不唯一"
    return _L1


def load_sentences(l2_src: str) -> pd.DataFrame:
    """句级表：L2 来自 l2_src，L1 恒来自 GPT。"""
    d = (pd.read_parquet(L2_DIR[l2_src] / "labeled_units.parquet")
         .sort_values(KEY).reset_index(drop=True))
    n0 = len(d)
    d = d.drop(columns=["L1"]).merge(gpt_l1(), on=KEY, how="left", validate="one_to_one")
    assert len(d) == n0 and d["L1"].notna().all(), "GPT L1 未能完整合并"
    d["l2"] = d["l2_set"].map(_as_set)
    for lab in ALL8:
        d[lab] = d["l2"].map(lambda s, l=lab: l in s)
    d["n_l2"] = d["l2"].map(len)
    d["n_coh"] = d[COH].sum(axis=1)
    d["n_bnd"] = d[BND].sum(axis=1)
    d["slen"] = d["sent_text"].map(lambda t: len(str(t)))
    if "boiler_tier" in d.columns:            # p2_00 已标注
        d["boiler"] = d["boiler_tier"].fillna("") != ""
    else:                                     # 旧 artifacts：现场判定
        d["boiler"] = boilerplate.annotate(d["sent_text"])["boiler_tier"] != ""
    d["len_bin"] = pd.qcut(d["slen"], 4, labels=False, duplicates="drop")
    d["coh_any"] = d[COH].any(axis=1)
    d["bnd_any"] = d[BND].any(axis=1)
    return d


def load_windows(l2_src: str, n: str) -> pd.DataFrame:
    w = pd.read_parquet(L2_DIR[l2_src] / f"windows/windows_{n}.parquet")
    w["l2"] = w["l2_set"].map(_as_set)
    return w


# ============================================================== NPMI 机器
def _npmi_batch(n_, m_, p_, pairs, li):
    """NPMI = PMI / -log P(A,B)  (Bouma 2009)。批量：n_ (B,), m_ (B,L), p_ (B,P)."""
    out = np.full((n_.shape[0], len(pairs)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (a, b) in enumerate(pairs):
            pab, pa, pb = p_[:, k] / n_, m_[:, li[a]] / n_, m_[:, li[b]] / n_
            ok = (pab > 0) & (pa > 0) & (pb > 0)
            lp = np.log(np.where(ok, pab, 1.0))
            val = (lp - np.log(np.where(ok, pa, 1.0))
                   - np.log(np.where(ok, pb, 1.0))) / (-lp)
            out[:, k] = np.where(ok & (lp < 0), val, np.nan)
    return out


def npmi_edges(windows: pd.DataFrame, pairs=PAIRS, labels=ALL8, n_boot=NBOOT):
    """点估计 + 按 doc_id 的 cluster bootstrap（与 p2_03 同一重抽样单位）。"""
    li = {l: j for j, l in enumerate(labels)}
    W = np.stack([windows["l2"].map(lambda s, l=l: l in s).values for l in labels], 1)
    docs = pd.factorize(windows["doc_id"].values)[0]
    D = docs.max() + 1
    nw = np.bincount(docs, minlength=D).astype(float)
    mg = np.stack([np.bincount(docs, weights=W[:, j].astype(float), minlength=D)
                   for j in range(len(labels))], 1)
    pc = np.stack([np.bincount(docs, weights=(W[:, li[a]] & W[:, li[b]]).astype(float),
                               minlength=D) for a, b in pairs], 1)

    pt = _npmi_batch(nw.sum(keepdims=True), mg.sum(0)[None], pc.sum(0)[None], pairs, li)[0]
    idx = RNG.integers(0, D, size=(n_boot, D))
    cnt = np.zeros((n_boot, D))
    for b in range(n_boot):
        cnt[b] = np.bincount(idx[b], minlength=D)
    bt = _npmi_batch(cnt @ nw, cnt @ mg, cnt @ pc, pairs, li)

    rows = []
    for k, (a, b) in enumerate(pairs):
        c = bt[:, k][~np.isnan(bt[:, k])]
        lo, hi = np.percentile(c, [2.5, 97.5]) if len(c) > 20 else (np.nan, np.nan)
        v = pc[:, k]
        share = np.sort(v)[::-1] / v.sum() if v.sum() else np.array([np.nan])
        rows.append({
            "pair": f"{a[3:]}-{b[3:]}", "n_ab": int(v.sum()),
            "n_a": int(mg[:, li[a]].sum()), "n_b": int(mg[:, li[b]].sum()),
            "n_windows": int(nw.sum()), "npmi": pt[k], "lo": lo, "hi": hi,
            "ci_width": hi - lo, "sig": bool(lo > 0 or hi < 0),
            "n_docs_with_pair": int((v > 0).sum()),
            "top1_doc_share": share[0], "top5_doc_share": share[:5].sum(),
            "eff_n_docs": 1 / ((share ** 2).sum()) if v.sum() else np.nan,
        })
    return pd.DataFrame(rows)


# ============================================ Layer A: L1 x L2 同句共标
def layer_a(df: pd.DataFrame, labels=COH, n_boot=NBOOT, by_camp=False):
    """plan §3 指定的统计量：计数 / 行% / 列% / lift / 调整后标准化残差 + cluster bootstrap.

    L1 单标（互斥），L2 多标，故每个 L2 码对应一张 L1 x {带/不带} 的 7x2 表，
    调整后标准化残差取 "带" 那一列：
        ASR = (O-E) / sqrt(E * (1 - n_row/N) * (1 - n_col/N))
    区间估计按 doc_id 重抽样（与 p2_03 同一单位），报 lift 的 95% 百分位 CI。
    """
    groups = [("ALL", df)] + ([(c, g) for c, g in df.groupby("camp")] if by_camp else [])
    out = []
    for camp, d in groups:
        N = len(d)
        docs = pd.factorize(d["doc_id"].values)[0]
        D = docs.max() + 1
        l1s = sorted(d["L1"].unique())
        idx = RNG.integers(0, D, size=(n_boot, D))
        cnt = np.zeros((n_boot, D))
        for b in range(n_boot):
            cnt[b] = np.bincount(idx[b], minlength=D)
        # 每文档：行数、各 L1 计数、各 (L1,L2) 计数、各 L2 计数
        nd = np.bincount(docs, minlength=D).astype(float)
        l1m = {l1: np.bincount(docs, weights=(d["L1"] == l1).values.astype(float),
                               minlength=D) for l1 in l1s}
        for lab in labels:
            lv = d[lab].values.astype(float)
            l2m = np.bincount(docs, weights=lv, minlength=D)
            for l1 in l1s:
                cell = np.bincount(docs, weights=((d["L1"] == l1).values & d[lab].values)
                                   .astype(float), minlength=D)
                O, n_row, n_col = cell.sum(), l1m[l1].sum(), l2m.sum()
                E = n_row * n_col / N
                den = E * (1 - n_row / N) * (1 - n_col / N)
                asr = (O - E) / np.sqrt(den) if den > 0 else np.nan
                # bootstrap lift
                bN, bO = cnt @ nd, cnt @ cell
                bR, bC = cnt @ l1m[l1], cnt @ l2m
                with np.errstate(divide="ignore", invalid="ignore"):
                    bl = (bO * bN) / (bR * bC)
                bl = bl[np.isfinite(bl)]
                lo, hi = np.percentile(bl, [2.5, 97.5]) if len(bl) > 20 else (np.nan, np.nan)
                out.append({
                    "camp": camp, "L1": l1, "l2": lab, "n": int(O),
                    "row_pct": O / n_row if n_row else np.nan,   # P(L2 | L1)
                    "col_pct": O / n_col if n_col else np.nan,   # P(L1 | L2)
                    "expected": E, "lift": (O * N) / (n_row * n_col) if n_row and n_col else np.nan,
                    "lift_lo": lo, "lift_hi": hi,
                    "lift_sig": bool(np.isfinite(lo) and np.isfinite(hi)
                                     and (lo > 1 or hi < 1)),
                    "asr": asr, "n_L1": int(n_row), "n_L2": int(n_col), "N": N,
                })
    return pd.DataFrame(out)


# =========================================================== 置换零模型
def perm_excess(df, strat, pairs=PAIRS, labels=COH, mask=None, tag="", lags=(0,)):
    """同句 (lag=0) 与句间 (lag>=1) 共现的观测/零模型比。

    n<2 或某码在层内不可移动时置换=恒等，观测值同时计入 null。
    """
    li = {l: j for j, l in enumerate(labels)}
    sub = df if mask is None else df[mask]
    nl = len(lags)
    obs = np.zeros((nl, len(pairs)))
    null = np.zeros((R, nl, len(pairs)))
    dirn = np.zeros((len(pairs), 2))

    def _co(A, B, n, lag):
        if lag == 0:
            return (A & B).sum(axis=-1)
        if n <= lag:
            return np.zeros(A.shape[:-1])
        return (A[..., :-lag] & B[..., lag:]).sum(axis=-1) + \
               (B[..., :-lag] & A[..., lag:]).sum(axis=-1)

    for _, g in sub.groupby(strat, observed=True, sort=False):
        M = g[labels].values.astype(bool)
        n = M.shape[0]
        co = np.zeros((nl, len(pairs)))
        for k, (a, b) in enumerate(pairs):
            A, B = M[:, li[a]], M[:, li[b]]
            for t, lag in enumerate(lags):
                co[t, k] = _co(A, B, n, lag)
            for lag in range(1, MAXLAG + 1):
                if n > lag:
                    dirn[k, 0] += (A[:-lag] & B[lag:]).sum()
                    dirn[k, 1] += (B[:-lag] & A[lag:]).sum()
        obs += co
        if n < 2:
            null += co
            continue
        ks = M.sum(axis=0)
        perm, fixed = {}, np.zeros(len(labels), bool)
        for j in range(len(labels)):
            kj = int(ks[j])
            if kj == 0 or kj == n:
                fixed[j] = True
                continue
            order = RNG.random((R, n)).argsort(axis=1)
            m = np.zeros((R, n), dtype=bool)
            np.put_along_axis(m, order[:, :kj], True, axis=1)
            perm[j] = m
        for k, (a, b) in enumerate(pairs):
            ja, jb = li[a], li[b]
            if ks[ja] == 0 or ks[jb] == 0:
                continue
            if fixed[ja] and fixed[jb]:
                null[:, :, k] += co[:, k]
                continue
            A = np.repeat(M[:, ja][None], R, 0) if fixed[ja] else perm[ja]
            B = np.repeat(M[:, jb][None], R, 0) if fixed[jb] else perm[jb]
            for t, lag in enumerate(lags):
                null[:, t, k] += _co(A, B, n, lag)

    rows = []
    for t, lag in enumerate(lags):
        for k, (a, b) in enumerate(pairs):
            m_, s_ = null[:, t, k].mean(), null[:, t, k].std(ddof=1)
            f, r = dirn[k]
            rows.append({"spec": tag, "lag": "same" if lag == 0 else f"lag{lag}",
                         "pair": f"{a[3:]}-{b[3:]}", "obs": int(obs[t, k]), "null": m_,
                         "ratio": obs[t, k] / m_ if m_ else np.nan,
                         "z": (obs[t, k] - m_) / s_ if s_ > 0 else np.nan,
                         "n_a_first": int(f), "n_b_first": int(r),
                         "dir_skew": (f - r) / (f + r) if (f + r) else np.nan})
    return pd.DataFrame(rows)


# ================================================ 句对 skip-gram NPMI
def skipgram_npmi(df, maxlag=MAXLAG, n_boot=NBOOT):
    """单位 = 同文档内距离 1..maxlag 的有序句对；同句共标结构上无法贡献。"""
    li = {l: j for j, l in enumerate(COH)}
    NP, SL, CO = [], [], []
    for _, g in df.groupby("doc_id", sort=False):
        M = g[COH].values.astype(bool)
        n = M.shape[0]
        npair, slot, co = 0, np.zeros(len(COH)), np.zeros(len(PAIRS))
        for lag in range(1, maxlag + 1):
            if n <= lag:
                break
            npair += n - lag
            slot += M[:-lag].sum(0) + M[lag:].sum(0)
            for k, (a, b) in enumerate(PAIRS):
                co[k] += (M[:-lag, li[a]] & M[lag:, li[b]]).sum()
                co[k] += (M[:-lag, li[b]] & M[lag:, li[a]]).sum()
        if npair:
            NP.append(npair); SL.append(slot); CO.append(co)
    NP, SL, CO = np.array(NP, float), np.array(SL), np.array(CO)
    D = len(NP)

    def _c(n_, s_, c_):
        o = np.full((n_.shape[0], len(PAIRS)), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            for k, (a, b) in enumerate(PAIRS):
                pab = c_[:, k] / n_
                pa, pb = s_[:, li[a]] / (2 * n_), s_[:, li[b]] / (2 * n_)
                ok = (pab > 0) & (pa > 0) & (pb > 0)
                lp = np.log(np.where(ok, pab, 1.0))
                o[:, k] = np.where(ok & (lp < 0),
                                   (lp - np.log(np.where(ok, pa, 1.0))
                                    - np.log(np.where(ok, pb, 1.0))) / (-lp), np.nan)
        return o
    pt = _c(NP.sum(keepdims=True), SL.sum(0)[None], CO.sum(0)[None])[0]
    idx = RNG.integers(0, D, size=(n_boot, D))
    cnt = np.zeros((n_boot, D))
    for b in range(n_boot):
        cnt[b] = np.bincount(idx[b], minlength=D)
    bt = _c(cnt @ NP, cnt @ SL, cnt @ CO)
    rows = []
    for k, (a, b) in enumerate(PAIRS):
        c = bt[:, k][~np.isnan(bt[:, k])]
        lo, hi = np.percentile(c, [2.5, 97.5]) if len(c) > 20 else (np.nan, np.nan)
        rows.append({"pair": f"{a[3:]}-{b[3:]}", "n_sentpairs": int(NP.sum()),
                     "n_co": int(CO[:, k].sum()), "npmi_skip": pt[k],
                     "lo": lo, "hi": hi, "sig": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


# ====================================================================== main
def main():
    S = {s: load_sentences(s) for s in L2_DIR}
    P = S[PRIMARY]
    hdr = lambda t: print("\n" + "=" * 88 + f"\n{t}\n" + "=" * 88)
    print(f"主规格：L1 = GPT，L2 = {PRIMARY}    对照：L2 = "
          f"{[s for s in L2_DIR if s != PRIMARY][0]}")

    # ---------- 1. 分布
    hdr("1. 四码边际与纯度（L2=%s，N=%d 句）" % (PRIMARY, len(P)))
    marg = pd.DataFrame([{
        "l2": l, "block": "凝聚" if l in COH else "边界", "n": int(P[l].sum()),
        "pct": 100 * P[l].mean()} for l in ALL8]).sort_values("n", ascending=False)
    print(marg.round(3).to_string(index=False))
    print(f"\n空标率 = {P.n_l2.eq(0).mean():.4f}   平均 L2 码数 = {P.n_l2.mean():.3f}")

    purity = pd.DataFrame([{
        "l2": l, "n": int(P[l].sum()),
        "alone_in_coh": P.loc[P[l], "n_coh"].eq(1).mean(),
        "alone_overall": P.loc[P[l], "n_l2"].eq(1).mean(),
        "mean_n_l2": P.loc[P[l], "n_l2"].mean(),
        "with_any_bnd": P.loc[P[l], "n_bnd"].gt(0).mean()} for l in COH])
    print("\n-- 纯度 --"); print(purity.round(3).to_string(index=False))

    pat = P[P.n_coh > 0].groupby(COH).size().reset_index(name="n")
    pat["pattern"] = pat.apply(lambda r: "+".join(l[3:] for l in COH if r[l]), axis=1)
    pat = pat[["pattern", "n"]].sort_values("n", ascending=False)
    pat["pct"] = 100 * pat.n / pat.n.sum()
    print("\n-- 句级联合模式 --"); print(pat.head(10).round(2).to_string(index=False))
    marg.to_csv(OUT / "01_marginals.csv", index=False)
    purity.to_csv(OUT / "02_purity.csv", index=False)
    pat.to_csv(OUT / "03_joint_patterns.csv", index=False)

    # ---------- 2. Layer A：L1 x L2
    hdr("2. Layer A — L1(GPT) x L2(%s) 同句共标，凝聚四码" % PRIMARY)
    l1cb = pd.read_csv(ROOT / "01_data/05_labels_guidance/01_annotation_guide_label1_v9.csv")
    l1n = dict(zip(l1cb.label_id.astype(str), l1cb.label_cn.astype(str)))
    la = layer_a(P, by_camp=True)
    la["L1_cn"] = la.L1.map(l1n)
    la.to_csv(OUT / "04_layerA_l1_x_l2.csv", index=False)
    lall = la[la.camp == "ALL"]
    print("-- lift（P(L2|L1) / P(L2)），** = bootstrap 95% CI 不含 1 --")
    piv = lall.pivot(index="L1", columns="l2", values="lift").round(2).astype(str)
    sg = lall.pivot(index="L1", columns="l2", values="lift_sig")
    piv = piv.where(~sg, piv + "**")
    piv.insert(0, "L1_cn", [l1n[i] for i in piv.index])
    piv.insert(1, "n_L1", lall.groupby("L1").n_L1.first())
    print(piv.to_string())
    print("\n-- 调整后标准化残差 --")
    print(lall.pivot(index="L1", columns="l2", values="asr").round(1).to_string())
    print("\n-- 行% = P(L2 | L1) --")
    print(lall.pivot(index="L1", columns="l2", values="row_pct").round(3).to_string())
    print("\n-- 列% = P(L1 | L2) --")
    print(lall.pivot(index="L1", columns="l2", values="col_pct").round(3).to_string())

    # 凝聚 vs 边界在 L1 上的相对落点
    t = P.groupby("L1", observed=True).agg(
        n=("coh_any", "size"), coh_any=("coh_any", "mean"),
        bnd_any=("bnd_any", "mean")).reset_index()
    t["L1_cn"] = t.L1.map(l1n)
    t["coh_over_bnd"] = t.coh_any / t.bnd_any
    print("\n-- 凝聚率 / 边界率 --")
    print(t.sort_values("coh_over_bnd", ascending=False).round(3).to_string(index=False))
    t.to_csv(OUT / "05_l1_block_balance.csv", index=False)

    # ---------- 3. Layer B：六条边
    hdr("3. Layer B — 六条凝聚边（L2=%s, n=3, 池化）+ 文档集中度" % PRIMARY)
    e3 = npmi_edges(load_windows(PRIMARY, "n3"))
    print(e3.round(3).to_string(index=False))
    e3.to_csv(OUT / "06_edges_primary_n3.csv", index=False)

    hdr("4. 窗口大小敏感性（L2=%s，池化）" % PRIMARY)
    sw = []
    for n in ["n1", "n3", "n5", "ndocument"]:
        x = npmi_edges(load_windows(PRIMARY, n), n_boot=400); x["window"] = n; sw.append(x)
    sw = pd.concat(sw, ignore_index=True)
    print(sw.pivot(index="pair", columns="window", values="npmi")
          [["n1", "n3", "n5", "ndocument"]].round(3).to_string())
    sw.to_csv(OUT / "07_window_sweep.csv", index=False)

    # ---------- 5. 同句 vs 句间
    hdr("5. 同句捆绑 vs 句间邻接：逐标签文档内置换 (R=%d)" % R)
    lags = (0, 1, 2)
    dec = pd.concat([perm_excess(S[s], ["doc_id"], tag=s, lags=lags) for s in
                     [PRIMARY] + [x for x in L2_DIR if x != PRIMARY]], ignore_index=True)
    print(dec.pivot_table(index="pair", columns=["spec", "lag"], values="ratio")
          .round(3).to_string())
    print("\n-- z --")
    print(dec.pivot_table(index="pair", columns=["spec", "lag"], values="z")
          .round(1).to_string())
    print("\n-- 方向偏斜（正 = a 先于 b），L2=%s --" % PRIMARY)
    print(dec[(dec.spec == PRIMARY) & (dec.lag == "same")]
          [["pair", "n_a_first", "n_b_first", "dir_skew"]].round(3).to_string(index=False))
    dec.to_csv(OUT / "08_same_vs_adjacent.csv", index=False)

    hdr("6. 同句超额：混杂控制（句长四分位 / 去 boilerplate / L1 内）")
    ctl = pd.concat(
        [perm_excess(S[s], ["doc_id"], tag=f"{s}|doc") for s in L2_DIR] +
        [perm_excess(S[s], ["doc_id", "len_bin"], tag=f"{s}|doc×len") for s in L2_DIR] +
        [perm_excess(S[s], ["doc_id"], mask=~S[s].boiler, tag=f"{s}|doc,去boiler")
         for s in L2_DIR] +
        [perm_excess(S[PRIMARY], ["doc_id", "L1"], tag=f"{PRIMARY}|doc×L1")],
        ignore_index=True)
    print(ctl.pivot(index="pair", columns="spec", values="ratio").round(3).to_string())
    print("\n-- z --")
    print(ctl.pivot(index="pair", columns="spec", values="z").round(1).to_string())
    ctl.to_csv(OUT / "09_confound_control.csv", index=False)

    bctl = pd.concat([perm_excess(S[s], ["doc_id"], BPAIRS, BND, tag=s) for s in L2_DIR],
                     ignore_index=True)
    print("\n-- 对照：边界簇六条边同句超额 ratio --")
    print(bctl.pivot(index="pair", columns="spec", values="ratio").round(3).to_string())
    bctl.to_csv(OUT / "10_boundary_control.csv", index=False)

    hdr("7. 句对 skip-gram NPMI（结构上排除同句共标）")
    sk = pd.concat([skipgram_npmi(S[s]).assign(src=s) for s in L2_DIR], ignore_index=True)
    print(sk.pivot(index="pair", columns="src", values="npmi_skip").round(3).to_string())
    sk.to_csv(OUT / "11_skipgram.csv", index=False)

    # ---------- 8. boilerplate / 词汇
    hdr("8. Boilerplate 抓取残留（语料属性，与 L2 源无关）+ 06/07 词汇边界")
    tx = P.sent_text.astype(str)
    bp = []
    for k, (tier, pat) in list(boilerplate.PATTERNS.items()) + [("正文", ("", None))]:
        m = ~P.boiler if pat is None else tx.str.contains(pat, regex=True)
        bp.append({"pattern": k, "tier": tier, "n": int(m.sum()), "pct": 100 * m.mean(),
                   "mean_len": P[m].slen.mean(), "mean_n_l2": P[m].n_l2.mean(),
                   "p_06and07": (P[m]["L2-06"] & P[m]["L2-07"]).mean()})
    bp = pd.DataFrame(bp)
    print(bp.round(3).to_string(index=False))
    n67 = int((P["L2-06"] & P["L2-07"]).sum())
    n67b = int((P.boiler & P["L2-06"] & P["L2-07"]).sum())
    print(f"\n06&07 同句共标 {n67} 句，其中 boilerplate {n67b} 句 "
          f"({100*n67b/max(n67,1):.1f}%)")
    print(f"最长 400 句中 boilerplate 占 {100*P.boiler[P.nlargest(400,'slen').index].mean():.1f}%")
    bp.to_csv(OUT / "12_boilerplate.csv", index=False)

    P["t07"] = tx.map(lambda t: any(w in t for w in TRIG_07))
    P["t06"] = tx.map(lambda t: any(w in t for w in TRIG_06))
    grp = {"06 only": P["L2-06"] & ~P["L2-07"], "07 only": P["L2-07"] & ~P["L2-06"],
           "06+07": P["L2-06"] & P["L2-07"], "neither": ~P["L2-06"] & ~P["L2-07"]}
    lex = pd.DataFrame([{"group": k, "n": int(m.sum()), "trig07_rate": P[m].t07.mean(),
                         "trig06_rate": P[m].t06.mean(), "mean_len": P[m].slen.mean()}
                        for k, m in grp.items()])
    print("\n-- 06/07 词汇边界（L2=%s）--" % PRIMARY)
    print(lex.round(3).to_string(index=False))
    lex.to_csv(OUT / "13_lexical_0607.csv", index=False)

    # ---------- 9. 稳健性：两套 L2
    hdr("9. 稳健性：主规格 L2=%s vs 对照 L2=%s" % (PRIMARY, "GPT"))
    ag = []
    for l in ALL8:
        a, b = S["GPT"][l].values, S["DS"][l].values
        both, na, nb = (a & b).sum(), a.sum(), b.sum()
        po = (a == b).mean()
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        ag.append({"l2": l, "block": "凝聚" if l in COH else "边界",
                   "n_DS(主)": int(nb), "n_GPT(对照)": int(na), "ratio_DS/GPT": nb / na,
                   "jaccard": both / (na + nb - both), "kappa": (po - pe) / (1 - pe)})
    ag = pd.DataFrame(ag)
    print(ag.round(3).to_string(index=False))
    ag.to_csv(OUT / "14_source_agreement.csv", index=False)

    tabs = []
    for s in L2_DIR:
        w = load_windows(s, "n3")
        tabs.append(npmi_edges(w).assign(src=s, camp="ALL"))
        for c in ["DPP", "KMT", "TPP"]:
            tabs.append(npmi_edges(w[w.camp == c], n_boot=800).assign(src=s, camp=c))
    ed = pd.concat(tabs, ignore_index=True)
    ed.to_csv(OUT / "15_edges_two_sources.csv", index=False)
    pv = ed.pivot_table(index=["camp", "pair"], columns="src", values="npmi")
    pv["Δ(对照-主)"] = pv.GPT - pv.DS
    print("\n-- 六条边 NPMI --"); print(pv.round(3).to_string())
    print("\n-- 边排序一致性 --")
    for c in ["ALL", "DPP", "KMT", "TPP"]:
        x = ed[ed.camp == c].pivot(index="pair", columns="src", values="npmi")
        print(f"  {c:4s} Spearman = {spearmanr(x.GPT, x.DS).statistic:+.3f}   "
              f"Pearson = {np.corrcoef(x.GPT, x.DS)[0,1]:+.3f}")
    print("\n-- 各源边排名（* = bootstrap 95% CI 含 0）--")
    for s in [PRIMARY, "GPT"]:
        for c in ["ALL", "DPP", "KMT", "TPP"]:
            x = ed[(ed.src == s) & (ed.camp == c)].sort_values("npmi", ascending=False)
            tag = "主" if s == PRIMARY else "照"
            print(f"  {s:3s}({tag}) {c:4s}: " + " > ".join(
                f"{r.pair}({r.npmi:.2f}{'' if r.sig else '*'})" for r in x.itertuples()))

    # ---------- 10. Layer B 可行性
    hdr("10. Layer B 可行性：L1(GPT) x camp x edge 的共现窗口数")
    sl1 = gpt_l1().rename(columns={"sent_idx": "sent_idx_start"})
    rows = []
    for s in L2_DIR:
        w = load_windows(s, "n3").merge(sl1, on=["doc_id", "sent_idx_start", "camp"],
                                        how="left", validate="one_to_one")
        for (l1, c), g in w.groupby(["L1", "camp"], observed=True):
            for a, b in PAIRS:
                rows.append({"src": s, "L1": l1, "camp": c, "pair": f"{a[3:]}-{b[3:]}",
                             "n_win": len(g),
                             "n_ab": int(g.l2.map(lambda x, a=a, b=b: a in x and b in x).sum())})
    av = pd.DataFrame(rows)
    av.to_csv(OUT / "16_layerB_feasibility.csv", index=False)
    print("主规格 L2=%s：" % PRIMARY)
    print(av[av.src == PRIMARY].pivot_table(index=["L1", "camp"], columns="pair",
                                            values="n_ab").to_string())
    m = av.pivot_table(index=["L1", "camp", "pair"], columns="src", values="n_ab")
    for thr in (20, 30):
        print(f"\nn_ab>={thr}:  主(DS) {int((m.DS>=thr).sum())}/{len(m)}   "
              f"照(GPT) {int((m.GPT>=thr).sum())}/{len(m)}   "
              f"两源都满足 {int(((m.GPT>=thr)&(m.DS>=thr)).sum())}/{len(m)}")

    # ---------- 10b. 例句
    hdr("10b. 例句：凝聚簇高频组合各抽 40 句（主规格 L2=%s）" % PRIMARY)
    EX = [("17_exemplars_0607", ("L2-06", "L2-07")),
          ("18_exemplars_0408", ("L2-04", "L2-08")),
          ("19_exemplars_0406", ("L2-04", "L2-06")),
          ("20_exemplars_0708", ("L2-07", "L2-08")),
          ("21_exemplars_04", ("L2-04",))]
    for name, codes in EX:
        sel = P[P[list(codes)].all(axis=1)]
        ex = (sel.sample(min(40, len(sel)), random_state=SEED)
                 .sort_values(KEY)
                 .assign(l2s=lambda x: x.l2.map(lambda s: "+".join(sorted(s))))
                 [["camp", "genre", "L1", "l2s", "sent_text"]])
        ex.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}: 母体 {len(sel)} 句 -> 抽 {len(ex)}")

    # ---------- 11. 调用架构
    hdr("11. 调用架构：全部 28 对的同句超额，一次多标签(GPT) vs 八次独立二元(DS)")
    arch = pd.concat(
        [perm_excess(S[s], ["doc_id"], PAIRS28, ALL8, tag=s) for s in L2_DIR],
        ignore_index=True)
    w = (arch[arch.lag == "same"].pivot(index="pair", columns="spec", values="ratio")
         .rename_axis(columns=None))
    z = (arch[arch.lag == "same"].pivot(index="pair", columns="spec", values="z")
         .rename_axis(columns=None))
    others = [s for s in L2_DIR if s != PRIMARY]
    ref = others[0]
    tab = pd.DataFrame({
        PRIMARY: w[PRIMARY], ref: w[ref],
        f"{PRIMARY}_minus_{ref}": w[PRIMARY] - w[ref],
        f"z_{PRIMARY}": z[PRIMARY], f"z_{ref}": z[ref],
    }).sort_values(f"{PRIMARY}_minus_{ref}", ascending=False)
    print(tab.round(3).to_string())
    stat, p = wilcoxon(w[PRIMARY], w[ref])
    print(f"\n平均 ratio: {PRIMARY} {w[PRIMARY].mean():.3f}   {ref} {w[ref].mean():.3f}")
    print(f"ratio<1（低于零模型，即互斥）: {PRIMARY} {(w[PRIMARY] < 1).sum()}/{len(w)}   "
          f"{ref} {(w[ref] < 1).sum()}/{len(w)}")
    print(f"{PRIMARY} > {ref} 的对数: {(w[PRIMARY] > w[ref]).sum()}/{len(w)}   "
          f"Wilcoxon 配对 p = {p:.3g}")
    tab.to_csv(OUT / "22_call_architecture_28pairs.csv")

    print(f"\n[所有表已写入 {OUT}]")


if __name__ == "__main__":
    main()
