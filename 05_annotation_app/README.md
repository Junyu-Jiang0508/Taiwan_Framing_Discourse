# Taiwan Framing 標註（Streamlit Cloud 獨立 repo）

此資料夾可單獨推成一支 GitHub repository 並部署，不需整份 `Taiwan_Framing_Discourse` 專案。

## 目錄內需納入版控的檔案

- `app.py`、`requirements.txt`、`supabase_schema.sql`
- `label1_set_v2.csv`（標註語料）
- `codebooks/01_annotation_guide_label1_v9.csv`、`codebooks/02_annotation_guide_label2_v10.csv`

## Supabase

1. 在 Supabase SQL Editor 執行 `supabase_schema.sql`。
2. **Project Settings → API** 複製 **Project URL** 與 **anon public** key（長 `eyJ...`）。

## 本機執行

```bash
cd 05_annotation_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# 編輯 secrets.toml 填入 url 與 key
streamlit run app.py
```

## Streamlit Community Cloud

### 若日誌出現 `Python 3.14.x`（你目前狀況）

首次部署時若在 **Advanced settings** 選了 **3.14**，之後**無法在後台改成 3.12**。官方說明：[Upgrade Python on Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python)（需 **刪除該 app → 再 Deploy 一次**，在 Advanced 裡改選 **3.12**；**Secrets 要重新貼**；自訂子網域可沿用）。

建議不要用 3.14 預覽版；日誌裡已出現 `pandas==3.0.2`、`numpy==2.4.4` 等極新版本，與本機 3.11/3.12 行為可能不一致。

### 部署步驟

1. 新建 GitHub repo，只放入本資料夾內容（或設為 monorepo 子目錄時見下）。
2. [share.streamlit.io](https://share.streamlit.io) → New app → 選該 repo。
3. **Main file path**：若 repo 根目錄就是本資料夾內容，填 `app.py`；若在子目錄則填 `05_annotation_app/app.py`（依你實際結構）。
4. **Advanced settings → Python version**：請選 **3.12**（或 **3.11**）；**勿選 3.14**，除非你知道相容風險。
5. **Secrets**（TOML）貼上並 **Save**；部署後若仍讀不到，到 **Manage app → ⋮ → Reboot app**。

```toml
[supabase]
url = "https://xxxx.supabase.co"
key = "eyJ... 或 sb_publishable_...（自 API 頁完整複製）"
```

6. **Deploy**。

## 安全提醒

預設未啟用 RLS；請將 app 連結僅提供給可信標註者，並勿將 `secrets.toml` 提交至 Git。
