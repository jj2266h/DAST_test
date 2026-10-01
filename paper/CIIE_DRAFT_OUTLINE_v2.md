# CIIE 2026 論文初稿架構 v2（時間壓縮版）

> 日期：2026-10-01
> 格式限制（取自 `工工協會投稿/CIIE 2026_template_fullpaper_1.docx`）：Word 檔、雙欄、含圖表與參考文獻共 **10 頁**、檔案 ≤ 2 MB；中文論文需在文末附**英文題目、作者、摘要（250 字）**，關鍵字 3–5 個。
> v1（`CIIE_MANUSCRIPT_SKELETON_2026-10-01.md`、`CIIE 2026fullpaper.docx`）只用 test-selected 的單次歷史數字。v2 改成以下面三批**比較站得住腳**的證據為主，舊數字降為背景。

---

## 0. 證據分級（寫作時只能依這張表下結論）

| 等級 | 證據 | 來源 | 在論文中的用途 |
|---|---|---|---|
| **A：可作為主結果** | RQ1 量化：OC 對 sensor 變異的解釋比例（η²）與 sensor–RUL 相關係數（\|ρ\|） | `paper/rq1_oc_variance.py` → `paper/data/rq1_oc_variance.csv`、`paper/figures/fig3_rq1_oc_variance.png` | RQ1 主結果 |
| **A：可作為主結果** | Clean 協定 72 runs：FD002/FD004、OC-z、engine-disjoint 70/15/15 切分、以 val 選模、在 holdout 評估；clip × warmup 4×3 網格，每格 3 seeds | `server_results/experiments/clean_sweep_results/summary.jsonl` | RQ3（訓練技巧的敏感度） |
| **A（尚未跑）** | **Global min-max / Global z / OC-z 的同協定對照** | 需補做，約 1 小時，見 §6 | RQ2 主結果；**沒有這組實驗，RQ2 只能寫成觀察** |
| B：探索性結果 | 144 次 sweep：FD001–FD004 × 12 設定 × 3 seeds，用 test 選 epoch | `server_results/summary/*.csv` | 補充說明各資料集的差異；需在表註明「test-selected」 |
| C：只能作為背景 | 單次歷史結果：14.39 / 20.52 → 13.49 / 13.01 等 | `紀錄/` 與簡報 | 只放在 Discussion 的一句話，或乾脆不放 |

---

## 題目（建議）

- 中文：**多操作工況下工況感知正規化對 DAST 剩餘壽命預測之影響分析**
- English：*Effect of Operating-Condition-Aware Normalization on DAST-Based Remaining Useful Life Prediction under Multiple Operating Conditions*

關鍵字：剩餘壽命預測、預知保養、操作工況、正規化、Transformer

---

## 頁數分配（10 頁，雙欄）

| 章節 | 頁數 | 圖表 |
|---|---|---|
| 摘要 + 1 緒論 | 1.3 | — |
| 2 文獻回顧 | 1.0 | — |
| 3 研究方法 | 1.7 | 圖 1、式 (1)–(4) |
| 4 實驗設計 | 1.2 | 表 1、表 2 |
| 5 結果與討論 | 3.0 | 圖 2、圖 3、表 3、表 4（、圖 4 選配） |
| 6 結論 | 0.4 | — |
| 參考文獻 + 英文摘要 | 1.4 | — |

---

## 1 緒論（約 1.3 頁，5 段）

1. **背景**：PHM、預知保養、資料驅動的 RUL 預測。引用 [Saxena 2008]、[Ramasso & Saxena 2020]。
2. **問題**：FD002/FD004 的工況幾乎每個 cycle 都在切換，sensor 讀值同時反映「工況」與「退化」兩件事。這裡要**明確定義**，本文談的是「工況造成的 sensor 分布異質性」，不是 train/test 之間的 covariate shift。
3. **缺口**：DAST [Zhang 2022] 以 sensor/time-step 雙注意力建模，但原文使用全域 min-max 正規化，沒有處理工況。STAGNN [Huang 2023] 雖已比較過 cluster normalization，但模型不同，而且把正規化效果和模型貢獻混在一起評估。目前缺少的是在**固定 Transformer 架構**下，把前處理效果和訓練技巧效果分開的評估。
4. **本文做法**：固定 DAST，只更換前處理；所有統計量都只由 training engines 估計；以 engine-disjoint validation 選模。
5. **研究問題與貢獻**：
   - RQ1：工況能解釋多少 sensor 變異？全域正規化能不能消除它？
   - RQ2：OC-aware z-score 能否改善 FD002/FD004 的預測？
   - RQ3：warmup 與 gradient clipping 能否在前處理之外帶來額外效益？
   - 貢獻寫法：(i) 量化分析，說明全域正規化在數學上無法移除工況造成的變異；(ii) leakage-safe 的前處理流程；(iii) 在固定 DAST 下的對照實驗與訓練技巧敏感度分析。

---

## 2 文獻回顧（約 1 頁，3 小節；只用已核對的 12 篇）

- **2.1 資料驅動 RUL 與 Transformer**：C-MAPSS [Saxena 2008]、benchmark 的可比性問題 [Ramasso & Saxena 2020]、Transformer [Vaswani 2017]、DAST [Zhang 2022]、sensor selection [Mp 2023]、工業資料上的 Transformer RUL [Dintén 2026]（選配）。
- **2.2 多操作工況下的 RUL 預測**：
  - 把工況當作輸入通道 [Cheng 2022]
  - 把工況資訊提供給 LSTM-AE [de Pater & Mitici 2023]
  - domain adaptation 觀點 [Wang 2025, preprint]
  - 結論：現有作法可分為「輸入層」、「模型層」、「領域適應」三類，本文屬於最輕量的**前處理層**。
- **2.3 依工況分群正規化**：STAGNN [Huang 2023] 是最直接的先例（依工況分群後做 min-max）。
  - 本文與它的差異：z-score、固定 DAST、統計量只用 train 計算並以 val 選模、加上 η² 的量化分析。
  - ⚠️ Pasa 2019 與 Zhang 2023（K-means + DCNN）只有摘要。**沒有取得全文就不要引用**；若來不及取得，這一節寫到 STAGNN 為止即可。
  - ⚠️ 目前沒有 gradient clipping 和 RAdam 的文獻。RQ3 只能引用 Vaswani 2017 的 warmup；如果要討論「RAdam 可能降低 warmup 的必要性」，需要補上 RAdam 原始論文。

---

## 3 研究方法（約 1.7 頁）

- **3.1 問題定義與 RUL 標籤**：piecewise 標籤，上限 125 [Zhang 2022]。以 14 顆 sensor 為輸入（移除 s1、s5、s6、s10、s16、s18、s19）。
- **3.2 工況辨識**：將 op1–op3 除以各自的標準差後，用 K-means 分群（FD002/FD004 取 k=6，FD001/FD003 取 k=1）。**分群中心只用 training engines 計算**；validation、holdout、test 的資料以最近中心指派到工況。
- **3.3 工況感知 z-score**：
  - 式 (1)：z = (x − μ_c,j) / σ_c,j，其中 μ、σ 只由 training engines 中屬於工況 c 的資料計算。
  - 式 (2)：全域正規化（min-max 或 z-score）對每顆 sensor 都只是一個仿射轉換 a_j·x + b_j，**因此工況之間的相對位移不會改變**。這是本文論證的關鍵，放在方法章節裡直接寫出來。
- **3.4 滑動視窗與統計特徵**：window 40、stride 1；在每個 window 後面附加斜率與平均值兩列（T = 42），依 DAST 原文作法。
- **3.5 DAST 架構概述**：sensor encoder、time-step encoder、特徵融合、decoder；參數沿用 `config.json`。**dropout 設為 0.1，原文是 0.2，必須明確寫出。**
- **3.6 訓練技巧**：線性 warmup（前 w 個 epoch）、梯度範數裁切（max_norm = c）、RAdam 優化器。
- **圖 1：研究框架**。要畫出 train 資料流（fit）和 val/holdout 資料流（只 transform）是**分開的兩條路徑**。

---

## 4 實驗設計（約 1.2 頁）

- **4.1 資料集**。**表 1**：

  | 資料集 | 工況數 | 故障模式 | train engines | test engines | 最短 test 長度 | 角色 |
  |---|---|---|---|---|---|---|
  | FD001 | 1 | 1 | 100 | 100 | 31 | 單工況對照 |
  | FD002 | 6 | 1 | 260 | 259 | 21 | 多工況主資料集 |
  | FD003 | 1 | 2 | 100 | 100 | 38 | 單工況對照 |
  | FD004 | 6 | 2 | 249 | 248 | 19 | 多工況主資料集 |

- **4.2 評估協定**：
  - 將官方 train 檔依 engine 切成 70/15/15（split seed 2026）。
  - 分群中心與 scaler 只 fit training engines。
  - 以 val RMSE 選 epoch，在 holdout engines 的**所有 window** 上評估。
  - **官方 test set 在這個協定中完全沒有被讀取**（`server_results/experiments/train_split.py:2`）。
  - 必須在 4.2 寫明：holdout 數值**不能和 DAST 原文或官方 test 的數值直接比較**，因為評估對象不同（所有 window 對上最後一個 window），Score 的聚合方式也不同（每個 engine 先取平均，對上直接加總）。
- **4.3 評估指標**：
  - RMSE（cycles）。
  - NASA Score：先在每個 engine 的所有 window 上取平均，再對 engine 取平均（`train_split.py:80-89`）。
- **4.4 實驗設定**。**表 2**：
  - 固定參數：epochs 100、batch 256、lr 1e-3、RAdam、RUL 上限 125、window 40。
  - 網格：clip ∈ {0.5, 1, 2, 5}、warmup ∈ {5, 10, 15}。
  - seeds {20, 42, 100}（新實驗改為 5 個 seeds，見 §6）。

---

## 5 結果與討論（約 3 頁）

### 5.1 RQ1：工況對 sensor 分布的影響（證據等級 A，已完成）

**圖 2**（直接沿用既有圖檔）：`analysis/figures/04_before_after_norm.png` 或 `04_sensor_boxplot_by_oc.png`，呈現 FD004 的 sensor 在正規化前後的分布。

**圖 3**（新產生）：`paper/figures/fig3_rq1_oc_variance.png`。

可寫的結果（數值取自 `paper/data/rq1_oc_variance.csv`，14 顆 sensor 的平均；只用 train 檔計算）：

| 指標 | FD002：Raw／Global | FD002：OC-z | FD004：Raw／Global | FD004：OC-z |
|---|---|---|---|---|
| 40-cycle window 內，工況可解釋的變異比例 η² | 0.999 | 0.212 | 0.999 | 0.223 |
| \|Spearman ρ(sensor, RUL)\| | 0.114 | 0.497 | 0.112 | 0.403 |

寫作重點：

1. 未處理時，一個 window 內約 99.9% 的 sensor 變動可以由工況切換解釋。DAST 的 time-step attention 看到的主要是工況雜訊。
2. **全域 min-max 或全域 z-score 對上面兩個指標完全沒有影響**（仿射轉換不改變 η² 和 ρ），所以原始 DAST 的正規化方式在數學上無法處理這個問題。
3. OC-z 之後 \|ρ\| 提高了 3.6 到 4.4 倍，表示退化趨勢被顯露出來。
4. 必須誠實交代的限制：
   - 全資料的 η²（pooled）在 OC-z 之後等於 0 是**定義使然**，所以正文只報告 window 內的 η²。
   - s8 和 s13 在 OC-z 之後仍有約 0.6 的 window 內 η²，可能表示這兩顆 sensor 的退化速率本身和工況有關。這是一個假說，要寫成「可能」。
   - FD004 的 s7、s12、s15、s20、s21 在 OC-z 之後 \|ρ\| 仍然偏低（0.06–0.12）。可能和 FD004 的兩種故障模式對這些 sensor 的影響方向不同有關，同樣是假說。

### 5.2 RQ2：正規化方式的比較（**需先完成 §6 的實驗**）

**表 3**（預留）：FD002/FD004（加 FD001/FD003 作為對照）× {Global min-max、Global z、OC-z} × 5 seeds，不開 clip 也不開 warmup。報告 holdout RMSE、Score 的 mean ± SD，以及 OC-z 減 Global z 的配對差值 Δ。

預期要寫的結構：

- FD002/FD004：預期 OC-z 的誤差最低。**實際寫法以跑出來的結果為準。**
- FD001/FD003：只有一個工況，OC-z 在數學上等於 Global z，兩者的差異只能反映 seed 雜訊。這是 negative control。
- 用詞：只有 5 個 seeds 時，只寫「平均較低」並附上 SD 與 Δ，**不要寫 statistically significant**。

**如果來不及補實驗（Plan B）**：
- 這一節改名為「5.2 工況感知正規化下之預測表現」，只報告 OC-z 在 holdout 上的結果（表 4 的最佳 val 設定）。
- RQ1 的機制分析搭配文字說明「全域正規化無法移除工況變異」。
- RQ2 在 Conclusion 降級為未來工作。
- 舊的 FD004 20.52 → 13.01 **不可以寫成改善幅度**；最多在 Discussion 用一句話提及，並註明「single run、test-selected、window 設定不同」。

### 5.3 RQ3：warmup 與 gradient clipping 的敏感度（證據等級 A，已完成）

**表 4**：OC-z 條件下的 holdout RMSE，mean ± SD，n = 3，以 val RMSE 選 epoch。

FD002：

| clip \ warmup | 5 | 10 | 15 |
|---|---|---|---|
| 0.5 | 14.04 ± 0.28 | 13.46 ± 0.37 | 14.13 ± 0.23 |
| 1.0 | 14.21 ± 0.33 | 14.21 ± 0.11 | 14.10 ± 0.61 |
| 2.0 | 14.38 ± 0.41 | 13.89 ± 0.48 | 13.99 ± 0.38 |
| 5.0 | 13.83 ± 0.70 | 14.19 ± 0.22 | 13.57 ± 0.52 |

FD004：

| clip \ warmup | 5 | 10 | 15 |
|---|---|---|---|
| 0.5 | 15.31 ± 0.54 | 15.35 ± 0.29 | 15.65 ± 0.41 |
| 1.0 | 15.40 ± 0.22 | 15.32 ± 0.71 | 15.18 ± 0.38 |
| 2.0 | 15.24 ± 0.13 | 15.52 ± 0.32 | 15.74 ± 0.43 |
| 5.0 | 14.98 ± 0.24 | 15.23 ± 0.52 | 15.04 ± 0.11 |

可寫的結果：

- 12 個設定之間的 RMSE 範圍：FD002 為 13.46–14.38（差距 0.92），FD004 為 14.98–15.74（差距 0.76）。同一設定在不同 seed 之間的 SD 為 0.11–0.71。
- 設定之間的差距只有 SD 的 1–2 倍，沒有任何一組設定在兩個資料集上都穩定勝出。
- 如果依 val 挑選設定，兩個資料集選出的都是 clip 5、warmup 5：FD002 holdout 13.83 ± 0.70、FD004 holdout 14.98 ± 0.24。
- **結論的寫法**：「在 OC-z 之下，DAST 對 warmup 與 clipping 的設定相對不敏感，看不到穩定的額外效益」。
- **不可以寫**：warmup 或 clipping「提升了訓練穩定性」。這個網格裡兩者都一直開著，沒有關閉的對照組，也沒有記錄 gradient norm。
- 144 次 test-selected sweep 可以在這裡用一句話補充：「在四個子資料集上，warmup 長度的影響方向也不一致」（見 `CLAIMS_FROM_RESULTS.md` 的 warmup 邊際表）。務必註明是 test-selected。

**圖 4（選配）**：在 holdout 上，最佳 val 設定的 True RUL vs Predicted RUL，或誤差分布 boxplot。需要從 checkpoint 做一次推論才能產生。

### 5.4 研究限制（一段，必寫）

- C-MAPSS 是模擬資料。
- 只使用 DAST 一種架構。
- holdout 的 engine 數量有限（FD002 39 台、FD004 38 台）。
- 每個設定只有 3–5 個 seeds。
- 沒有評估官方 test set，不和原論文的數值比較。
- 早期實驗曾以 test 選模；這些結果只作為探索用途，沒有進入主表。

---

## 6 結論（約 0.4 頁）

回答 RQ1–RQ3 各一句，再寫一句未來工作：官方 test 評估、更多 seeds、STAGNN 式 min-max 對照、真實產線資料。

---

## 7. 唯一需要補的實驗（強烈建議，約 1–2 小時 GPU）

每次訓練約 2 分鐘（clean sweep 平均：FD002 1.9 分、FD004 2.2 分）。

| 優先 | 實驗 | runs | 時間 |
|---|---|---|---|
| **必做** | FD002、FD004 × {Global min-max、Global z、OC-z} × clip/warmup 全關 × seeds {20, 42, 100, 7, 2026} | 30 | 約 1 小時 |
| 建議 | FD001、FD003 × {Global min-max、Global z} × 同樣 5 個 seeds（negative control） | 20 | 約 40 分鐘 |
| 建議 | FD002、FD004 × OC-z × {只開 warmup、只開 clip、兩者都開}，使用 val 選出的 c=5、w=5 × 5 seeds | 30 | 約 1 小時 |

需要修改的程式（小改動）：

1. `experiments/prepare_splits.py`：新增 `--norm {oc_z, global_z, global_minmax}`。目前只有 OC-z（`:152-163`）。
2. `experiments/train_split.py`：讓 `--clip 0` 代表**不做梯度裁切**。⚠️ 目前如果直接傳 0，會以 max_norm = 0 呼叫 `clip_grad_norm_`，梯度會全部被歸零；關閉與否只由 config 的 `grad_clip_enabled` 決定（`:150-151`）。`--warmup 0` 代表關閉 warmup，這一點目前已經支援（`:131`）。
3. `summary.jsonl` 要多記錄 `norm` 欄位。

---

## 8. 寫作順序（配合時間）

1. **現在就能寫**：§3 研究方法、§4 實驗設計、§5.1 RQ1、§5.3 RQ3、§2 文獻回顧。
2. 補實驗跑完後：§5.2 RQ2、摘要、結論。
3. 最後：英文題目與摘要、參考文獻格式（依中國工業工程學會的格式：中文文獻在前，英文文獻依作者姓氏排序）。
