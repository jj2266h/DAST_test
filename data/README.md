# 資料用途

- `raw/cmapss/local/`：原始 C-MAPSS；server_snapshot 副本獨立保留。
- `processed/cmapss_legacy/local/`：既有歷史 MAT，生成設定與版本未完全確立，請勿用新前處理覆寫。
- `processed/cmapss_current/local/`：現用 config 的新前處理輸出；本次未產生資料。
- `processed/femto/local/`：搬移前的 FEMTO 處理資料。
- `legacy_packages/cmapss_window40/`：舊 ZIP，未解壓合併。
- 根目錄 `phm2012_dataset/`：獨立 Git 資料集儲存庫，原位保留。

模型權重、特徵維度與 MAT 的配對不可只靠資料集名稱判定。
