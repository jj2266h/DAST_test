// Builds the CIIE 2026 full-paper draft (v2) as a Word file.
// Usage (from repo root): NODE_PATH=<dir with node_modules> node paper/build_docx.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  AlignmentType, WidthType, ShadingType, SectionType, TabStopType, Tab, VerticalAlign,
  Math: OMath, MathRun, MathFraction, MathSubScript, MathSuperScript, MathSum, MathRadical,
} = require("docx");

const FIG = (f) => fs.readFileSync(path.join(__dirname, "figures", f));
const COL_W = 4322;            // one text column in DXA (A4, 2.5 cm margins, 0.75 cm gap)
const COL_PX = 288;            // same width in px at 96 dpi
const BODY = 20, SMALL = 16;   // half-points: 10 pt body, 8 pt tables

// ---------- text helpers ----------
// Segments wrapped in 【】 are placeholders and get a yellow highlight.
// Inline "x_{c,j}" or "T_w" becomes a subscript run.
function runs(text, opts = {}) {
  const out = [];
  for (const s of text.split(/(【[^】]*】)/).filter(Boolean)) {
    const highlight = s.startsWith("【") ? "yellow" : undefined;
    for (const t of s.split(/(_\{[^}]*\}|_[A-Za-z0-9]+)/).filter(Boolean)) {
      const isSub = t.startsWith("_");
      out.push(new TextRun({ text: isSub ? t.replace(/^_\{?|\}$/g, "") : t, subScript: isSub,
        size: opts.size || BODY, bold: opts.bold, italics: opts.italics, highlight }));
    }
  }
  return out;
}
const para = (text, o = {}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  indent: o.noIndent ? undefined : { firstLine: 400 },
  spacing: { after: o.after ?? 60, line: 240 },
  children: runs(text, o),
});
const h1 = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 },
  children: [new TextRun({ text, bold: true, size: 22 })],
});
const h2 = (text) => new Paragraph({
  spacing: { before: 120, after: 60 },
  children: [new TextRun({ text, bold: true, size: BODY })],
});
const caption = (text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
  children: runs(text, { size: 18 }),
});
const figure = (file, pxW, pxH, cap) => [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100 },
    children: [new ImageRun({ type: "png", data: FIG(file),
      transformation: { width: COL_PX, height: Math.round((COL_PX * pxH) / pxW) } })] }),
  caption(cap),
];

// ---------- equation helpers ----------
const R = (t) => new MathRun(t);
const sub = (b, s) => new MathSubScript({ children: [R(b)], subScript: [R(s)] });
const sup = (b, s) => new MathSuperScript({ children: typeof b === "string" ? [R(b)] : b, superScript: [R(s)] });
const frac = (n, d) => new MathFraction({ numerator: n, denominator: d });
const eq = (parts, no) => new Paragraph({
  tabStops: [{ type: TabStopType.CENTER, position: COL_W / 2 }, { type: TabStopType.RIGHT, position: COL_W }],
  spacing: { before: 120, after: 120 },
  children: [new TextRun({ children: [new Tab()] }), new OMath({ children: parts }),
             new TextRun({ children: [new Tab(), `(${no})`], size: BODY })],
});

// ---------- table helpers ----------
const border = { style: "single", size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, w, o = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, borders, columnSpan: o.span,
    verticalAlign: VerticalAlign.CENTER,
    shading: o.head ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" } : undefined,
    margins: { top: 30, bottom: 30, left: 60, right: 60 },
    children: [new Paragraph({ alignment: o.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(text, { size: SMALL, bold: o.head || o.bold }) })],
  });
}
function table(widths, rows) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({ children: r.map((c, k) => {
      if (c && c.span) return cell(c.text, widths.slice(c.start, c.start + c.span).reduce((a, b) => a + b, 0), { span: c.span, left: true, bold: true });
      return cell(c, widths[k], { head: i === 0 });
    }) })),
  });
}

// ---------- content ----------
const front = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "多操作工況下工況感知正規化對DAST剩餘壽命預測之影響分析", bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("【作者姓名】*", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("【服務機關與地址】", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs("*聯絡人電子郵件：【待補】", { size: 24 }) }),
  h1("摘要"),
  para("航空發動機在多種操作工況下運轉時，感測器量測值同時反映設備退化與操作條件，使資料驅動的剩餘壽命（remaining useful life, RUL）預測面臨輸入分布異質的問題。本研究以NASA C-MAPSS資料集與雙面向自注意力Transformer（DAST）為基礎，在不修改模型架構的前提下，探討工況感知正規化的影響。研究流程以訓練引擎的操作設定建立K-means工況分群，並於各工況內進行逐感測器正規化（z-score或min-max）；所有統計量僅由訓練引擎估計，並以引擎層級的驗證集選擇模型。分析顯示，在FD002與FD004中，長度40的視窗內約99.9%的感測器變異可由工況切換解釋，且全域正規化無法改變此比例；經工況感知正規化後，該比例降至0.21至0.22，感測器與RUL的平均相關係數由約0.11提高至0.40以上。在保留引擎上以每台引擎一個隨機截斷點評估時，工況感知min-max相較DAST原文採用的全域min-max，使FD002與FD004的RMSE分別降低4.2與6.8，以隨機種子配對之差值的95%信賴區間皆不含零。在工況感知正規化之下，12組梯度裁切與學習率暖身設定之間的保留集RMSE差距約為隨機種子間標準差的1至2倍，未呈現一致的額外效益。"),
  para("關鍵字：剩餘壽命預測、預知保養、操作工況、正規化、Transformer", { noIndent: true, after: 200 }),
];

const body = [
  h1("1. 緒論"),
  para("剩餘壽命預測是預知保養（predictive maintenance）與預測健康管理（prognostics and health management, PHM）的核心工作，其目的在於依據設備的狀態監測資料估計距離失效尚餘的運轉時間，以協助維修規劃並降低非預期停機。隨著感測資料的累積，資料驅動方法可直接由多變量時間序列學習退化模式；NASA釋出的C-MAPSS航空發動機模擬資料集具備完整的run-to-failure軌跡，已成為評估RUL方法的常用基準[5][6]。"),
  para("然而，發動機感測器的量測值不只反映健康狀態，也隨飛行高度、馬赫數與油門解析角等操作設定而改變。C-MAPSS的FD002與FD004子資料集包含六種操作工況（operating condition, OC）[6]，且工況在引擎生命週期中頻繁切換。在此情形下，同一感測器的數值依工況分成數個群，群間差距遠大於退化造成的緩慢變化。本文將此現象稱為「工況造成的感測器分布異質性」，以區別於訓練與測試資料之間的共變量偏移：在C-MAPSS各子資料集中，訓練與測試引擎共享相同的工況集合，問題在於工況與退化訊號混雜於同一輸入之中。"),
  para("Zhang等人提出的DAST以平行的感測器編碼器與時間步編碼器，分別學習不同感測器與不同時間位置的重要性，其前處理採用對每一感測器的全域min-max正規化[9]。Huang等人指出多數研究在前處理時忽略工況，並在其時空注意力圖神經網路（STAGNN）中比較全域正規化與依工況分群的正規化[3]。然而，該研究將正規化方式與新模型架構一併評估，較難單獨判斷前處理本身的影響；對於DAST這類既有的Transformer架構，工況處理的作用仍缺乏在固定架構與無資料洩漏協定下的分析。"),
  para("因此，本研究固定DAST架構，僅改變輸入前處理，探討三個研究問題。RQ1：工況可解釋多少感測器變異？全域正規化能否消除此變異？RQ2：工況感知正規化能否改善多工況子資料集的RUL預測？工況內採用z-score或min-max是否有差異？RQ3：在工況感知正規化之下，學習率暖身（warmup）與梯度裁切（gradient clipping）是否帶來額外效益？本文的貢獻包括：(1)以視窗層級的工況解釋變異量與感測器–RUL相關係數量化工況對模型輸入的影響，並說明全域正規化在數學上無法移除此影響；(2)建立工況分群與正規化統計量僅由訓練引擎估計、並以引擎層級驗證集選模的前處理流程；(3)在固定DAST下分析訓練策略的敏感度。"),

  h1("2. 文獻回顧"),
  h2("2.1 資料驅動RUL預測與Transformer"),
  para("Saxena等人說明C-MAPSS退化資料的產生方式，包括以熱力學模擬模型建立感測器響應、於模組的流量與效率施加指數型退化，以及非對稱的評分函數[6]。Ramasso與Saxena整理使用C-MAPSS的預測方法，指出感測器雜訊、操作工況效應與多故障模式是此資料集的主要挑戰，並提醒不同研究在資料使用與評估方式上的差異會影響結果的可比性[5]。在模型方面，Transformer以自注意力機制取代遞迴結構[7]；DAST將此概念延伸至RUL預測，同時在感測器與時間步兩個面向計算注意力，再經特徵融合與解碼器輸出RUL[9]。另外，感測器的選擇對RUL估計的準確度與建置成本亦有影響[4]。"),
  h2("2.2 多操作工況下的RUL預測"),
  para("既有研究處理工況的方式可大致分為三類。第一類在輸入層加入工況資訊：Cheng等人以多維遞迴神經網路將多感測器資料與操作工況資料分通道輸入，處理變動工況與多故障模式[1]；de Pater與Mitici在N-CMAPSS上將操作工況提供給LSTM自編碼器以建立健康指標[2]。第二類為領域適應，Wang等人回顧渦扇發動機RUL的領域適應方法，並將工況變動造成的分布偏移列為主要挑戰之一[8]。第三類則在前處理階段依工況分群後分別正規化[3]。前處理方法不需修改模型架構，較容易套用於既有模型，也是本研究的關注點。"),
  h2("2.3 依工況分群之正規化"),
  para("STAGNN以分群模型辨識操作工況，再對每個工況群分別進行max-min正規化，並在多工況資料集上觀察到較佳的預測表現[3]。本研究與其差異在於：(1)同時比較工況內z-score與min-max兩種尺度轉換；(2)固定既有的DAST架構，使效能差異可歸因於前處理；(3)明確以訓練引擎估計工況中心與正規化統計量，並以引擎層級驗證集選模；(4)以視窗層級指標量化正規化對模型輸入的作用。【若取得Pasa等人（2019）與Zhang等人（2023）全文，於此補充比較；未取得則刪除本註記。】"),

  h1("3. 研究方法"),
  ...figure("fig1_framework.png", 850, 960, "圖1. 研究框架"),
  para("研究框架如圖1所示。左側為訓練引擎的流程，負責估計工況分群中心與正規化統計量；右側的驗證與保留引擎只套用這些已估計的參數，不重新計算。正規化後的資料經滑動視窗處理後輸入DAST，並以驗證集選擇模型。"),
  h2("3.1 問題定義與RUL標籤"),
  para("設第i台引擎在第t個cycle的觀測包含操作設定o = (op1, op2, op3)與21個感測器量測值。依DAST原文，移除在退化過程中維持定值的感測器s1、s5、s6、s10、s16、s18、s19，保留其餘14個感測器[9]。訓練引擎在cycle t的RUL標籤採分段線性形式："),
  eq([sub("y", "i,t"), R(" = min("), sub("T", "i"), R(" − t, "), sub("R", "max"), R(")")], 1),
  para("其中T_i為引擎i的失效cycle，R_max = 125，與DAST的設定一致[9]。", { noIndent: true }),
  h2("3.2 工況辨識"),
  para("將三個操作設定分別除以其在訓練引擎上的標準差，得到標準化設定õ，再以K-means分群，並依最近中心指派工況："),
  eq([sub("c", "i,t"), R(" = "), sub("argmin", "k"), R(" ‖"), sub("õ", "i,t"), R(" − "), sub("m", "k"), R("‖")], 2),
  para("FD002與FD004取K = 6，對應其六種工況[6]；FD001與FD003僅有單一工況，取K = 1，以免將操作設定的微小擾動誤分為不同工況。分群中心m_k只由訓練引擎估計；驗證與保留引擎以相同中心依式(2)指派工況，不重新分群。", { noIndent: true }),
  h2("3.3 工況感知正規化"),
  para("對工況c與感測器j，以訓練引擎中屬於工況c的全部資料點計算平均μ_{c,j}與標準差σ_{c,j}，並正規化為："),
  eq([sub("z", "i,t,j"), R(" = "), frac([sub("x", "i,t,j"), R(" − "), sub("μ", "c,j")], [sub("σ", "c,j")]), R(",  c = "), sub("c", "i,t")], 3),
  para("工況感知min-max則以訓練引擎中屬於工況c的最小值與最大值，將式(3)的μ_{c,j}換成min_{c,j}、σ_{c,j}換成max_{c,j} − min_{c,j}。驗證與保留引擎直接套用訓練引擎估計的統計量。作為對照，全域正規化（無論min-max或z-score）對每一感測器皆為與工況無關的仿射轉換：", { noIndent: true }),
  eq([sub("x̃", "i,t,j"), R(" = "), sub("a", "j"), sub("x", "i,t,j"), R(" + "), sub("b", "j"), R(",  "), sub("a", "j"), R(" > 0")], 4),
  para("仿射轉換只改變各感測器的尺度與位置，不改變不同工況之間的相對位移，因此全域正規化無法移除工況造成的群間差異。第5.1節以量化指標驗證此性質。", { noIndent: true }),
  h2("3.4 滑動視窗與統計特徵"),
  para("正規化後，以長度T_w、步長1的滑動視窗切分每台引擎的序列，並以視窗最後一個cycle的RUL作為該視窗的標籤[9]。依DAST原文，另對每一視窗計算各感測器的平均值與線性迴歸斜率[9]；兩者以訓練資料擬合的min-max尺度轉換後附加於視窗之後，使模型輸入維度為(T_w + 2) × 14。T_w的設定見第4.4節。"),
  h2("3.5 DAST模型"),
  para("DAST由感測器編碼器、時間步編碼器、特徵融合層與解碼器組成[9]。感測器編碼器以多頭自注意力學習感測器之間的關係與權重；時間步編碼器則學習不同時間位置的重要性。兩個編碼器平行運作，輸出經特徵融合後送入解碼器預測RUL。本研究不修改DAST的架構，僅調整輸入前處理與訓練策略，模型超參數列於表2。"),
  h2("3.6 訓練策略"),
  para("模型以RMSE作為訓練損失，並使用RAdam最佳化器[9]。學習率暖身在前w個epoch內，將學習率由接近0線性提高至設定值η_0："),
  eq([sub("η", "s"), R(" = "), sub("η", "0"), R(" · min(1, "), frac([R("s + 1")], [R("w · "), sub("N", "b")]), R(")")], 5),
  para("其中s為參數更新步數，N_b為每個epoch的批次數。Transformer原始論文即在學習率排程中採用暖身[7]。梯度裁切則在每次更新前檢查所有參數梯度的整體L2範數，若超過門檻c，便將梯度等比例縮小至範數為c。", { noIndent: true }),

  h1("4. 實驗設計"),
  h2("4.1 資料集"),
  para("本研究使用C-MAPSS的四個子資料集[6]，其特性如表1。FD002與FD004為主要分析對象；FD001與FD003只有單一工況，作為參照。"),
  caption("表1. C-MAPSS子資料集特性"),
  table([900, 750, 850, 900, 922], [
    ["資料集", "工況數", "故障模式數", "訓練引擎數", "本文角色"],
    ["FD001", "1", "1", "100", "單工況參照"],
    ["FD002", "6", "1", "260", "多工況主體"],
    ["FD003", "1", "2", "100", "單工況參照"],
    ["FD004", "6", "2", "249", "多工況主體"],
  ]),
  h2("4.2 資料切分與評估協定"),
  para("為避免測試資料參與模型選擇，本研究只使用各子資料集的官方訓練檔。以引擎為單位，依固定亂數種子（2026）將引擎隨機分為訓練、驗證與保留（holdout）三組，比例為70%、15%、15%；FD002為182、39、39台，FD004為174、37、38台。每台引擎只屬於其中一組，且先切分引擎、再建立滑動視窗，因此不會因視窗重疊而洩漏資訊。工況分群中心、正規化統計量與統計特徵的尺度轉換皆只由訓練引擎估計。每個epoch結束時計算驗證引擎的RMSE，保存最低者的模型參數，最後以該模型在保留引擎上評估一次；官方測試集在此協定中完全未被使用。"),
  para("保留引擎取自官方訓練檔，皆運轉至失效，其最後一個視窗的真實RUL恆為0，因此本文以兩種方式評估。(1)全視窗：涵蓋保留引擎生命週期內的所有滑動視窗。(2)截斷點：仿照官方測試集的產生方式，以固定亂數種子（2026）為每台保留引擎均勻抽取一個時間點，只評估該點的視窗；FD002與FD004各有39與38個評估點，且同一視窗長度下各正規化方式使用相同的截斷點。本文數值仍不宜與官方測試集上的文獻結果（如[9]）直接比較。"),
  h2("4.3 評估指標"),
  para("RMSE定義如下，其中N為評估視窗數："),
  eq([R("RMSE = "), new MathRadical({ children: [frac([R("1")], [R("N")]),
        new MathSum({ children: [sup([R("("), sub("ŷ", "n"), R(" − "), sub("y", "n"), R(")")], "2")], subScript: [R("n=1")], superScript: [R("N")] })] })], 6),
  para("NASA Score的單點評分為s(d) = exp(−d/13) − 1（d < 0）或exp(d/10) − 1（d ≥ 0），其中d = ŷ − y；此評分對高估RUL（延後預警）的懲罰較重[6]。本文以截斷點計算Score，與DAST原文相同，將所有保留引擎的評分加總[9]：", { noIndent: true }),
  eq([R("Score = "), new MathSum({ children: [R("s("), sub("d", "u"), R(")")], subScript: [R("u=1")], superScript: [R("U")] })], 7),
  para("其中U為保留引擎數，d_u為引擎u在截斷點的預測誤差。全視窗評估只報告RMSE。", { noIndent: true }),
  h2("4.4 實驗設定"),
  para("實驗設定列於表2。RQ2沿用DAST原文在FD002與FD004的視窗長度60[9]，關閉梯度裁切與暖身，以5個隨機種子重複；RQ3的梯度裁切與暖身網格則以視窗長度40、3個隨機種子進行。由於兩組實驗的視窗長度不同，表3與表4的數值不宜直接比較。此外，本研究的dropout為0.1（原文為0.2），也不採用原文重複10次取平均的做法[9]，因此結果不作為與原文數值的直接比較。"),
  caption("表2. 實驗設定"),
  table([1500, 2822], [
    ["項目", "設定"],
    ["輸入感測器", "14個（移除s1, s5, s6, s10, s16, s18, s19）"],
    ["視窗長度", "RQ2：60；RQ3：40（附加統計特徵後各加2）"],
    ["RUL上限", "125"],
    ["最佳化器／學習率", "RAdam／0.001"],
    ["批次大小／epoch", "256／100"],
    ["DAST結構", "維度64、注意力頭數4、編碼器2層、解碼器1層、dropout 0.1"],
    ["梯度裁切門檻c", "RQ2：關閉；RQ3：0.5, 1, 2, 5"],
    ["暖身長度w（epoch）", "RQ2：關閉；RQ3：5, 10, 15"],
    ["隨機種子", "RQ2：20, 42, 100, 7, 2026；RQ3：20, 42, 100"],
    ["模型選擇", "驗證RMSE最低之epoch"],
  ]),

  h1("5. 結果與討論"),
  h2("5.1 工況對感測器分布的影響（RQ1）"),
  ...figure("fig2_fd004_s12_s14_before_after.png", 1440, 960, "圖2. FD004感測器s12與s14在工況感知正規化前後與RUL的關係"),
  para("圖2以FD004的s12與s14為例。正規化前，s12依工況分成六條幾乎平行的水平帶，同一工況內幾乎看不出隨RUL變化的趨勢；s14也呈現明顯的工況分群。經工況感知正規化後，不同工況的資料點重疊在一起，RUL接近0時的退化趨勢隨之顯現。"),
  para("為量化此現象，本文使用官方訓練檔的全部引擎計算兩項描述性指標。第一項為視窗層級工況解釋變異量η²_w：對每個長度40（步長10）的視窗，以視窗內的工況標籤對感測器值進行單因子變異數分解，計算組間變異占總變異的比例，再對所有包含兩種以上工況的視窗取平均。第二項為各感測器與RUL的Spearman相關係數絕對值|ρ|。依式(4)，全域min-max與全域z-score不會改變這兩項指標，因此其數值與未正規化時相同。"),
  ...figure("fig3_rq1_column.png", 1020, 1680, "圖3. 工況感知正規化前後的(a)(b)視窗層級工況解釋變異量與(c)(d)感測器–RUL相關係數"),
  para("如圖3所示，未進行工況感知正規化時，FD002與FD004的14個感測器之η²_w平均皆為0.999，也就是視窗內的感測器變動幾乎完全來自工況切換。經工況感知z-score後，平均η²_w降至0.212（FD002）與0.223（FD004）。同時，感測器與RUL的平均|ρ|由0.114提高至0.497（FD002），由0.112提高至0.403（FD004）。這表示在未處理工況時，DAST的時間步編碼器所看到的相鄰時間步差異主要反映工況切換，而非退化；工況感知正規化則使退化趨勢在輸入中顯現。"),
  para("有三點需要說明。第一，若以全體資料而非個別視窗計算，工況感知正規化後的組間變異依定義即為零，因此本文只以視窗層級指標作為證據。第二，s8與s13在正規化後仍有約0.6的η²_w（FD002為0.640與0.642，FD004為0.607與0.615），可能表示這兩個感測器的退化程度與工況之間存在交互作用。第三，FD004的s7、s12、s15、s20、s21在正規化後的|ρ|仍只有0.06至0.12；由圖2可見s12在正規化後呈現兩種不同的走向，推測與FD004的兩種故障模式有關。後兩點仍待進一步驗證。"),

  h2("5.2 正規化方式之比較（RQ2）"),
  caption("表3. 不同正規化方式在保留引擎上的預測結果（5個種子，平均 ± 標準差）"),
  table([1300, 1000, 1000, 1022], [
    ["正規化方式", "全視窗RMSE", "截斷點RMSE", "截斷點Score"],
    [{ text: "FD002", span: 4, start: 0 }],
    ["全域min-max", "15.71 ± 0.52", "16.29 ± 1.01", "190.0 ± 39.2"],
    ["全域z-score", "15.83 ± 0.39", "17.12 ± 2.51", "289.0 ± 147.7"],
    ["工況感知min-max", "12.49 ± 0.43", "12.07 ± 0.96", "108.6 ± 19.8"],
    ["工況感知z-score", "12.54 ± 0.58", "14.54 ± 1.13", "171.9 ± 41.8"],
    [{ text: "FD004", span: 4, start: 0 }],
    ["全域min-max", "20.20 ± 1.23", "21.40 ± 1.71", "421.9 ± 101.6"],
    ["全域z-score", "17.86 ± 0.32", "18.07 ± 1.01", "339.2 ± 63.0"],
    ["工況感知min-max", "15.32 ± 0.31", "14.65 ± 0.47", "193.5 ± 32.8"],
    ["工況感知z-score", "15.50 ± 0.72", "15.73 ± 1.08", "204.4 ± 73.9"],
  ]),
  new Paragraph({ spacing: { before: 40, after: 120 }, children: runs("註：全域min-max為DAST原文的做法。視窗長度60，梯度裁切與暖身皆關閉。", { size: SMALL }) }),
  para("表3比較四種正規化方式。各設定皆以驗證RMSE選模，再於保留引擎上評估。兩種工況感知正規化在兩個資料集上的平均誤差皆低於兩種全域正規化：以全視窗RMSE而言，FD002的工況感知方式約為12.5，全域方式為15.71至15.83；FD004的工況感知方式為15.32至15.50，全域方式為17.86至20.20。截斷點RMSE與Score呈現相同的排序。"),
  para("最直接的比較是工況感知min-max與全域min-max，兩者只差在是否依工況分別估計正規化統計量。以同一隨機種子配對，截斷點RMSE的平均差值在FD002為−4.22（95%信賴區間[−6.24, −2.21]），在FD004為−6.76（[−9.26, −4.25]）；Score的差值分別為−81.4（[−146.1, −16.6]）與−228.5（[−386.2, −70.7]），信賴區間皆不含零。工況感知z-score的改善在FD004同樣明確，相對全域min-max的截斷點RMSE差值為−5.67（[−8.51, −2.84]）；在FD002，其全視窗RMSE較全域z-score低3.30（[−4.34, −2.25]），但截斷點RMSE的信賴區間跨過零（−2.59，[−5.84, +0.67]）。"),
  para("工況內採用何種尺度轉換的影響則隨評估方式而不同。以全視窗RMSE而言，工況感知z-score與min-max幾乎相同（FD002相差0.05，FD004相差0.18）；以截斷點評估時min-max較佳，z-score減min-max的RMSE差值在FD002為+2.47（[+0.76, +4.17]），在FD004為+1.08（[+0.02, +2.15]）。截斷點評估每個種子只有38至39個評估點，變異較大，因此本文將此差異視為初步觀察。全域min-max與全域z-score之間也沒有一致的優劣：FD004的全域z-score較佳（截斷點RMSE差值−3.33，[−5.45, −1.21]），FD002則無明確差異。整體而言，影響預測誤差的主要因素是是否依工況分別正規化，而非全域正規化的形式，這與第3.3節的推論一致：全域min-max與全域z-score皆為仿射轉換，都無法移除工況造成的群間差異。上述信賴區間僅基於5個種子，作為效果大小的描述，而非顯著性檢定。"),

  h2("5.3 學習率暖身與梯度裁切的敏感度（RQ3）"),
  caption("表4. 工況感知z-score下不同梯度裁切門檻c與暖身長度w的保留集全視窗RMSE（視窗長度40，3個種子，平均 ± 標準差）"),
  table([900, 1140, 1140, 1142], [
    ["c \\ w", "w = 5", "w = 10", "w = 15"],
    [{ text: "FD002", span: 4, start: 0 }],
    ["0.5", "14.04 ± 0.28", "13.46 ± 0.37", "14.13 ± 0.23"],
    ["1", "14.21 ± 0.33", "14.21 ± 0.11", "14.10 ± 0.61"],
    ["2", "14.38 ± 0.41", "13.89 ± 0.48", "13.99 ± 0.38"],
    ["5", "13.83 ± 0.70†", "14.19 ± 0.22", "13.57 ± 0.52"],
    [{ text: "FD004", span: 4, start: 0 }],
    ["0.5", "15.31 ± 0.54", "15.35 ± 0.29", "15.65 ± 0.41"],
    ["1", "15.40 ± 0.22", "15.32 ± 0.71", "15.18 ± 0.38"],
    ["2", "15.24 ± 0.13", "15.52 ± 0.32", "15.74 ± 0.43"],
    ["5", "14.98 ± 0.24†", "15.23 ± 0.52", "15.04 ± 0.11"],
  ]),
  new Paragraph({ spacing: { before: 40, after: 120 }, children: runs("註：† 表示依驗證RMSE選出的設定。", { size: SMALL }) }),
  para("表4列出在工況感知z-score下，12組梯度裁切門檻與暖身長度組合在保留引擎上的RMSE。FD002各組合的平均RMSE介於13.46至14.38，FD004介於14.98至15.74；同一組合在不同種子之間的標準差為0.11至0.71。組合之間的差距約為種子間標準差的1至2倍，且沒有任何組合在兩個資料集上皆為最佳：FD002的最低平均值出現在c = 0.5、w = 10，FD004則為c = 5、w = 5。以邊際平均觀察，FD004在c = 5時的平均為15.08，低於其他門檻（15.30至15.50）；但在FD002，c = 0.5與c = 5的邊際平均相近（13.88與13.86）。暖身長度的邊際平均在FD002依w = 5、10、15分別為14.12、13.94、13.95，在FD004則為15.23、15.35、15.40，兩個資料集的方向並不一致。"),
  para("若依驗證RMSE選擇設定，兩個資料集都會選出c = 5、w = 5，其保留集全視窗RMSE分別為13.83 ± 0.70（FD002）與14.98 ± 0.24（FD004）；截斷點RMSE為13.75 ± 1.01與15.18 ± 0.33，Score為166.7 ± 69.9與168.1 ± 10.5。整體而言，在工況感知正規化之下，DAST對梯度裁切門檻與暖身長度的選擇相對不敏感，本實驗未觀察到一致的額外效益。需要注意的是，此網格中梯度裁切與暖身皆為開啟狀態，且視窗長度為40；表3雖有兩者皆關閉的結果，但視窗長度為60，兩個條件同時改變，因此無法由兩表的差異推論梯度裁切或暖身的效果。此外，本研究未記錄梯度範數等訓練過程指標，因此不據此推論兩者對訓練穩定性的影響。【可補充：RAdam的設計即在降低訓練初期自適應學習率的變異，可能減少暖身的邊際效益；需補RAdam原始文獻，否則刪除本句。】"),

  h2("5.4 研究限制"),
  para("本研究有以下限制。第一，C-MAPSS為模擬資料，結果能否推廣至實際設備仍待驗證。第二，本研究只使用DAST一種架構。第三，保留集僅有39台（FD002）與38台（FD004）引擎，RQ2與RQ3分別只有5個與3個隨機種子，截斷點評估每個種子也只有38至39個評估點，因此只報告描述性統計與信賴區間，未進行顯著性檢定。第四，本文未使用官方測試集，也不與文獻數值直接比較。第五，本研究早期曾以官方測試集選擇epoch進行探索性實驗，該批結果未納入本文的主要結果。第六，RQ2與RQ3使用不同的視窗長度（60與40）。"),

  h1("6. 結論"),
  para("本研究在固定DAST架構的前提下，探討多操作工況對RUL預測輸入的影響。針對RQ1，在FD002與FD004中，工況切換幾乎完全主導視窗內的感測器變異，而全域正規化因屬仿射轉換無法移除此影響；工況感知正規化後，工況解釋的變異大幅降低，感測器與RUL的相關性明顯提高。針對RQ2，兩種工況感知正規化在FD002與FD004保留引擎上的誤差皆低於DAST原文採用的全域min-max與全域z-score；其中工況感知min-max相對全域min-max的改善，在兩個資料集與兩項指標上皆穩定。工況內採用z-score或min-max的差異則依評估方式而異。針對RQ3，在工況感知正規化之下，梯度裁切門檻與暖身長度的選擇對保留集RMSE的影響與種子間變異相當，未呈現一致的額外效益。未來將以官方測試集、更多隨機種子、與RQ2相同視窗長度下的梯度裁切與暖身對照，以及更多模型架構進一步驗證。"),

  h1("參考文獻"),
  ...[
    "1. Cheng, Y., C. Wang, J. Wu, H. Zhu, and C.K.M. Lee, “Multi-dimensional recurrent neural network for remaining useful life prediction under variable operating conditions and multiple fault modes,” Applied Soft Computing, 118, 108507 (2022).",
    "2. de Pater, I. and M. Mitici, “Developing health indicators and RUL prognostics for systems with few failure instances and varying operating conditions using a LSTM autoencoder,” Engineering Applications of Artificial Intelligence, 117, 105582 (2023).",
    "3. Huang, Z., Y. He, and B. Sick, “Spatio-temporal attention graph neural network for remaining useful life prediction,” Proceedings of the 2023 International Conference on Computational Science and Computational Intelligence (CSCI), 【頁碼待補】 (2023).",
    "4. MP, P.K., Z.-J. Gao, and K.-C. Chen, “Time series-based sensor selection and lightweight neural architecture search for RUL estimation in future Industry 4.0,” IEEE Journal on Emerging and Selected Topics in Circuits and Systems, 13(2), 514-【末頁待補】 (2023).",
    "5. Ramasso, E. and A. Saxena, “Performance benchmarking and analysis of prognostic methods for CMAPSS datasets,” International Journal of Prognostics and Health Management, 5(2), 1-15 (2014).",
    "6. Saxena, A., K. Goebel, D. Simon, and N. Eklund, “Damage propagation modeling for aircraft engine run-to-failure simulation,” Proceedings of the International Conference on Prognostics and Health Management, 1-9 (2008).",
    "7. Vaswani, A., N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in Neural Information Processing Systems, 30 (2017).",
    "8. Wang, Y., M. Ragab, Y. Hou, Z. Chen, M. Wu, and X. Li, “Deep domain adaptation for turbofan engine remaining useful life prediction: methodologies, evaluation and future trends,” arXiv preprint, arXiv:2510.03604 (2025).",
    "9. Zhang, Z., W. Song, and Q. Li, “Dual-aspect self-attention based on transformer for remaining useful life prediction,” IEEE Transactions on Instrumentation and Measurement, 71, 2505711 (2022).",
  ].map((t) => new Paragraph({ alignment: AlignmentType.LEFT, indent: { left: 300, hanging: 300 },
    spacing: { after: 40 }, children: runs(t, { size: 18 }) })),
];

const english = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Effect of Operating-Condition-Aware Normalization on DAST-Based Remaining Useful Life Prediction under Multiple Operating Conditions", bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("【Author Name】*", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("【Affiliation and address】", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs("*Corresponding author’s e-mail: 【to be completed】", { size: 24 }) }),
  h1("ABSTRACT"),
  para("Sensor measurements of aero-engines operating under multiple conditions reflect both degradation and operating context, which makes data-driven remaining useful life (RUL) prediction difficult. Using the NASA C-MAPSS dataset and the Dual-Aspect Self-Attention Transformer (DAST), this study examines operating-condition-aware normalization while keeping the model architecture unchanged. Operating conditions are identified by K-means clustering of the operating settings of training engines, and sensors are normalized within each condition by z-score or min-max scaling. All statistics are estimated from training engines only, and models are selected on engine-disjoint validation engines. In FD002 and FD004, operating-condition switching explains about 99.9% of the sensor variance within a 40-cycle window, a proportion that global normalization cannot change; after condition-aware normalization it drops to 0.21–0.22, and the mean absolute correlation between sensors and RUL rises from about 0.11 to above 0.40. Evaluated at one random cut point per holdout engine, condition-aware min-max scaling lowers RMSE by 4.2 and 6.8 cycles on FD002 and FD004 relative to the global min-max scaling used by DAST, with 95% confidence intervals of the seed-paired differences excluding zero. Under condition-aware normalization, differences in holdout RMSE across 12 gradient-clipping and learning-rate-warmup settings are about one to two times the seed-to-seed standard deviation, showing no consistent additional benefit.", { noIndent: true }),
  para("Keywords: remaining useful life, predictive maintenance, operating condition, normalization, Transformer", { noIndent: true }),
];

// ---------- document ----------
const page = { size: { width: 11906, height: 16838 },
               margin: { top: 1418, bottom: 1418, left: 1418, right: 1418 } };
const doc = new Document({
  styles: { default: { document: { run: {
    font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "新細明體" }, size: BODY } } } },
  sections: [
    { properties: { page }, children: front },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: 425 } }, children: body },
    { properties: { page, type: SectionType.NEXT_PAGE }, children: english },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "CIIE2026_fullpaper_v2.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, buf.length, "bytes");
});
