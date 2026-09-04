# 預註冊:integer-alibi(v1,已鎖定)

**v1 鎖定:2026-08-13(Asia/Taipei)。任何 backend 比較量測皆未開始。** v0 草案經作者核准(2026-08-13);v0→v1 變更僅二:(1) §0 填入 artifacts SHA;(2) P1 依理論精確化拆為 P1a/P1b(見該節註記)——此為預資料修正,使主張更嚴格可否證,非放寬。修改治理:A 類(預資料)自由記錄於文末;資料後修改=協定偏離,須如實報告。

## 0. 鎖定的 artifacts(SHA-256[:16])

| artifact | SHA | 說明 |
|---|---|---|
| `harness/p1_predictions.py` | `ef00d53a25687db2` | fp64 精確模擬,產出逐層預測 |
| `artifacts/calib_prompts.json` | `2e965138f8d8942d` | 64 個 320-token 窗口(wikitext-2-raw-v1 test,revision b08601e0,GPT-2 tokenizer,確定性推導) |
| `artifacts/p1_predictions_qwen3-1.7b.json` | `ff1f30c8de9d7651` | **P1 逐層預測:196/196 層 BITWISE-SAFE**(max acc 2,374,517 < 2²⁴;headroom 最小 2.82 bits、中位 6.34 bits;bound_also_safe 0 層——安全主張為 observed-activation 條件式) |
| checkpoint `qwen3-1.7b-int8-w8a8` | `bc6258648cc6c380` | llm-compressor W8A8(channelwise W、dynamic per-token A);parent `Qwen/Qwen3-1.7B` revision `70d244cc86ccca08` |
| checkpoint `qwen3-1.7b-fp8-dynamic` | `69d137b187ef43fc` | FP8-DYNAMIC,同 parent |
工作標題方向:*Determinism by Algebra: Integer Quantization as a Reproducibility Guarantee for LLM Inference*(暫定,寫作期再定)。

## 0. 一句話主張

浮點推論的位元可重現性需要工程購買(batch-invariant kernels、吞吐代價);整數量化路徑的 GEMM 可重現性由代數免費保證——本文以逐層可否證的預測驗證這個保證在真實 serving 堆疊中的成立範圍,並定位所有例外的來源層。

## 1. 分歧預算分類法(理論,量測前寫死)

對每條路徑,指認「最早可合法分歧的運算」:

| 路徑 | GEMM 乘積 | 累加 | 最早合法分歧點 | 預測 |
|---|---|---|---|---|
| INT8-W8A8 | 精確(int×int) | INT32,精確、順序無關 | **epilogue**(scale 套用) | 跨 kernel/batch/tile 順序:pre-epilogue 必位元一致 |
| FP8-W8A8 | 精確(FP8 乘積可表示於 FP32) | FP32,順序相關 | **累加順序**(split-K/tile) | 跨 kernel 必分歧,幅度 ~√K |
| W4A16(Marlin 類) | dequant 後 FP16 乘 | FP16/FP32,順序相關 | dequant 之後全部 | 跨 kernel 必分歧 |
| W4A8(QServe 類,若納入) | int4→int8 dequant 後整數 | INT32 精確 | epilogue(同 INT8)+dequant 查表 | 中間類:GEMM 段精確 |

## 2. 預先計算的可否證預測

**P1(2²⁴ 相變,逐層)——v1 精確化為兩層**。理論註記:|acc| < 2²⁴ 保證累加值可被 FP32 **精確表示**,但 epilogue 的 scale 乘法本身仍是浮點捨入運算,不同 backend 的乘法順序((acc·s_w)·s_a vs acc·(s_w·s_a))或中間精度可以合法地不同。故:
- **P1a(無條件定理檢定)**:建構 **probe checkpoint 變體**——weight scales 改寫為最近的 2 的冪次、activation 量化改為 static per-tensor 且 scale 設為 2 的冪次(品質無關緊要,此為 conformance 探針)。此時全部 epilogue 乘法在 FP32 中精確 ⇒ 對 BITWISE-SAFE 層,輸出**必須跨 backend 位元一致,無任何例外**。違反 = kernel bug,直接升級為主結果。
- **P1b(真實 scale 條件式)**:原始 checkpoint 下,BITWISE-SAFE 層的任何跨 backend 分歧**只能**來自 epilogue 捨入順序/精度,且幅度受限於每次乘法 ≤1 ulp(全程 FP32 時合計 ≤2 ulp 級);超出此界 = bug。分歧存在與否、及其 ulp 級幅度,均為照報的量測結果。
程序:量測前以 fp64 精確模擬(int 值 < 2⁵³ 無誤差)在釘住 prompts 上逐層記錄 max|acc|,**發布逐層預測清單**(已完成,SHA 見 §0:196/196 層 BITWISE-SAFE)。
判定:P1a 違反或 P1b 超界 = **kernel bug report**(主結果);全部通過 = 「整數路徑位元可重現性在實測堆疊成立」(主結果)。

**P2(K-縮放律)**:FP8/W4A16 路徑的跨 kernel 分歧 RMS 隨 reduction 深度 K 按 ~√K 成長(隨機捨入模型);系統性偏離(如 ~K)指向非隨機的順序結構。獨立 GEMM 呼叫掃 K∈{512..32768}。

**P3(探針歸因)**:P1a 的 probe checkpoint 同時是歸因工具——真實 scale 下觀察到的任何分歧(P1b),在 probe checkpoint 下:分歧消失 = 來源為 scale 乘法捨入順序;仍存在 = 來源為非 scale 的融合/順序差異(或 bug)。每個觀察到的分歧都必須被歸因到此二類之一。probe checkpoint 的建構腳本與 SHA 於使用前以 A 類修訂補入 §0。

**P4(傳遞函數)**:token 翻轉率可由 logit 分歧分布 × greedy margin 分布預測。以 teacher-forcing 量 per-step logit delta(與自由生成的級聯放大**分開報告**,HEAL 式控制),用 margin 分布推翻轉率,對照實測一致率(`quality_gates.py` 計分)。任務層差異用 McNemar 檢定。

**P5(端到端誠實定量)**:attention 與 LM head 仍為浮點 ⇒ 整數 GEMM 只消除部分端到端分歧。量化「GEMM 整數化消除的分歧占比」,明示剩餘來源(attention 累加、RoPE、LayerNorm)。

## 3. 設計矩陣

- **引擎**:vLLM(digest 釘住,主研究同版)+ SGLang(釘 commit)——雙引擎預堵審稿意見。
- **Backend 路徑**:每格式 2–3 條(CUTLASS / Triton / Marlin 系,以 `VLLM_DISABLED_KERNELS` 等機制逼出;每 run 從 log 確認實際 kernel 並記錄)。
- **模型**:Qwen3 家族兩尺寸(1.7B、8B;4090 24GB 內),量化 recipe 同主研究紀律(llm-compressor,校準集釘住)。
- **層級**:(a) 獨立 GEMM 呼叫(shape 取自兩模型的真實層幾何);(b) 逐層(hook 擷取);(c) 端到端 greedy。
- **控制**:GPU 時鐘鎖定;batch ∈ {1,4,16}(整數路徑預測 batch 不變性——順序無關 ⇒ batch 只改 tile 順序);seed 固定;teacher-forcing 與自由生成分開;正逆序兩輪。
- **硬體**:RTX 4090(sm_89)為主——**設計選擇而非道歉**(Marlin 為 Ampere 調校、Machete 瞄準 Hopper,消費卡是 kernel 成熟度落差最大且無人量測的一層)。可選擴充:租 1 小時 sm_86/sm_90(~$2)驗「跨世代位元一致」格。

## 4. Null 承諾(預先寫死)

- P1 全部命中、觀察不到任何 INT8 分歧 → 正面結果:「整數路徑的位元可重現性在實測堆疊成立」,照發。
- P1 違反 → bug report + 歸因(P3),升級為主結果。
- P2 偏離 √K → 順序結構分析,照報。
- P4 傳遞函數不準 → 照報差距與假說。
- **不存在使本研究「失敗」的結果**;唯一中止條件是第一週檢查點(backend 逼出的工程摩擦超標)。

## 5. 對外定位(寫作期執行)

- Related work 定位表:對六篇最近鄰(Silent Hyperparameter 2605.19537、HEAL 2606.21023、2506.09501、Thinking Machines batch-invariance、2607.11368、LiquidGEMM)逐格標明本文的新格。2607.11368 自承的 kernel/記憶體混淆由本設計的「同 checkpoint 同流量、只換 kernel」直接關閉——引為動機。
- Conformance suite 以可安裝工具發布(CI 可掛)。
- 投稿:arXiv v1(資料完成後)→ TMLR,申請 Reproducibility Certification;TMLR 稿內自引一律第三人稱。

## 6. 時程與上限

四週上限:W1 harness+逼出路徑驗證+P1 預測清單發布(檢查點);W2 GEMM 層+逐層量測;W3 端到端+P4;W4 分析+寫作啟動。超出即砍(先砍 W4A16,再砍 8B 尺寸,P1/P2/P3 為不可砍核心)。

---

### A-1(A 類,2026-08-13)
§0 之 p1_predictions.py SHA 由佔位補為實值 `ef00d53a25687db2`(鎖定 commit 91ca5e1→0e28d43 之管理性補填,無內容變更)。

### A-2(A 類,2026-08-13,於任何 probe 量測之前)
P5 操作化與 P3 probe checkpoint 具體化。**Probe 一級**:複製 INT8-W8A8 checkpoint,僅將全部 `weight_scale` 逐元素改寫為最近的 2 的冪次(log 空間四捨五入),其餘不變;SHA 於建構後補記於下。數學依據:乘以 2^k 僅移指數、與捨入可交換,故 `round(acc·s_a)·2^k ≡ round(acc·s_a·2^k)`——epilogue 乘法順序在 pow2 weight scale 下不影響結果(本模型無 linear bias,無加法洩漏)。預測(先於執行寫死):
- **E1**:probe 下 vLLM-CUTLASS vs vLLM-Triton 端到端 greedy **必須 8/8 位元一致**(兩臂共用同一 act-quant op ⇒ 相同 int8 輸入與 s_a;pow2 使乘序無關)。violation ⇒ 分歧定位到「兩臂 act-quant 實作不同」,啟動二級 probe(靜態 pow2 act scales)。
- **E2**:probe 下同臂 prefill-重放 vs decode-生成的 rail 差異**應持續存在**(來源為 attention 浮點 reduction 的 M-regime 依賴,非 GEMM)——E1 消除量與 E2 殘留量即 P5 的線性路徑/attention 份額分解。
- **E3**:probe 下 SGLang vs vLLM **應仍分歧**(跨引擎差異不止 GEMM)。邊界陳述:pow2 修復 kernel 互換的可重現性,非引擎互換。

### A-3(A 類,2026-08-13)
Probe 一級 checkpoint 建構完成:`models/qwen3-1.7b-int8-w8a8-pow2`,573,440 個 weight_scale 值改寫,合併 safetensors SHA-256[:16] = `c147608401cd8f02`。建構先於 E1–E3 任何量測;E1–E3 預測已於 A-2 鎖定。

### A-4(A 類性質但於資料後記錄,2026-08-13,Codex 數學審核後的更正)
本修訂為**量測後**的表述與方法更正,不改動任何已鎖定的預測內容;E1–E3 資料於 A-2/A-3 之後、本修訂之前收集。更正項:
1. **M2 精確界**:INT32 值於 FP32 中「所有整數皆精確」的保證界為 **|acc| ≤ 2²⁴**(非 <)。|acc| > 2²⁴ 不代表個別值必不精確,僅失去全稱保證。程式 `p1_predictions.py` 用嚴格 < 判定,屬保守方向,判定結果不受影響。
2. **M3 適用條件**:pow2 等價 `round(x·2^k)=round(x)·2^k` 僅在**有限 normal 值、無 overflow/underflow、兩臂使用相同 FP32 運算與相同 bf16 捨入語意**下成立。E1 的 8/8 觀察與此相容,但該等價不得無條件外推。
3. **P1b 上界表述**:原「2 bf16 ulp 上界 7.81e-3」不嚴謹——0.0078125 是 bf16 相鄰可表示值的最大相對間距(spacing)量級,非一般意義的 2-ulp 相對誤差界。嚴格判定改用 **bf16 位元距離(ulp_distance)**,以有序化 bit pattern 直接計算;relative-error 指標保留但近零值另行標記。
4. **P2 降格**:「確證 √K」改為「**7 點資料與 √K crossing model 相容**」;飽和值 50% 為經驗觀察,非理論天花板。後續以多 seed、nested-prefix 輸入、自由指數擬合(α 與 bootstrap CI,不預設 0.5)重測。
5. **E2 歸因限制**:pow2 probe 下的殘餘同臂分歧(2.9%)**不得**直接歸因為 attention,亦不得稱「attention 份額」;正確表述為「**非受控 INT8 scaled-MM epilogue 的殘餘份額**」(候選來源:attention reduction、未量化之 lm_head GEMM、KV cache 讀寫、RoPE/Norm)。歸因須待逐層 first-divergence instrumentation(工作項已排)。

### A-4 分類更正(2026-08-13,append-only)
A-4 原標記「A 類性質但於資料後記錄」**分類錯誤**:A 類依本文件定義為預資料修訂,A-4 於 E1–E3 資料之後作成,正確分類為**量測後 analysis/protocol amendment**(非本文件預先定義之 A/B 類;其效力限於指標與表述,不追溯任何已鎖定預測)。A-4 原文保留不動,以本則更正取代其分類宣稱。A-3 之 probe 建構先於 probe 量測,分類 A 類無誤;惟其記錄時點在 E1 執行指令發出後、判讀前,時序證據以 git log 為準。

### A-5(量測後 analysis-plan amendment,2026-08-13,Codex 審核 #1.5 後、重分析前固定)
本則記錄兩類內容,均為 append-only,不改動 A-1–A-4 任何原文:
**(一)A-4 未具體化、但於重跑前已在程式中固定的分析自由度**(git 時序 c01efd7→ccd0486 可證):near-zero 門檻 = 0.01×參照臂(CUTLASS)RMS;abs/RMS 僅報 max;relative 分位數僅取非近零已分歧元素;ULP bins 0/1/2/>2;unweighted linear SSE;CI90 與 2000 次 seed bootstrap;sat_preferred 之 0.5 SSE ratio 門檻;P4 之 even/odd 分割、10 bins、空 bin fallback、5000 次 bootstrap。
**(二)#1.5 審核後、任何重分析執行前固定的 v3 語義**(實作於 metrics_v2.py、w3_p2_refit_v3.py、w4_p4_stats_v3.py,19 項合成回歸測試全過):位元分歧(n_differ:=bit-pattern 不等,含 signed zero)與 finite-only 數值 ULP 距離分離,NaN/Inf/finite-mismatch 另計;數值 ULP 映射摺疊 signed zero(明文);近零區獨立子報告(計數、ULP 直方圖、絕對誤差分位數);abs/RMS 補 p50/p99 與 RMSE/RMS;K-模型全體共用 unweighted linear SSE 目標、保留 p=0、加入廣義飽和模型 p=1−exp(−cK^α);5-seed bootstrap 標記為 exploratory conditional interval,輔以 seed jackknife、leave-one-K-out、K-range sensitivity;不再使用誤設純冪律 CI 是否含 0.5 之二元判準;PR 指標改為 grouped-tie average precision(無正例回 None);配對改 (prompt,pos) 嚴格 join 並報告缺失/重複;<16 prompts 之 cluster CI 標 exploratory,LOPO 附 prompt ID 且僅作 influence diagnostic;calibration 合併重複分位邊界、bin 最低支援 20、Beta(1,1) 平滑、fallback 取最近 bin 並計數、報 per-bin 支援數、Brier 附 constant-prevalence baseline 與 prompt-bootstrap(exploratory)。
重分析僅使用既有 raw artifacts(w3_p2_fp8_v2.json 之 raw counts、tf*.json),新輸出使用新檔名(w3_p2_fits_v3.json、p4_stats64_v3.json、p4_stats8_v3.json),舊 artifacts 不覆寫。

### A-6(2026-08-13,8B 複製組之時序揭露與補釘;v3 重跑之前)
**揭露**:Qwen3-8B 之 P1 預測清單(`p1_predictions_qwen3-8b.json`,SHA `c103affd9556722e`,252/252 BITWISE-SAFE,parent revision 記錄於 models/parent8b_manifest.json)與其 per-layer verdict(v2 語義)由同一執行鏈生成——8B 的預測未先於驗證公開釘住,其 v2 verdict 之時序證據弱於 1.7B 主組,論文中標記為「複製組(同批生成)」。
**補釘**:本修訂於 v3 語義重跑之前將 8B 預測清單 SHA 釘住並推送;隨後之 v3 per-layer verdict(1.7B 與 8B)具備「預測公開先於驗證」時序,為論文之權威版本。8B E1(probe 端到端):16/16 位元一致;raw 對照 0/16——與 1.7B 主組一致。

### A-7(2026-08-13,Codex 覆核 #2 後之治理更正;append-only)
1. **時序揭露**:v3 分析計畫(A-5)、v3 程式、19 項測試與首批重分析輸出於同一 commit(95e2787)公開——Git 只能證明「同批公開」,不能證明 v3 語義在首批重分析執行前已不可變固定。此後採三階段 commit:計畫+程式+測試先 pin,執行,結果另 commit(本修訂本身依此執行;v3 GPU 重跑結果已單獨 commit 472591e,晚於其 pin commit)。
2. **8B parent 證據入 repo**:`experiment/artifacts/parent8b_manifest.json`(及 1.7B 之 parent_manifest.json)自本 commit 起受版本控制;A-6 之引用以 tracked artifact 為準。
3. **表述限定**(append-only,原文保留):「M_gensat 全面勝出」→「在四個預先列出的候選模型與共同 training SSE 下擬合誤差最低;α≈0.518 與 sqrt-shaped saturation 相容;未以 held-out 比較排除其他函數族,不單獨識別因果機制」(v4 起補 held-out predictive SSE)。「預測公開先於驗證/權威版本」→「v3 判準與 prediction SHA 於 v3 rerun 前公開固定;v3 為論文採用之指標版本;因同一 8B 資料之 v2 結果先前已知,此時序不構成獨立先驗複製證據」。cluster CI 之 16-prompt 門檻為分析自由度選擇,非治理辯護過的通用門檻。
4. **v4 修正版語義**(本 commit pin,執行於後續 commit):metrics_v3(有限參照 RMS、zero-RMS/no-finite-ref 顯式 status、無 JSON NaN、dtype/shape/device 防護、互斥有限性分類、join 重複鍵 fail-closed、prompt_sha 身分驗證)、w4_p4_stats_v4(尾端 bin 合併保證 ≥MIN_BIN_N、margin 距離 fallback、bin metadata)、w3_p2_refit_v4(α∈[0,1.5]、optimizer 診斷與 fail-closed、可識別性防護、四模型 LOSO/LOKO held-out SSE、不重疊 K 切分)。重分析輸出使用新檔名(*_v4)。

### A-8(2026-08-13,isotonic 敏感度分析之預固定;append-only,先於任何 isotonic 執行)
Decreasing isotonic(加權 PAVA)敏感度分析預固定如下:僅以 calibration prompts(偶數 index)之 merged-bin 機率擬合、以 bin cal_n 加權、evaluation 完全隔離;比較量 = held-out Brier 與 reliability curve;**角色限定為敏感度分析,binned-Beta(A-5)維持主方法,不因結果擇優替換**。實作於 w4_p4_stats_v4 之 --isotonic 旗標,本 commit pin,首次執行於後續 commit。

### A-9(2026-08-13,Codex 覆核 #3 後之治理更正;append-only)
1. **4d53c7f 例外承認**:該 execution commit 除結果外含兩個 post-pin 程式修正,其中 near-zero `abs_err_max` 修正**改變輸出欄位語義**(pin 版在近零集合為空時錯誤回退至全 pair max;執行版回 None)——commit message 稱「semantics unchanged」不準確,以本則更正。自本輪起嚴格執行:execution/result commit 不含 harness 修改;執行中發現 bug 即停止、修正、重新 pin、重頭執行。
2. **A-8 分類更正**:isotonic 敏感度分析係於 P4 原始資料與 v3 calibration 結果已知後制定,正確分類為 **post-data sensitivity-analysis amendment**,非 pre-data confirmatory plan。
3. **artifact 版本與模組檔名分離**:472591e 之 per-layer artifacts 由 w3_perlayer.py(其時 import metrics_v2)產生,「v3」指 #1.5 語義版本而非 metrics_v3.py 模組;該批 GPU 資料無非有限值,主結果不受本輪 metrics_v3 邊界修正影響。
4. **標籤治理**:seed CI 標籤自本輪起由程式依 len(seeds) 動態產生(<20 → exploratory;≥20 → primary-but-conditional),「conditional」與「exploratory/primary」分離;20 為分析自由度選擇。決策紀錄與 artifact 標籤不一致(前者稱卸下 exploratory、後者仍寫 5 seeds)以本則更正,以動態標籤為準。
5. **本輪 v5 語義(pin 於本 commit,執行於後續 commit)**:metrics_v3(abs_over_rms 用全部 finite/finite pairs、main-only 另名、fp64 統計防 bf16 極值 overflow、nonfinite-rms-computation status、混合 schema fail-closed、tf_v4 envelope);refit(L-BFGS-B 真有界、邊界最適解合法、K>0/p∈[0,1]/count 一致/逐 seed 完整 K grid 防護、LOSO/LOKO fold expected/completed/failed 記帳與比較有效性旗標);p4 stats(恢復 Brier prompt-bootstrap CI[標 conditional]與 cal_flips、空/單類資料 graceful status、空群組合併根修、isotonic 排除 cal_n=0 + margin fallback 計數 + held-out curve、prompt 索引化 bootstrap);tf_v4(requested/actual 長度驗證、prompt/rails/model manifest 內嵌)。輸出 schema p4-stats-v5,新檔名 *_v5。測試:tests_metrics_v3(26)+tests_metrics_v4(全部覆核反例);exact lock requirements-test.lock。

### A-10(A 類,預資料,2026-08-14,先於 TMLR 輪任何量測)

TMLR 輪新增兩個實驗:**P5 positive control**(量測 §X 七項 conformance 檢查的靈敏度與漏檢率)與 **P6 pow2 干預代價**(精度與吞吐)。二者皆為 v1 已揭露之缺口(論文 §XI(6) 與 §XI(3)),v1 主張不因本輪結果改變;本輪若失敗,v1 的七項檢查降級為「未經靈敏度驗證的提議」,而非追溯修改 v1。

**本則只釘設計與預測矩陣。** 實作程式的 SHA-256 於 A-10.1 另行釘住,且依三階段規則(計畫+程式+測試先 pin → 執行 → 結果另 commit)執行;A-10.1 未 pin 之前不得開始量測。

#### P5:positive control

**注入點與理由。** 故障注入於 epilogue 與 accumulator 兩處。只注入 epilogue 不足以驗證檢查二與檢查三——那兩項斷言的是 alibi 的前提,必須以刻意違反前提的案例測試,否則其偽陰性率無從得知。

**故障目錄(固定八項,不得於執行後增刪)。** F1 scale 於 bf16 而非 fp32 相乘;F2 double rounding(fp32→bf16→fp32→bf16);F3 兩個 scale 相乘次序互換($a\cdot(s_w s_a)$ 對 $(a s_w)\cdot s_a$);F4 輸出轉型改為 truncation 而非 round-to-nearest-even;F5 epilogue 融合次序改變(fused 對 unfused);F6 強制 accumulator 溢位(以 $K$ 超出 (1) 之矩陣);F7 強制 accumulator 進入 fp32 時超過 $2^{24}$ 但不溢位 INT32;**F8 null fault**——語意等價的恆等改寫(例如乘以 1.0、重排無關指令),作為偽陽性對照,**預測為不得被任何檢查判為違反**。

**嚴重度階梯。** F1–F5 各以三個預先固定的幅度執行(最小可表示擾動、中位、最大),使靈敏度可對嚴重度作圖;F6/F7 為二元(觸發/不觸發前提)。

**預測矩陣(本則釘住,執行前公開)。** 對 8 故障 × 7 檢查共 56 格,逐格預先寫定「應觸發／不應觸發／不適用」。預測寫入 `experiment/artifacts/p5_prediction_matrix.json` 並於 A-10.1 釘 SHA。**每一格皆可否證**:預測應觸發而未觸發者計為漏檢,預測不應觸發而觸發者計為偽陽性。

**樣本與判準。** 於 Qwen3-1.7B 全部 196 層 × 64 個既有 prompt 執行;檢查「觸發」的判準沿用 v1 定義(檢查一至五為位元或界的可否證結果,檢查六七為容差),不得於執行後調整門檻。

**回報量。** 逐檢查之偵測率、漏檢率、F8 之偽陽性率;以及預測矩陣 56 格的命中率。**不設通過門檻**——本實驗的產出是靈敏度的數值,不是及格與否。

#### P6:pow2 干預代價

**精度。** wikitext-2-raw-v1 test 上的 perplexity,窗口集**與 `calib_prompts.json` 的 64 個窗口不重疊**(同一 revision `b08601e0`、同一 GPT-2 tokenizer、同樣 320-token 非重疊切法,取其後續窗口),$n$ 於 A-10.1 釘住。比較 base INT8 對 pow2 probe,報告差值與 prompt-cluster bootstrap 90% CI。

**吞吐。** prefill 與 decode 分別量測,batch 與 ISL 組合於 A-10.1 釘住,每點重複次數固定,回報中位與 IQR,並記錄同一 identity contract 的九旗標。

**無抽屜承諾。** 精度與吞吐**無論結果為何都全部回報**。若 pow2 造成可觀測的精度或吞吐損失,論文必須明說該干預不是免費的,且不得以「本研究不評估部署代價」迴避——本則即為評估該代價的承諾。

#### 新增 Null 承諾

1. 若 P5 顯示某項檢查對其預測應捕捉的故障漏檢率高,即如實回報該檢查靈敏度不足,不得改寫該檢查定義後重測並只報後者。
2. 若 P6 顯示 pow2 干預有實質代價,即如實回報,不得將 pow2 重新表述為「僅供診斷、不建議部署」以迴避代價數字——診斷用途的主張已在 v1 成立,本輪問的是部署代價。
3. F8 若被任何檢查判為違反,即為檢查存在偽陽性的直接證據,須與偵測率並列回報。

### A-11(A 類,預資料,2026-08-14,先於任何第二 pair 之量測)

TMLR 輪新增 **P7:第二組 kernel pair**。這是**新的處理變數**,不是補既有缺口:v1 的全部結論來自單一 pair(vLLM 的 CUTLASS 對 Triton INT8 scaled-MM),因此「本方法可套用於其他 pair」在 v1 是**未經檢驗的推論**,而非量測結果。論文 §X 已把措辭限定為「本研究的 scripts 已實例化此組 pair,其他 pair 需改造而非原樣執行」;P7 檢驗的正是這個限定能否放寬。

#### 什麼算「第二組 pair」——選取判準先釘,對象後釘

**硬性判準(三條全滿足才算)**:

1. **同一 engine build 內的兩條路徑**。必須是在單一固定 engine 內以組態切換的兩個 scaled-INT8 GEMM 實作。**跨 engine 的比較(例如 vLLM 對 SGLang)明確不算**——那會把 kernel、cache、graph 執行與 scheduler 預設綁在一起,正是 v1 §II 批評他人研究的那種混合處理。若以跨 engine 冒充第二 pair,本研究就失去它唯一的方法論資產。
2. **兩臂共用同一組量化 operand**。必須能證明兩臂吃到位元相同的 int8 權重與 activation、相同的 scale;v1 是靠兩臂共用 `ops.scaled_int8_quant` 建立此點,第二 pair 需以同等強度證明,不接受「應該相同」。
3. **兩臂的 kernel 選擇皆可自執行 log 擷取**,並通過既有九旗標 identity contract(含 treatment 旗標)。無法擷取實際選中之 kernel class 者不算,依 v1 既有規則 fail closed。

**候選(依優先序,不預設哪一個成立)**:SGLang 內部的 INT8 GEMM 路徑切換(v1 已有其 digest 釘住的映像與 `probe_sglang.json`、`sglang_out_{1,2}.json`,基礎設施現成);vLLM 於本研究釘住版本後新增的 INT8 路徑;其他開源 engine 內部具兩條 INT8 路徑者。

**若三條判準無一候選滿足**:本條款以**負面報告收束**——寫明找過哪些候選、各因哪一條判準不成立而排除,並保留 §X 現有的限定措辭。**不得改以跨 engine 比較替代後宣稱 P7 完成。** 此條是為了防止事後把判準放寬到剛好容納手上能跑的東西。

實際選定之 pair、engine digest 與切換機制於 **A-11.1** 釘住,先於任何 P7 量測。

#### 預先寫定的預測

alibi 是關於算術的主張,不是關於某一組實作的主張,因此若第二 pair 滿足上述三條判準,理論預測如下,且**每一項皆可否證**:

1. **pow2 scale 下逐層位元相同**。預測:全部層通過。
2. **真實 scale 下逐層差異 ≤ 1 個 bfloat16 spacing**。預測:全部層通過。
3. **pow2 probe checkpoint 恢復端到端逐位元組一致**。預測:通過。
4. **§X 七項檢查在第二 pair 上可原樣執行**(不需改造)。這一項預測**較弱**:v1 的 scripts 綁定 vLLM 的介面與 log 格式,故預期需要 adapter;預先寫定的預測是「**檢查的定義**不需改動,**取值的 adapter** 需要改動」,並在報告中明確區分改了哪一層。

**任一項失敗的處理**:先以判準二(共用 operand)為首要嫌疑,提出可否證的診斷(逐元素比對兩臂輸入的 int8 tensor 與 scale);若 operand 確實相同而預測仍失敗,則 v1 的定位主張在該 pair 上不成立,**須如實回報並限縮 v1 的普適性陳述**,不得以「該 pair 不符合我們的設定」事後排除。

#### 新增 Null 承諾

1. 第二 pair 的結果無論支持或不支持 v1,皆全部回報。若不支持,論文須明說 v1 的結論限於原 pair。
2. 不得在看過第二 pair 的結果後回頭調整判準或預測。判準與預測以本條款(及 A-11.1 的對象釘定)為準。
3. 若第二 pair 的端到端比較顯示**兩臂一致**(與 v1 的 0/8、0/16、0/64 相反),這是關於「pair 之間差異程度不同」的發現,須與 v1 並列陳述,不得僅報告支持 v1 的那一組。

### A-11 收束:負面報告(2026-08-14,依 A-11 自身條文;**未進行任何量測**)

依 A-11「若三條判準無一候選滿足,以負面報告收束」執行。五個唯讀視角勘查 engine 原始碼(SGLang、vLLM、TensorRT-LLM、LMDeploy、llama.cpp/ggml、MLC-LLM、TGI、ONNX Runtime、OpenVINO、PyTorch/torchao),**零個候選同時滿足 C1/C2/C3**。P7 不執行,論文 §X 保留現有限定措辭。

**SGLang(A-11 列為第一優先)比預期更早失敗。** CUDA 上的 per-token INT8 dense GEMM **只有一條**:`sgl_kernel` 的 `int8_scaled_mm`。兩條看似不同的路徑——`compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` 的 `CompressedTensorsW8A8Int8.apply_weights` 與 `w8a8_int8.py` 的 `W8A8Int8LinearMethod.apply`——呼叫的是**同一個 kernel 符號**,差別僅在 `x_q`/`x_scale` 的 `.view()` reshape。兩臂跑同一個實作,是**空的 pair 而非 pair**(C1 不成立)。決定性的結構事實:`server_args.py` 提供 `--fp8-gemm-backend` 與 `--fp4-gemm-backend`,**沒有 int8 的對應項**。其餘 SGLang 分支各因不同理由失敗:`int8_gemm_kernel.cu` 的 dispatch 只以 `sm_version` 與 `out_dtype` 為鍵、tile 選擇是純形狀啟發式(不可組態、C3 亦不成立);CPU/AMX 臂呼叫 `int8_scaled_mm_with_quant`,**在 kernel 內部自行量化 activation**(C2 直接違反);NPU 臂是硬體分叉;block-wise int8 與 per-channel int8 的 scale 粒度不同,operand 本就不同。

**vLLM 的第三條路徑存在,但因 C2 排除。** 已釘住的 0.27.1 build 內確有第三個 INT8 實作 `HummingInt8ScaledMMLinearKernel`,SM75+ 故 sm\_89 可達,`requirements/cuda.txt:35` 釘 `humming-kernels[cu13]==0.1.10` 因此官方映像內已安裝,可由 `VLLM_DISABLED_KERNELS` 或 `--linear-backend humming` 切換,並由同一行 `Selected %s for %s` 記錄(C1、C3 皆成立)。**因不共用 activation 量化路徑而 C2 不成立**,故排除。ROCm 的 `AiterInt8ScaledMMLinearKernel` 反而三條判準最乾淨(subclass CUTLASS kernel、呼叫同一個 `ops.scaled_int8_quant`),但 gate 在 ROCm 且 compute capability ≥ 90,4090 不可達;若日後借到 MI300 級 GPU,Aiter 對 Triton 是真正的第二 pair。CPU 的 Zentorch 對 CPUInt8 亦因 C2 失敗(前者在自有 fused op 內量化,後者用 `ops.onednn_scaled_int8_quant`,與 `ops.scaled_int8_quant` 不同)。

**其他 engine 的失敗歸為三類**:(a) 可執行路徑中根本沒有 scaled-INT8 w8a8 GEMM(TensorRT-LLM main 的 `get_quant_method` 無 int8 分支、兩個 scaled\_mm op 硬性要求 `kFloat8_e4m3fn`,plugin 樹已刪除;MLC-LLM 無 activation\_dtype int8 之 preset);(b) 只有一條實作(LMDeploy CUDA、TGI);(c) 兩條實作不可由組態於單一 build 內切換(llama.cpp 為 CMake 編譯期選項,兩臂在不同 binary;ONNX Runtime 是不同 ONNX operator,即不同的模型圖;TensorRT-LLM legacy SmoothQuant plugin 為 `trtllm-build` 期選擇,產生不同的 `.engine` 檔)。全勘查中唯一的執行期切換是 PyTorch Inductor 對 `aten._int_mm` 的 ATen 對 Triton lowering——**這是 library-level op pair,無 engine 環繞**,依 A-11 條文標記而不採用(v1 已在做逐層 op 比較)。

**明確拒絕的誘惑**:SGLang 的 `--fp8-gemm-backend`(choices 含 cutlass/triton/deep\_gemm/flashinfer\_\*)與 vLLM 的 FP8 `linear_backend` 集合,**機制上正是 C1 想要的形狀**——兩個實作、單一 build、一個開關,且 SGLang 的 FP8 路徑在分支**之前**完成 activation 量化(`fp8_utils.py:1745` 的 `apply_fp8_linear` 於 1806–1878 產生 `qinput`/`x_scale`,同一組值傳入 1887 的 `triton_scaled_mm` 與 1910 的 `fp8_scaled_mm`),故 C2 成立。但 A-11 的 C1 釘的是 **scaled-INT8(w8a8)**,而 Null 承諾 2 禁止看過結果後調整判準。**以 FP8 pair 充當 P7 即為判準放寬,不採用。** 若要做,須另立條款(A-12),並如實記載它是在 A-11 收束為負面報告之後才提出。

#### 本次勘查的附帶收穫(非 P7,但入證據層)

1. **釘住的 SGLang 版本已確定為 v0.5.17(CUDA 13.0 build)**。`sha256:16aba892…` 是 OCI image-index digest,由四個 tag 共同持有(`latest`、`latest-cu130`、`v0.5.17`、`v0.5.17-cu130`),推送時間 2026-08-08T00:09:39–44Z;GitHub tag `v0.5.17` → commit `2948168…`,release 發布於 2026-08-08T00:19:16Z,與推送相隔約十分鐘,與發版流程一致。`latest` 為 cu130 而非 cu129(後者 digest 為 `sha256:220bb1a1…`)。本機 `experiment/artifacts/run_logs/sglang_pull.log` 記錄該次 pull 的 digest,與登錄檔一致。**v1 論文只記 digest 未記版本號,此處補足。**
2. **pair 1 的來源反而被強化。** 兩臂共用 activation 量化在原始碼層獲得確認:`cutlass.py:133` 與 `triton.py:130` 是**逐位元組相同**的 `ops.scaled_int8_quant(x.contiguous(), i_s, i_zp, symmetric=symmetric)` 呼叫。
3. **「Triton 類 subclass CUTLASS 類」的疑慮已釐清。** `TritonInt8ScaledMMLinearKernel` 確實 subclass 自 `CutlassInt8ScaledMMLinearKernel`(`triton.py:27`),但它覆寫了 `process_weights_after_loading`(:40–119)與 `apply_weights`(:121–156),GEMM 本身是獨立實作(`compressed_tensors/triton_scaled_mm.py` 的 `triton_scaled_mm`)。subclass 只是共用權重前處理的介面,不是共用計算。
4. **v2 可補的揭露(非錯誤)**:§XI 現寫「two INT8 kernel implementations」,陳述的是研究對象而非存在數量,措辭正確;但可加一句說明**該 build 內實有三個可選 INT8 實作,第三個(Humming)因不共用 activation 量化路徑而未納入比較**,以免讀者把它讀成「只存在兩個」。

### A-10.2(A 類,預資料,2026-08-14,先於任何 P5 執行;補 A-10 故障目錄之缺陷)

實作 `p5_checks.py` 並逐格填寫預測矩陣時發現 A-10 的八項故障目錄有一個結構缺陷:**檢查一(shared operands)在該目錄下沒有任何應觸發的案例,因此它的偵測率無法量測。** F1–F5 與 F8 皆為 epilogue 內的擾動,兩臂 operand 相同;F6/F7 是前提違反,兩臂吃到同一組(刻意溢位的)operand。沒有一項會使兩臂的 operand 相異,而檢查一正是比對兩臂 operand 的那一項。若照原目錄執行,報告只能寫「檢查一在八項故障下零誤報」,而那是**把不適用講成通過**——正是 `p5_checks.py` 以 `applicable=False` 明確區分、不計為 pass 的那種混淆。

因此**新增 F9:operand 不匹配**。第二臂收到與第一臂不同的 int8 operand 或 scale,以三個嚴重度執行:(a) 單一 int8 元素差 1;(b) 單一 per-channel scale 差一個 float32 ulp;(c) 整個 activation tensor 以不同 seed 重新量化。**預測:檢查一必觸發**;後續各檢查對 F9 的預測逐格另填,並在報告中明確標示 F9 之下「其他檢查是否觸發」不代表那些檢查的偵測能力——因為檢查一失敗時,其餘檢查的結論本就不成立(此依存關係已寫入 `p5_checks.py` 檢查一的 docstring)。

目錄自本則起為**九項**,並自本則起封閉:**執行後不得增刪**。本次增補發生於任何 P5 量測之前,無資料可據以調整,故仍屬 A 類預資料修正;記錄於此以免日後被誤讀為事後擴充。

**同時記錄一個實作期的自我更正(非條款變更)**:`p5_checks.py` 初版的 ULP 有序映射對 ±0 給出 65536 而非 0,由回歸測試 `tests_p5_checks.py` 捕獲。查核後確認 **v1 的 `metrics_v3.bf16_ulp_distance_finite`(第 54–55 行)是正確的**——其 `torch.where(ia >= 0, ia + 2**15, 2**15 - (ia & 0x7FFF))` 將兩個帶號零同映至 32768,距離 0,故**論文報告的 max ULP distance = 1 不受影響**。`p5_checks.py` 已改為與該公式逐字相同,並加入一項測試直接比對兩個映射在探針張量上的輸出相等,使兩者由構造而非巧合保持一致。

### A-10.3(A 類,預資料,2026-08-14,先於任何 P5 執行;嚴重度階梯與可觀測性)

實作 `p5_inject.py` 時發現 A-10 的嚴重度規格與故障的實際性質不符,並發現一項必須寫入報告結構的限制。兩者皆在任何量測之前確定。

**一、嚴重度改為覆蓋率階梯。** A-10 原寫 F1–F5「各以三個預先固定的幅度執行(最小可表示擾動、中位、最大)」。實作後確認**這五項故障是類別性的而非純量性的**:bf16 相乘、雙重捨入、重結合、截斷、融合次序——沒有一項有幅度旋鈕,其效果就是算術給出的值。可控制的是**它觸及多少輸出元素**。因此改為 `one_element` / `one_percent` / `all_elements` 三級。這也更貼近讀者真正想知道的量:**多小的故障各項檢查還抓得到**。F6/F7(前提違反)與 F8(null)為二元,只跑一次;F9 的階梯本就是幅度性的(一個 int8 步 / 一個 float32 ulp / 整張重抽),維持不變。

**二、必須區分「檢查不靈敏」與「故障在輸出精度下不可觀測」。** bfloat16 只有 8 位 mantissa,任何低於其捨入步長的 fp32 擾動都會被吸收。實測(`observability_survey`,三個 regime):F1 差異 211/5719/1317 個元素、F2 為 141/4347/1080、F4 為 231/8285/2027,而 **F3(scale 重結合)在三個 regime 全部為 0**,F5 為 0/2/0。

F3 的零差異是算術事實而非注入器缺陷:fp32 重結合的相對誤差約 2⁻²³,bf16 捨入步長為 2⁻⁸,擾動低了約 15 個 binade,只有當值落在捨入邊界 2⁻²³ 之內才顯現(隨機資料下每元素約 2⁻¹⁵)。

因此 `inject()` 一律記錄 `n_output_differing`,而**報告必須把 `n_output_differing == 0` 讀為「該故障在輸出精度下不可觀測」,絕不可讀為「檢查不靈敏」**。此規則寫入 `p5_inject.py` 的模組註解與回歸測試(F3 的不可觀測性本身是一項測試,使日後若有人「修好」它會被注意到,而非默默改變靈敏度表的語意)。

**三、由此導出的一項限制,屬於論文而不只是 harness。** 檢查六(real-scale tolerance)存在**偵測地板**:通過它並**不建立 epilogue 等價**,只建立「任何差異低於 bf16 解析度加一個 spacing」。兩個 kernel 可以採用不同的 scale 結合順序或融合方式而輸出逐位元組相同。v1 論文的方向不受影響——v1 觀察到的是兩臂**確實**在真實 scale 下有差異(196 層中 188 層),即其 epilogue 差異足以被看見;但 §X 的 conformance 表若被讀成「檢查六通過即 epilogue 等價」則過度。此項列為 TMLR 輪必須寫入的限制,並在 P5 報告中以 F3/F5 的可觀測性數據支撐。

回歸測試:`tests_p5_checks.py` 37 項、`tests_p5_inject.py` 60 項,全部通過。

### A-10.4(A 類,預資料,2026-08-14,預測矩陣寫定後的必要揭露)

63 格預測矩陣已寫定並提交(`experiment/artifacts/p5_prediction_matrix.json`,產生器 `harness/p5_make_matrix.py`),於任何 P5 量測之前。統計:8 格預測觸發、46 格預測不觸發、9 格不適用、**5 格預測漏檢**。

逐格寫完後浮現一項必須事前揭露的結構事實:**八格預測觸發全部來自 F6/F7(前提違反)與 F9(operand 不匹配);五個 epilogue 故障 F1–F5 沒有任何一格預測會被任何檢查抓到。**

理由是算術性的而非實作性的。F1(scale 以 bf16 相乘)改變 scale 的相對量最多 2⁻⁹,約半個輸出 spacing;F2(雙重捨入)每步最多半個 spacing;F4(截斷取代 RNE)最多一個 spacing;F3/F5 已測為低於 bf16 解析度。因此**四者的輸出差異皆落在 ≤1 ulp,而檢查六的容差正是 1 ulp**,故全部通過。檢查五是位元檢查但只在 pow2 scale 下適用,而在 pow2 scale 下 F1/F2/F4 皆為 no-op(2 的次方在 bf16 內精確可表示,乘積亦精確),故亦不觸發。

**由此導出的預先陳述**:P5 預期顯示這套 conformance 檢查**能偵測前提違反與 operand 不匹配,但對停留在一個輸出 spacing 之內的 epilogue 實作差異沒有已證實的偵測力**。這與 v1 的觀察一致而非矛盾——v1 量到兩個真實 kernel 的差異正是 ≤1 ulp,即真實 kernel 的 epilogue 差異恰好落在此套件無法判為違反的區間。檢查六的容差本就是依 v1 觀察到的值設定的,所以它對 epilogue 缺陷沒有展示過的檢定力,這一點在 v1 未被指出。

**若量測推翻上述預期**(即某個 epilogue 故障確實使檢查六觸發),則本則的推理是錯的那一方,如實回報並以實測取代。**若量測證實**,則 TMLR 輪的 §X 必須改寫:七項檢查的定位由「判定兩個實作是否可互換」限縮為「判定 alibi 的前提是否成立、operand 是否共用,以及輸出差異是否超出一個 spacing」——後者是敘述性門檻而非缺陷偵測器。此改寫已在本則預先承諾,不待審稿人指出。

**同時揭露檢查七未被 P5 檢定**:它是 token 層的容差檢查,而 P5 是層級注入、不提供 margin 與 flip,故九列皆標 `not_applicable`,其靈敏度**不在本輪量測範圍內**。矩陣的 `scope` 欄位明文記載此事,以免七項檢查的計數被讀成七項都測過。

### A-10.1(PIN,2026-08-15;計畫+程式+測試釘定,量測尚未開始)

依 A-10「實作程式的 SHA-256 於 A-10.1 另行釘住,且依三階段規則執行;A-10.1 未 pin 之前不得開始量測」。本則所在的 commit **只含計畫、程式與測試,不含任何量測結果**;P5/P6 的執行結果將於後續各自獨立的 commit 提交。A-11(第二 kernel pair)已收束為負面報告,故本輪範圍為 P5 與 P6。

#### 釘定的 artifacts

| artifact | SHA-256[:16] |
|---|---|
| `harness/p5_checks.py` | `9d1a9d435782a539` |
| `harness/p5_inject.py` | `991a871dc01af5d8` |
| `harness/p5_make_matrix.py` | `31f2c4f80f82fd82` |
| `harness/p6_accuracy.py` | `4e340e2927b4135a` |
| `harness/p6_throughput.py` | `86d460be206f17b2` |
| `harness/tests_p5_checks.py` | `a1c26e6be004139a` |
| `harness/tests_p5_inject.py` | `50e7783683c29a83` |
| `harness/tests_p6.py` | `60404abfbfcc2ba1` |
| `harness/tests_p6_windows.py` | `da2ed0158375e5da` |
| `artifacts/p5_prediction_matrix.json` | `8f6ada4492c688b0` |

#### 釘定的參數(執行後不得調整)

- **P5**:故障目錄九項(A-10 八項 + A-10.2 之 F9),覆蓋率階梯 `one_element` / `one_percent` / `all_elements`(A-10.3),F6/F7/F8 為二元只跑一次。檢查容差沿用 v1 定義:`max_ulp = 1`、product bound 16256(Eq. (1) 的保守中層)、`max_flip_rate = 0.05`。預測矩陣 63 格。
- **P6 精度**:wikitext-2-raw-v1 test,revision `b08601e0`,GPT-2 tokenizer,320-token 非重疊窗口;跳過研究用的前 64 個,取其後 **256** 個(語料共 283,287 token / 885 個窗口,故僅用 36%)。NLL 以 **float64** 累加。bootstrap 10,000 次、以窗口為 cluster、**兩臂配對重抽**、seed 20260814、90% CI。
- **P6 吞吐**:batch×ISL 網格固定為 (1,128)、(1,2048)、(4,128)、(4,2048)、(16,128)、(16,512);OSL 64;warmup 2;repeats **7**(奇數,使中位數為實測值);回報中位與 IQR 並保留逐次序列;decode 秒數為完整跑的 wall time 減去同組的 prefill 中位數(明示為配對中位相減,非逐次相減)。

#### 測試狀態

四支回歸測試共 **134 項全部通過**,記錄於 `artifacts/run_logs/p5_p6_test_record_2026-08-15.log`:`tests_p5_checks.py` 37、`tests_p5_inject.py` 60、`tests_p6.py` 23、`tests_p6_windows.py` 14。窗口推導測試的第一項直接驗證 `derive_windows(0, 64)` **逐字重現已提交的 `calib_prompts.json`**,這是「評估窗口與研究用 prompt 不重疊」這項承諾的構造基礎;兩個守衛(span 重疊、推導漂移)亦各有一項測試確認會正確觸發。

#### 執行前的最後狀態

五份 checkpoint 已於 2026-08-14 重建並逐位元組驗證(`artifacts/rebuild_verification_2026-08-14.json`,5/5 digest 與 30/30 metadata 檔相符),故 P5/P6 所用的模型與 v1 量測所用的是同一批位元。

**自本 commit 起可以開始執行。執行 commit 不得含 harness 修改**;執行中發現 bug 即停止、修正、重新 pin、從頭執行(A-9 第 1 點)。


### A-10.6(**資料後**修正,2026-08-15;冒煙測試推翻兩項預測)

**分類必須說清楚:本則是資料後修正,不是預資料修正。** 觸發它的是 `p5_runner.py` 在 196 層中的**前 2 層**上的冒煙執行(`out/p5_smoke.json`,84 列,為驗證 runner 可運行而跑)。那是資料。因此本則依 A-9 第 2 點的先例(A-8 事後被正確重分類為 post-data)標記為資料後修正,而非偽裝成 A 類。冒煙執行**不構成 P5 的量測**,但它已經讓我看到兩格的結果,所以此後對那兩格的任何預測都不再是盲的。

**一、F4 × 檢查五:原預測「不觸發」錯誤,實測觸發。** 原推理為「pow2 下乘積精確可表示,故截斷與 round-to-nearest-even 一致」。該推理混淆了兩件事:`acc × 2^k` 在 **fp32** 內精確,但 `acc` 需 17 位 mantissa 而 **bf16 只有 8 位**,故轉型到 bf16 仍會捨入,截斷與 RNE 因此不同。更正後的預測為**應觸發**。

此更正的方向對本研究**有利**,這點必須明說以免被讀成事後美化:它意味著檢查五(pow2 regime 下的位元檢查)**確實能偵測錯誤的捨入模式**,故 A-10.4 所預告的「五個 epilogue 故障全數漏檢」在 F4 這一格上不成立。對照觀察:F1 與 F2 在 pow2 下確為 no-op,而那正是論文 Eq. (2) 的交換律在作用。因此更精確的陳述是:**pow2 regime 使 epilogue 對中間精度與捨入順序免疫,但不對錯誤的捨入模式免疫**;檢查五偵測後者。此陳述取代 A-10.4 的對應段落,但 A-10.4 原文保留。

**二、F9 需要逐嚴重度的預測。** 原矩陣為 F9 給單一格涵蓋三個嚴重度,而三級改動的對象不同:`one_element` 改一個 int8 元素、`all_elements` 重抽整張 activation(兩者都改變累加器),但 **`one_percent` 改的是一個權重 scale 的 float32 ulp,int8 張量未變**,故累加器相同、檢查四正確沉默。原預測「不同 operand 給出不同累加器」對前兩級成立,對第三級不成立。同理 `real_scale_tolerance` 在該級的實測為「max ulp distance 1(容差 1)」,落在容差內,是偵測地板的一個實例而非檢查缺陷。

因此 F9 的七格擴為逐嚴重度三組,共 21 格;矩陣總格數自 63 增為 **77**(其他八個故障不變)。

**三、報告義務。** P5 報告須同時呈現原預測與更正後預測,並揭露觸發更正的是冒煙執行的 2 層資料。偵測率須分別以「原矩陣」與「更正後矩陣」兩種分母各報一次,使讀者能自行判斷更正是否有利於結論。更正後的矩陣與 runner 於 **A-10.5** 一併釘定,釘定後才執行全部 196 層。

### A-10.7(**資料後**,2026-08-15;同一冒煙執行的第三、四項發現)

觸發來源同 A-10.6:`p5_runner.py` 在前 2 層的冒煙執行(`out/p5_smoke2.json`)。標記為資料後修正。

**三、F9 `one_element` × 檢查六:原預測「不觸發」錯誤,實測 max ulp 366–384。** 原推理為「約四分之一 spacing 的差異落在 1 ulp 以內」。錯在只看絕對變化:單一 int8 activation 元素改變一步,會使該列所有 N 個累加器項各變動至多 `|w[:,j]| ≤ 127`;而**部分累加器項因抵消而接近零**,對它們而言 127 是巨大的相對變化,故輸出可差數百個 spacing。更正為**應觸發**。檢查六對 operand 差異的偵測力因此比矩陣原先預期的強。

**四、結構性發現(本輪最重要,非單格更正)。** 冒煙資料顯示:**F1、F2、F3、F4 在所有可觀測嚴重度下的 max ulp distance 全部恰好等於 1**,而檢查六的容差正是 1,故全部判為 `correct_silence`。

這不是巧合或調參問題,而是構造上的必然:**單一捨入模式錯誤或單一中間精度縮減,最多只能把一個值移動一個可表示步長**(bf16 的 spacing 相對於任何 single-rounding 差異都是粗的)。因此容差為 1 的輸出比較檢查對這一整類 epilogue 缺陷是**結構性失明**;要偵測它們,容差必須為 0,也就是位元比較——而那正是檢查五在 pow2 regime 下所做的事,也正是它抓到 F4 的原因(A-10.6)。

對照組完成了這個論證:F6/F7 的 max ulp 為 **0**(兩臂相同、值卻是錯的),F9 `all_elements` 與 `one_element` 則遠超容差。所以檢查六能偵測 operand 差異,不能偵測 epilogue 缺陷,而這兩件事的界線是容差 1 本身。

**對 TMLR 稿件的後果,在此預先承諾**:pow2 干預的定位須由「決定性工具」擴為「**使 epilogue 缺陷可偵測的條件**」——它把一個容差問題轉為位元問題。同時必須明說 v1 觀察到真實兩臂「≤1 ULP」這件事**對是否存在 epilogue 缺陷沒有資訊量**,因為任何此類缺陷都會落在同一區間。此陳述取代 A-10.4 對檢查六的表述;A-10.4 與 A-10.6 原文均保留。

以上兩項的更正後預測連同 A-10.6 的兩項,一併於 A-10.5 釘定後才執行 196 層。冒煙輸出 `p5_smoke2.json` 不作為結果提交,但其觸發的四項更正已如上記錄。

### A-10.5(PIN,2026-08-15;取代 A-10.1,量測仍未開始)

A-10.1 的 pin 有兩個缺陷,均於嘗試執行時發現:**沒有 P5 runner**,而 `p6_accuracy.py` 在單一進程內同時推導窗口與跑推論,與機器實際的環境切分不符(vLLM 只在釘住的容器、`datasets` 只在量化 venv)。P6 首跑因此全部失敗並已存證(`artifacts/p6_first_run_record.txt`、決策日誌 2026-08-15 節),**未產生任何結果檔**,故 A-10.1 未曾用於任何量測,本則直接取代它。

本則所在的 commit **只含計畫、程式與測試,不含任何量測結果**。P5 與 P6 的結果將於後續各自獨立的 commit 提交。

#### 釘定的程式

| artifact | SHA-256[:16] |
|---|---|
| `harness/p5_checks.py` | `9d1a9d435782a539` |
| `harness/p5_inject.py` | `01485659ac9a80c9` |
| `harness/p5_make_matrix.py` | `4c96fe6f02b07363` |
| `harness/p5_runner.py` | `f0d42e873c6e9d4b` |
| `harness/p6_windows.py` | `f3b737700d49e28c` |
| `harness/p6_accuracy.py` | `f2316138a4f1c965` |
| `harness/p6_throughput.py` | `86d460be206f17b2` |
| `artifacts/p5_prediction_matrix.json` | `718e04cd93eb2653` |
| `harness/drivers/p6_run.sh` | `a54a87a96148184b` |

#### 釘定的測試(196 項全過)

| test | SHA-256[:16] |
|---|---|
| `harness/tests_p5_checks.py` | `a1c26e6be004139a` |
| `harness/tests_p5_inject.py` | `50e7783683c29a83` |
| `harness/tests_p5_accumulator.py` | `04b639bc4ef5020c` |
| `harness/tests_p5_runner.py` | `20a10f68af474672` |
| `harness/tests_p6.py` | `60404abfbfcc2ba1` |
| `harness/tests_p6_windows.py` | `599db740939ad87f` |

記錄於 `artifacts/run_logs/p5_p6_test_record_2026-08-15.log`:checks 37、inject 60、accumulator 9、runner 39、p6 邏輯 23、p6 窗口 28。

#### 釘定的參數(執行後不得調整)

- **P5**:九項故障;覆蓋率階梯 `one_element`/`one_percent`/`all_elements`(F6/F7/F8 為二元只跑一次);`M_TILE = 512`(整份捕獲,fp64 累加使其可負擔);`F5_TILE = (32, 32)`(其 (M,N,K) 中間值在真實形狀為 25 GB,逐列記錄該子塊);**兩個 scale regime**(`real` 與 `pow2`,後者用 `make_probe_pow2.py` 的同一變換);容差沿用 v1:`max_ulp = 1`、product bound 16256、`max_flip_rate = 0.05`;seed 20260815;預測矩陣 **77 格**。
- **P6 精度**:窗口由 stage A 在 venv 內推導(跳過研究用的前 64 個,取其後 256 個),附涵蓋 texts 與 spans 的內容雜湊;stage B 在釘住的容器內**先驗雜湊再計分**,不符即拒絕。NLL 以 float64 累加;bootstrap 10,000 次、以窗口為 cluster、兩臂配對重抽、seed 20260814、90% CI。
- **P6 吞吐**:網格 (1,128)、(1,2048)、(4,128)、(4,2048)、(16,128)、(16,512);OSL 64;warmup 2;repeats 7;回報中位與 IQR 並保留逐次序列。
- **執行環境**:所有碰模型的階段一律在 `vllm/vllm-openai@sha256:0a51ea5b…` 內,HF 快取唯讀掛載且容器離線(避免中途取得未釘住的輸入)。

#### 已知並已記錄的限制,執行前重申

1. 檢查七不在 P5 檢定範圍內(token 層,P5 是層級注入),矩陣十一格標 `not_applicable`。
2. F3 與部分 F5 在輸出精度下不可觀測,依 A-10.3 排除於分母之外而**不計為偵測**。
3. 三格為**資料後更正**(A-10.6、A-10.7),矩陣以 `corrected_post_data` 標記;報告須以原矩陣與更正後矩陣兩種分母各報一次偵測率,並揭露觸發更正的是兩層冒煙資料。
4. 靈敏度僅針對九種**我們自己選定形式**的合成故障,不等於對真實 kernel bug 的靈敏度。

**自本 commit 起可以執行。執行 commit 不得含 harness 修改**;執行中發現 bug 即停止、修正、重新 pin、從頭執行(A-9 第 1 點)。


### A-12(A 類,預資料,2026-08-15;新處理變數:量化時即採用 pow2 scale)

**P8:以 power-of-two scale 進行量化,而非事後捨入既有 checkpoint 的 scale。** 這是新的處理變數,不是補既有缺口。P6 已量出事後捨入版探針的代價為 **+157.4% PPL**(23.204 → 59.729,90% CI [+34.78, +38.29]),因此「pow2 能否作為可部署的決定性機制」目前是未解問題。P8 檢定它。

**時序與預測來源必須說清楚**:本條款寫於 P6 結果之後,其核心假說**由 P6 的資料導出**,故該假說不是盲的。這一點在此標明,不以「A 類預資料」掩蓋——A 類指的是 P8 本身的量測尚未開始,而非該假說未受既有資料影響。

#### 核心假說:+157% 的主要來源是截斷而非 pow2 約束本身

`make_probe_pow2.py` 用的是 `exp2(round(log2(s)))`,即**最近**的 2 次方,而那可能使 scale **變小**;scale 變小則 `w/s` 超出 [-127, 127] 而被截斷,產生的誤差與 pow2 約束無關。若改用**向上取整** `exp2(ceil(log2(s)))`,scale 永不小於原值,**在數學上不可能發生截斷**。

因此預測:**`ceil` 版的 PPL 代價將顯著小於 `nearest` 版的 +157%**。若量測推翻,則截斷不是主因,pow2 約束本身即代價高昂,如實回報。

#### 三個(必要時四個)臂,全部沿用 P6 的量測機具

1. `base`:既有 INT8 checkpoint(PPL 23.204,已量)
2. `pow2-nearest`:既有探針(PPL 59.729,已量;作為對照保留)
3. `pow2-ceil`:每個 per-channel weight scale 取 `exp2(ceil(log2(s)))`,權重以該 scale 重新量化(**不是只改 scale 欄位**,而是重新計算 int8 權重),故不截斷
4. `pow2-search`(條件執行):於 `k ∈ {ceil-2, ceil-1, ceil, ceil+1}` 內逐通道選擇使重建誤差 `||w - s·clamp(round(w/s), -127, 127)||_F` 最小者,允許以少量截斷換取較細解析度——即 RAPQ 一類構造的精神。**僅在 `pow2-ceil` 的代價仍不可接受時執行**,且該條件於此預先定義為「PPL 相對增幅 > 5%」;此閾值是執行與否的分流,**不是可部署性的判準**。

精度與吞吐一律沿用 P6 的釘定機具與參數(同 256 個不重疊窗口、同窗口內容雜湊、float64 NLL、窗口 cluster 配對 bootstrap 10,000 次 90% CI;同 batch×ISL 網格、OSL 64、warmup 2、repeats 7、中位與 IQR),使新數字與已公布的 +157.4% **直接可比**。

#### 其餘預先寫定的預測

- **跨 kernel 位元一致性必須成立,且這是算術而非經驗主張**:Eq. (2) 的交換律對**任何** 2 的次方成立,與該次方由 nearest 或 ceil 選出無關。故預測 `pow2-ceil` 在逐層比較上 **196/196 位元相同**,且端到端恢復位元一致。**若不成立,則交換律的適用條件在本實作中有我們尚未理解的破口**,那比精度結果重要得多,須優先追查而非歸為雜訊。
- **吞吐不變**:pow2 只改 scale 數值,不改執行的算術、形狀或 dtype。P6 已在 `nearest` 版上量到 ≈0%(prefill −7.79%…+0.64%),預測 `ceil` 版同樣 ≈0%。此項的價值在於它是一致性檢查:若量到顯著差異,應先懷疑量測而非機制。
- **`pow2-ceil` 的 scale 平均較原值大**,故量化步長較粗、解析度損失是其代價來源(與截斷無關)。預測其 PPL 代價為正但遠小於 157%;**不預測具體數值**。

#### 不設可部署性閾值

報告 PPL 差值與 90% CI、吞吐變化,並**明確拒絕宣告一個普適的「可接受」門檻**——可接受性取決於工作負載,由讀者對照自身容忍度判斷。P8 提供的是價格,不是許可。

#### 新增 Null 承諾

1. 三個(或四個)臂的結果全部回報,不論哪一個較佳。**不得僅因 `pow2-search` 數字較好而只報它**;`ceil` 是主要臂,`search` 是條件性補充,角色於此固定。
2. 若 `pow2-ceil` 的代價依然巨大,即如實回報「pow2 約束本身代價高昂」,不得以「可再調搜尋空間」延後結論。
3. 若位元一致性未成立,該結果優先於精度結果報告,且不得以「探針品質不佳」淡化——那會是對 §III 交換律論證的直接挑戰。

實作程式與其測試依三階段規則於 **A-12.1** 釘定;A-12.1 之前不得開始量測。

### A-12 增補(A 類,預資料,2026-08-15;先於任何 P8 量測)

寫執行路線時發現 A-12 的臂清單少了一個能**分離三種效應**的臂。既有探針把三件事混在一起,而 A-12 原本的設計無法把它們拆開:

| 臂 | scale 規則 | int8 權重 | 隔離出的效應 |
|---|---|---|---|
| `base` | minmax | 為該 scale 量化 | 基準(PPL 23.204) |
| `probe`(既有) | nearest pow2 | **未重算**,仍是為舊 scale 量化的 | 三效應混合(PPL 59.729) |
| **`requant-nearest`(新增)** | nearest pow2 | 重算 | 去除**權重與 scale 不匹配**,保留 scale 變粗與截斷 |
| `requant-ceil` | ceil pow2 | 重算 | 再去除**截斷**,只剩 scale 變粗 |

既有探針之所以混合,是因為 `make_probe_pow2.py` **只改寫 `weight_scale` 欄位而不重算 int8 權重**——那些權重是為原始 scale 量化的,搭配新 scale 使用時,反量化值 `w_int8 × s_new` 系統性偏離原權重。這個偏差與 pow2 約束無關,也與截斷無關,是第三種效應。無 `requant-nearest` 則無法判斷 +157% 中有多少來自它。

因此臂數自四增為五(含條件執行的 `pow2-search`)。**四個必跑臂的比較順序與其解釋在此固定**:`base → probe → requant-nearest → requant-ceil` 的 PPL 差分即為「不匹配」與「截斷」各自的貢獻估計;若三者無法把 157% 大致解釋完,則存在第四種尚未辨識的效應,須如實回報而不得歸為餘差。

其餘設計(量測機具、bootstrap、吞吐網格、位元一致性優先於精度、不宣告可部署門檻、三條 Null 承諾)一律沿用 A-12,不變。

### A-12.1(PIN,2026-08-15;P8 之程式與測試,量測尚未開始)

依 A-12「實作程式與其測試依三階段規則於 A-12.1 釘定;A-12.1 之前不得開始量測」。本則所在 commit **只含程式與測試,不含任何 P8 結果**。

#### 釘定的 artifacts

| artifact | SHA-256[:16] |
|---|---|
| `harness/p8_requant.py` | `010eda7602eb6ee9` |
| `harness/tests_p8_requant.py` | `8caff48c725be8ea` |
| `artifacts/quantization_convention_2026-08-15.json` | `67bccafb3347fe4a` |
| `artifacts/p8_report_minmax.json`(gate 的全 checkpoint 證據) | `acd9981ce0e9ecca` |

P8 的精度與吞吐量測**沿用 A-10.5 已釘定的 `p6_windows.py`、`p6_accuracy.py`、`p6_throughput.py` 與 `drivers/p6_run.sh`,零修改**,使新數字與已公布的 +157.4% 直接可比。位元一致性沿用 v1 的 `w3_perlayer.py` 與 token 比對路徑。

#### 等價性證明(這是本 pin 的核心)

`p8_requant.py --rule minmax` 產出的 checkpoint digest 為 **`bc6258648cc6c380…`**,與已提交的 base checkpoint **逐位元組相同**(`reproduces_base: true`,196 層全部)。因此「各 pow2 臂與 base 只差在 scale 約束」由重現證明而非論證支撐。

達成該重現需要三項慣例,皆為量測所得而非假設:除數 **127.5**(與 `compressed_tensors.quantization.utils.calculate_qparams` 的 `max_val_pos / (bit_range/2)`、`bit_range = 255` 一致,測試中即時比對而非寫死)、scale 儲存為 **bfloat16**、以及**量化的除法在 bfloat16 內進行**(改為 fp32 則 6.42% 的 int8 不同;此點以「fp32 除法**不得**重現 base」的形式釘成測試,避免日後被「最佳化」而靜默破壞等價性)。

#### 測試狀態:23 項全過(`artifacts/run_logs/p8_test_record_2026-08-15.log`)

含兩個真實層的 gate、bf16 除法的必要性、四個規則的 scale 皆為 2 的次方、`pow2_search` 逐通道取最小重建誤差、以及各臂的 int8 範圍與推得的乘積界。

#### 執行前確立的三項量測事實(影響假說解讀)

1. **base 本身即有截斷**:除數 127.5 使 `round(127.5) = 128` 超出 QMAX,196 層共 151,329 個元素。故 A-12 的核心假說須讀為**截斷量的變化**而非有無。
2. 單層(`layers.0.mlp.down_proj`)的截斷量:base **323**、`pow2_ceil` **11**、`pow2_nearest` **3,483**。方向與假說一致,但這是構造性質而非 PPL 結果,PPL 仍待量測。
3. 四個規則在該層的 int8 範圍皆為 [−128, 127],推得乘積界皆為 **16256**;各臂的界仍逐臂記錄,不假設相同。

**自本 commit 起可執行 P8。執行 commit 不得含 harness 修改。**


### A-12 範圍增補:8B 複製(A 類,預資料,2026-08-15;先於任何 8B 量測)

P8 的全部數字目前只有 Qwen3-1.7B。**pow2 量化的決定性是否在更大模型上成立,是一項未量測的主張**,而本研究的立場正是不接受「機制是算術所以應會轉移」這種推論代替量測。故於 Qwen3-8B 複製 P8 的三項載重量測。

**複製範圍(三項,對 `base-8B` 對 `requant-nearest-8B`)**:

1. **PPL**:沿用同一份 256 窗口檔(內容雜湊 `db8ffa33e242a6b2`;窗口是文字,與模型無關,故 8B 與 1.7B 的數字在同一評估集上可比)。8B base 的 PPL 尚未量過,一併量。
2. **逐層位元一致性**:252 層(8B 的層數),沿用 `w3_perlayer.py --stage verdict` 與已提交的 `perlayer_capture_8b`(253 檔)與 `p1_predictions_qwen3-8b.json`。
3. **端到端 token 序列**:沿用 `run_arm_generic.py` 與 `compare.py`,8 個 prompt × 64 token。

**不複製的兩項,連同理由**:`requant-ceil`(1.7B 上它是較差的臂,+0.54% 對 nearest 的 +0.32%;複製較差的臂不增加關於機制的資訊)、**吞吐**(1.7B 已量得為零效應,且機制上就該如此——pow2 不改變執行的算術、形狀或 dtype;在 8B 重量一次零效應的成本高於其資訊量)。二者於報告中明列為未複製,不得以「已在 1.7B 驗證」暗示已複製。

**預測**:三項皆與 1.7B 同向——逐層 252/252 位元相同、端到端 8/8、PPL 增幅為正但小(**不預測具體數值**;1.7B 為 +0.32%,8B 的層更深故 scale 分佈不同,無據以預測其量值)。

**若逐層或端到端在 8B 不成立**,則交換律的適用在更大模型上有我們尚未理解的破口,該結果**優先於 PPL 報告**,且必須限縮 P8 結論的範圍至 1.7B——不得以「8B 的探針品質」或「捕獲條件不同」淡化。

**磁碟**:為騰出空間將刪除 `qwen3-8b-int8-w8a8-pow2`(8.9G,即 v1 的 8B 探針,digest `c6ae749cb3e9144a` 已記錄且重建已於 2026-08-14 證明逐位元組可行)與 `qwen3-1.7b-fp8-dynamic`(2.0G,digest `69d137b187ef43fc` 已記錄,本輪不需要)。二者皆為可重建衍生物,非證據。

### A-12 第二次範圍增補:14B 複製(A 類,預資料,2026-08-15;先於任何 14B 量測)

P8 已在 1.7B 與 8B 上成立。**Qwen3-14B** 為第三個尺寸,母模型 **revision `40c069824f4251a9`**(27.5 GiB,8 shards;於下載之前查 HF API 取得並在此釘住)。

**複製範圍(三項,對 `base-14B` 對 `requant-nearest-14B`)**:PPL(同一份 256 窗口,內容雜湊 `db8ffa33e242a6b2`)、逐層位元一致性(層數依模型結構,需先以 `w3_perlayer.py --stage capture` 產生 14B 捕獲並以 `p1_predictions.py` 產生預測清單)、端到端 token 序列(8 prompt × 64 token)。

**不複製**:`requant-ceil` 與吞吐,理由同 8B 增補。**吞吐另有一項具體理由**:14B 的 INT8 權重約 14 GiB,而釘定網格的 batch 16 × ISL 2048 需約 5.4 GiB 的 KV,合計逼近 `gpu_memory_utilization=0.85` 於 24 GiB 卡上的 20.9 GiB 預算;為不改動已釘定的網格而降低 OOM 風險,吞吐不在 14B 範圍內。

**分階段磁碟流程(於此預先宣告,使刪除不是臨場決定)**。起點 52 GiB 可用:

1. 下載母模型 27.5 GiB → 約 24.5 GiB 可用
2. 建 `base-14B`(約 14 GiB)→ 約 10.5 GiB
3. 以 base 產生捕獲(約 1 GiB)與預測清單,量 base 的 PPL、逐層、端到端
4. **刪除 `base-14B`**(digest 先記入報告)→ 約 23.5 GiB
5. 建 `requant-nearest-14B`(約 14 GiB)→ 約 9.5 GiB
6. 量 pow2 臂的 PPL、逐層、端到端(逐層使用步驟 3 產生的捕獲,故不需 base 仍在)
7. 兩臂皆量畢後刪除母模型 → 約 37 GiB

峰值約 42 GiB。每一步刪除的對象皆先將 digest 寫入報告檔,故所刪者為可重建衍生物而非證據。

**32B 明確排除,連同理由**:母模型 61 GiB 已超出可用空間,且 INT8 權重約 33 GiB 超出單張 24 GiB 卡,只能以兩卡 tensor parallel 執行——而 **TP 引入跨 GPU 的歸約順序,那是另一個決定性變數**(v1 §II 已引 TBIK 論其影響)。以 TP 執行將無法分辨「pow2 失效」與「TP 引入的差異」,故一個混淆的答案比沒有答案更糟。此排除為設備與設計限制,不是結果導向的選擇。

**預測**:三項皆與 1.7B/8B 同向(逐層全數位元相同、端到端 8/8、PPL 增幅與零無法區分)。**不預測 PPL 的具體數值**。若逐層或端到端不成立,該結果優先於 PPL 報告,且 P8 結論限縮至已成立的尺寸,不得以設備或捕獲條件淡化。

### A-12 第二次增補之執行前註記(2026-08-15,先於任何 14B 量測)

**逐層位元一致性在 14B 可能不可得,原因是設備限制而非設計選擇。** `w3_perlayer.py --stage capture` 以 `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16, device_map="cuda")` 載入,而 Qwen3-14B 的 bf16 為 **29.6 GiB,超出單張 24 GiB 卡**。transformers 對 compressed-tensors checkpoint 是否會反量化成 bf16(則 OOM)或維持 INT8(約 15 GiB,則可行),我未確認。

因此 14B 的執行採**嘗試並記錄**:捕獲階段照原樣執行,若 OOM 即在 log 與報告中如實記載「14B 逐層位元一致性因單卡記憶體不足未取得」,並繼續執行 PPL 與端到端。**不改動 `w3_perlayer.py`**(改 `device_map` 會使 14B 的捕獲條件與 1.7B/8B 不同,且屬於執行時修改已釘定程式)。**亦不以兩卡 `device_map` 迴避**:雖然單一 linear 仍在單卡上執行、不引入跨卡歸約,但那仍是執行時的程式修改,且使三個尺寸的捕獲條件不一致。

若逐層未取得,則 14B 的結論建立於 **PPL 與端到端**兩項。端到端直接檢驗部署主張(整個模型的 token 序列是否逐位元組相同),故該兩項足以支撐或推翻「pow2 決定性在 14B 成立」;逐層測試提供的是層級歸因,不是該主張的必要條件。此區分於此預先寫定,以免結果出來後被讀成事後降低標準。

**額外磁碟回收(預先宣告)**:為無人值守執行留餘裕,另刪除 `models--Qwen--Qwen3-1.7B` 母模型快取(3.8 GiB,revision 已記錄)、`perlayer_capture_8b`(650 MiB,8B 逐層量測已完成,可由已記錄 digest 之 checkpoint 重建)、`qwen3-1.7b-int8-w8a8`(2.0 GiB,digest 已記錄且重建已於 2026-08-14 證明)。峰值仍為母模型加兩臂共約 55.5 GiB;`pow2` 臂建構前設 16 GiB 的守衛,不足即中止而非填滿共用磁碟。

### A-12 第二次增補之執行中註記(2026-08-15;端到端於 14B 的失敗與處置)

14B 首次執行結果:**PPL 兩臂成功**;**端到端兩臂失敗**;**逐層捕獲失敗**。

**逐層捕獲失敗即執行前註記所預期的情況**,且機制已確認:`compressed_tensors` 於 forward 中反量化為 bf16(`forward_helpers.py` 的 `dequant_value = dequant_value * scale`),14B 的 bf16 超出 23.52 GiB 可用顯存,OOM。依註記,如實記載「14B 逐層位元一致性未取得」,不改動 `w3_perlayer.py`。

**端到端失敗的性質不同,必須區分**。vLLM 的錯誤為 `No available memory for the cache blocks. Try increasing gpu_memory_utilization`:`run_arm_generic.py` 寫死 `gpu_memory_utilization=0.5`(= 11.8 GiB),而 14B 的 INT8 權重為 16 GiB,權重載入後無空間配置 KV cache。**這是一個為較小模型設定的引擎資源參數,不是關於假說的量測結果。** 對照:`p6_accuracy.py` 使用 0.85(20.4 GiB)而成功。

**處置**:以新腳本 `run_arm_mem.py`(參數化 `gpu_memory_utilization`,不修改 v1 的 `run_arm_generic.py`)於 14B 重跑端到端,取 0.85。**須揭露的條件差異**:14B 的端到端在 `gpu_memory_utilization=0.85` 下執行,而 1.7B 與 8B 在 0.5 下執行。

該差異的影響範圍:兩臂於 14B 使用**相同**設定,故 14B 內部的比較有效;KV cache 大小可改變批次組成,而批次組成在文獻中已知可影響數值(v1 §II 引 he2025batchinvariance),故**跨尺寸比較的是決定性判定(0/8 對 8/8)這一定性結果,不是任何跨尺寸的量值**。

**時序與分類**:本處置決定於觀察到失敗之後,但該失敗為 OOM 與資源配置錯誤,**不含任何關於假說的資訊**——未產生任何 token 序列,故無結果可據以調整。分類為執行中的基礎設施處置,非資料後修正;預測不變(base 0/8、pow2 8/8)。

若 0.85 下仍失敗,則 14B 的端到端亦如實記載為未取得,14B 的結論僅建立於 PPL,而 PPL 不檢驗決定性——屆時須明說 **14B 未對決定性主張提供證據**,不得以 1.7B/8B 的結果代替。
