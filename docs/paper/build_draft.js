const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle,
  AlignmentType, TabStopType, ShadingType, HeightRule, VerticalAlign, Header, Footer, PageNumber,
  MathRun, MathFraction, MathSubScript, MathSuperScript, MathSubSuperScript, MathRoundBrackets, MathSum } = D;
const M = D.Math;

const FONT = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "楷体", cs: "Times New Roman" };
const BODY = 21, SMALL = 18, W = 9026;

// ---------- inline text: **bold** *italic* _{sub} ^{sup}
function runs(text, base = {}) {
  const out = [];
  for (const seg of text.split(/(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})/)) {
    if (!seg) continue;
    const o = { font: FONT, size: base.size || BODY, bold: base.bold, italics: base.italics, color: base.color };
    if (seg.startsWith("**")) out.push(new TextRun({ ...o, text: seg.slice(2, -2), bold: true }));
    else if (seg.startsWith("*")) out.push(new TextRun({ ...o, text: seg.slice(1, -1), italics: true }));
    else if (seg.startsWith("_{")) out.push(new TextRun({ ...o, text: seg.slice(2, -1), subScript: true }));
    else if (seg.startsWith("^{")) out.push(new TextRun({ ...o, text: seg.slice(2, -1), superScript: true }));
    else out.push(new TextRun({ ...o, text: seg }));
  }
  return out;
}
const P = (text, o = {}) => new Paragraph({
  children: runs(text, o), alignment: o.align || AlignmentType.JUSTIFIED,
  indent: o.noIndent ? undefined : { firstLine: 420 }, spacing: { after: o.after ?? 60, line: 300 } });
const H1 = t => new Paragraph({ children: runs(t, { bold: true, size: 26 }), spacing: { before: 240, after: 120 } });
const H2 = t => new Paragraph({ children: runs(t, { bold: true, size: 22 }), spacing: { before: 160, after: 80 } });
const TODO = t => P(`【待补】${t}`, { color: "C00000", noIndent: true });

// ---------- math
const m = s => new MathRun(s);
const A = x => (Array.isArray(x) ? x : [x]).map(e => (typeof e === "string" ? m(e) : e));
const sb = (b, s) => new MathSubScript({ children: A(b), subScript: A(s) });
const sp = (b, s) => new MathSuperScript({ children: A(b), superScript: A(s) });
const ss = (b, s, u) => new MathSubSuperScript({ children: A(b), subScript: A(s), superScript: A(u) });
const fr = (n, d) => new MathFraction({ numerator: A(n), denominator: A(d) });
const br = c => new MathRoundBrackets({ children: A(c) });
const sum = (s, body, u) => new MathSum({ children: A(body), subScript: A(s), superScript: u ? A(u) : [] });
let eqNo = 0;
function EQ(children) {
  eqNo += 1;
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: W / 2 }, { type: TabStopType.RIGHT, position: W }],
    spacing: { before: 80, after: 80 },
    children: [new TextRun({ text: "\t", font: FONT }), new M({ children: A(children) }),
      new TextRun({ text: `\t(${eqNo})`, font: FONT, size: BODY })] });
}

// ---------- three-line tables (caption above, CVPR style)
let tabNo = 0, figNo = 0;
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const rule = sz => ({ style: BorderStyle.SINGLE, size: sz, color: "000000" });
function cell(text, w, o = {}) {
  return new TableCell({
    width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    columnSpan: o.span, margins: { top: 30, bottom: 30, left: 60, right: 60 },
    shading: o.shade ? { type: ShadingType.CLEAR, color: "auto", fill: o.shade } : undefined,
    borders: { top: o.top || NONE, bottom: o.bottom || NONE, left: NONE, right: NONE },
    children: [new Paragraph({ alignment: o.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(text, { size: SMALL, bold: o.bold }) })] });
}
function TABLE(caption, headers, rows, widths, note) {
  tabNo += 1;
  const scale = W / widths.reduce((a, b) => a + b, 0);
  const ws = widths.map(x => Math.round(x * scale)); ws[ws.length - 1] += W - ws.reduce((a, b) => a + b, 0);
  const trs = [new TableRow({ tableHeader: true, children: headers.map((h, i) =>
    cell(h, ws[i], { bold: true, top: rule(12), bottom: rule(6), left: i === 0 })) })];
  rows.forEach((r, ri) => {
    const last = ri === rows.length - 1;
    if (r.group) {   // full-width group label row
      trs.push(new TableRow({ children: [cell(r.group, W, { span: headers.length, left: true, shade: "F2F2F2",
        top: ri ? rule(4) : undefined, bottom: last ? rule(12) : undefined })] }));
      return;
    }
    const cells = r.cells || r;
    trs.push(new TableRow({ children: cells.map((c, i) => cell(String(c), ws[i], { left: i === 0,
      bold: r.bold, shade: r.shade, top: r.sep ? rule(4) : undefined, bottom: last ? rule(12) : undefined })) }));
  });
  const out = [new Paragraph({ keepNext: true, spacing: { before: 160, after: 60 }, alignment: AlignmentType.JUSTIFIED,
      children: [...runs(`表 ${tabNo}. `, { bold: true, size: SMALL }), ...runs(caption, { size: SMALL })] }),
    new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: ws, rows: trs,
      borders: { top: NONE, bottom: NONE, left: NONE, right: NONE, insideHorizontal: NONE, insideVertical: NONE } })];
  if (note) out.push(new Paragraph({ spacing: { before: 40, after: 160 }, children: runs(note, { size: 16 }) }));
  else out.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  return out;
}

// ---------- figure placeholders (caption below)
function FIG(caption, lines, h = 2600) {
  figNo += 1;
  const dash = { style: BorderStyle.DASHED, size: 6, color: "7F7F7F" };
  return [new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [new TableRow({
      height: { value: h, rule: HeightRule.ATLEAST }, children: [new TableCell({
        width: { size: W, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
        shading: { type: ShadingType.CLEAR, color: "auto", fill: "F5F5F5" },
        borders: { top: dash, bottom: dash, left: dash, right: dash },
        margins: { top: 100, bottom: 100, left: 200, right: 200 },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: runs(`【图 ${figNo} 占位】`, { bold: true, color: "7F7F7F" }) }),
          ...lines.map(l => new Paragraph({ alignment: AlignmentType.LEFT, children: runs(l, { size: SMALL, color: "595959" }) }))] })] })] }),
    new Paragraph({ spacing: { before: 60, after: 200 }, alignment: AlignmentType.JUSTIFIED,
      children: [...runs(`图 ${figNo}. `, { bold: true, size: SMALL }), ...runs(caption, { size: SMALL })] })];
}

// =====================================================================================
const C = [];
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: runs("LiteGTR：面向无人机小目标检测的轻量级全局 Token 精修网络", { bold: true, size: 32 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
  children: runs("LiteGTR: Lightweight Global-Token Refinement for Small Object Detection in UAV Imagery", { size: 24 }) }));
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: runs("作者（待定）", { size: BODY }) }));

// ---------------- Abstract
C.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: runs("摘  要", { bold: true, size: 24 }) }));
C.push(P("无人机图像中的目标以小目标为主且分布密集，而机载平台又要求检测器参数少、延迟低、可部署。轻量检测器依赖局部卷积，难以获得全局上下文；引入全局建模的方法要么计算量随空间分辨率二次增长，要么依赖与数据相关的稀疏计算，导出到推理引擎时形状不固定。本文提出 LiteGTR，一个 2.32M 参数的单模态 RGB 检测器。LiteGTR 在 P3–P5 特征层上以固定预算、分区域的方式选出 56 个 token，经 token 混合建模全局关系后，通过几何感知写回将全局信息以残差形式写回特征图：每个 token 预测自身的各向异性高斯范围，使描述小目标的 token 只作用于其邻域。我们发现，仅靠检测损失和一致性约束训练的打分器会塌缩为平坦分数图，路由退化为随机选择；为此引入基于真值中心热力图的路由监督，并以光照扰动视角下的 EMA 教师约束路由对光照变化保持稳定。整个网络为静态形状，可直接导出 ONNX/TensorRT。在 VisDrone 上，LiteGTR 以 [xx] GFLOPs 取得 [xx] AP_{S}，[实验结论待补]。"));
C.push(P("**关键词：**无人机目标检测；小目标检测；轻量级网络；Token 路由；边缘部署", { noIndent: true, after: 200 }));

// ---------------- 1 Introduction
C.push(H1("1  引言"));
C.push(P("无人机航拍在交通监测、应急搜救和巡检中应用广泛。与自然图像不同，航拍图像视场大、目标小且密集：VisDrone 验证集约 68.5% 的实例面积小于 32^{2} 像素，每张图平均约 71 个目标 [待 3.3 节实测替换]。图像缩放到 640 输入后，大量目标只覆盖一个 stride-8 特征单元的一部分。与此同时，机载计算平台对参数量、延迟和算子兼容性都有严格限制。"));
C.push(P("现有工作沿两条路线展开。其一是轻量 YOLO 类检测器 [LEAF-YOLO, RemDet, FBRT-YOLO]，通过改进局部特征提取与融合降低计算量，但感受野受限，在密集小目标场景中缺少区分相似目标所需的上下文。其二是引入全局建模：Transformer 类方法的注意力代价随位置数二次增长；QueryDet、CEASC 等稀疏方法只在部分位置计算，但稀疏位置的数量由输入决定，导出到 TensorRT 等引擎时出现动态形状和算子回退，推理端的加速难以兑现。"));
C.push(P("我们认为全局上下文并不需要在所有位置上计算。只要少量 token 落在目标所在位置，且写回时尊重每个 token 所描述区域的空间范围，就能以很低的代价为小目标补充上下文。基于这一观察，LiteGTR 在每个特征层上把特征图划分为固定网格，每个区域选出固定数量的 token，总预算为 56，所有形状在编译期确定。写回时，每个 token 预测一个各向异性高斯范围，与内容注意力共同决定写回权重。训练中我们观察到打分器会塌缩为平坦的分数图，因此进一步引入路由监督。"));
C.push(P("本文的贡献如下：", { noIndent: true }));
C.push(P("（1）提出固定预算的局部 top-k 路由与几何感知写回，以 O(N·HW) 的代价（N = 56）为特征图引入全局上下文，整网静态形状，可直接部署；", { noIndent: true }));
C.push(P("（2）揭示了仅靠门控梯度和一致性约束训练时打分器的塌缩现象（分数图归一化熵趋近 1），提出基于真值中心热力图的路由监督，并以光照扰动视角下的 EMA 教师约束路由的光照不变性；", { noIndent: true }));
C.push(P("（3）在 VisDrone 和 DroneVehicle 上，在统一训练协议下与参数量匹配的基线和同量级检测器对比，并按极小、微小、小目标分档报告精度，[实验结论待补]。", { noIndent: true }));
C.push(...FIG("同量级轻量检测器在 VisDrone val 上的 AP_{S}–参数量对比。LiteGTR 在更少参数下取得更高的小目标精度。", [
  "内容：横轴参数量（M，0–7），纵轴 AP_{S}（%）；气泡大小表示 GFLOPs。",
  "点：LEAF-YOLO-N、RemDet-Tiny、EdgeYOLO-Tiny、PDWT-YOLO、LEAF-YOLO（引用值，空心）；YOLOv8n、YOLO11n、CSP-N、CSP-T（同协议复现，实心）；LiteGTR-S、LiteGTR（红色五角星）。",
  "数据来源：表 1。"], 2400));

C.push(H1("2  相关工作"));
C.push(TODO("轻量无人机检测；小目标 / 极小目标检测（NWD、RFLA、DQ-DETR、SimD）；稀疏与 token 化的全局建模（QueryDet、CEASC、Sparse DETR）。"));

// ---------------- 3 Preliminaries
C.push(H1("3  准备工作"));
C.push(H2("3.1  问题定义与记号"));
C.push(P("输入图像经 letterbox 缩放到 *I* ∈ ℝ^{3×640×640}。骨干输出 {*C*_{l}}，经特征金字塔得到 {*P*_{l}}，*l* ∈ {2, 3, 4, 5}，步长 *s*_{l} = 2^{l}，*P*_{l} 的尺寸为 *C*×*H*_{l}×*W*_{l}，统一通道 *C* = 64。在 640 输入下，P2–P5 的空间尺寸分别为 160、80、40、20。token 路径只作用于 P3–P5；P2 只经过局部卷积，以控制计算量。"));
C.push(P("为区分小目标内部的尺度差异，除 COCO 的 AP_{S}（面积 < 32^{2}）外，本文沿用 AI-TOD 的分档，按目标边长 √*a* 统计："));
C.push(EQ([sb("AP", "vt"), m(": √a∈[2,8),  "), sb("AP", "t"), m(": √a∈[8,16),  "), sb("AP", "s"), m(": √a∈[16,32),  "), sb("AP", "m"), m(": √a∈[32,64)")]));
C.push(H2("3.2  检测头与训练目标"));
C.push(P("检测头采用 GFL [Li et al.]，在 P2–P5 的每个位置预测类别质量分数和边框分布。分类使用质量焦点损失（QFL），目标 *y* 为预测框与所分配真值的 IoU："));
C.push(EQ([sb("L", "QFL"), m("(σ)=−"), sp(br("|y−σ|"), "β"), br([m("(1−y)log(1−σ)+y log σ")])]));
C.push(P("边框的四条边距离各表示为 *n* + 1 个离散值上的分布（*n* = 16），用分布焦点损失（DFL）约束："));
C.push(EQ([sb("L", "DFL"), m("=−"), br([br([sb("y", "i+1"), m("−y")]), m("log "), sb("S", "i"), m("+"), br([m("y−"), sb("y", "i")]), m("log "), sb("S", "i+1")])]));
C.push(P("并与 GIoU 损失共同优化。正样本由 TaskAligned 分配器选取，对齐度量为 *t* = *s*^{α}·*u*^{β}，其中 *s* 为分类分数、*u* 为 IoU，取 α = 1、β = 6、每个真值取 top-13 候选。候选点必须位于真值框内部；在 640 输入下，边长小于 P2 步长（4 像素）的目标可能落在网格中心之间而没有任何候选点，按公开尺度统计估算约占全部真值的 9.5%，这些目标从未成为正样本，反而被当作背景训练。为此，对没有候选点的真值，我们把离其中心最近的 P2 网格点加入候选，并将其分类目标下限设为 0.1，使预测框尚未与之重叠时仍能获得回归梯度；其余真值的分配不受影响。"));
C.push(H2("3.3  无人机图像的尺度统计"));
C.push(P("本文预处理（屏蔽 ignored 区域、去除 others 类）后，VisDrone val 平均每图 [xx] 个目标，面积小于 32^{2} 的实例占 [xx]%，边长小于 16 像素的占 [xx]%，大目标仅占 [xx]%（由 tools/analyze_dataset.py 统计；公开统计的参考值分别约为 71、68.5%、30.8%、2.8%）。网络实际看到的是 640 输入：在该尺度下，[xx]% 的目标边长小于一个 P3 单元（8 像素）。这是本文在 P3 上分配最多 token、并用 P2 保留局部细节的依据。"));
C.push(H1("4  方法"));
C.push(H2("4.1  总体结构"));
C.push(P("LiteGTR 的结构如图 2 所示。TinyNeXt 骨干提取 *C*_{2}–*C*_{5}，经金字塔投影与 FPN 得到 P2–P5，每层再经过局部卷积路径。P3–P5 进入 token 路径：固定预算局部路由选出 56 个 token，token 混合器在 token 之间建模全局关系，几何感知写回把结果以残差形式写回 P3–P5。P2 与写回后的 P3′–P5′ 送入 GFL 检测头。路由监督和 EMA 教师只在训练时使用，不参与推理。"));
C.push(...FIG("LiteGTR 总体结构。(a) 整体流程；(b) TinyNeXt Block；(c) 固定预算局部路由；(d) 几何感知写回；(e) 仅训练使用的路由监督与光照一致 EMA 教师（虚线）。", [
  "(a) 输入航拍图 → TinyNeXt Stage1–4（32×160² / 64×80² / 128×40² / 192×20²）→ 1×1+LN → FPN（自顶向下相加）→ DWSep 平滑 → 局部卷积路径（P2 ×1，P3–P5 ×2）。",
  "P3/P4/P5 向下取 token（橙色，32/16/8），经 Token Mixer、几何写回，红色箭头经 ⊕ 写回得到 P3′/P4′/P5′；P2 直连检测头。",
  "(c) 4×4 / 4×4 / 2×2 网格，每区域 top-k 白点。(d) 内容项 ⊕ 高斯先验 → softmax → Σ → γ·Proj → ⊕；旁边画各向异性高斯椭圆：小目标上的 token 椭圆窄，背景 token 椭圆宽。",
  "(e) 真值中心热力图 → focal loss 监督分数图；光照扰动图像 → EMA 教师 → KL 一致性。右下角图例。"], 3600));

C.push(H2("4.2  骨干与特征金字塔"));
C.push(P("骨干采用 ConvNeXt 风格的 TinyNeXt：4×4 步长为 4 的 stem 后接四个阶段，通道为 [32, 64, 128, 192]，深度为 [2, 4, 8, 2]，阶段间以 LN 加 2×2 步长卷积下采样。每个 block 为"));
C.push(EQ([m("y=x+λ⊙"), sb("W", "2"), m(" GELU"), br([sb("W", "1"), m(" LN"), br([sb("DW", "7×7"), m("(x)")])])]));
C.push(P("其中 *W*_{1} 将通道扩展 4 倍，λ 为逐通道缩放系数。相对原始配置，我们缩减了对小目标贡献最小的第 4 阶段，骨干参数由 3.76M 降至 2.03M。金字塔投影将各层统一到 64 通道并自顶向下融合："));
C.push(EQ([sb("P", "l"), m("=Φ"), br([m("LN"), br([sb("W", "l"), sb("C", "l")]), m("+Up"), br(sb("P", "l+1"))])]));
C.push(P("Φ 为深度可分离卷积（DW 3×3 + PW 1×1）。之后每层经过局部卷积路径：P2 使用 1 个深度可分离卷积块，P3–P5 各使用 2 个，负责边缘、纹理与相邻目标的区分。"));

C.push(H2("4.3  固定预算局部路由"));
C.push(P("对每个 token 层，打分器（3×3 卷积–BN–SiLU–1×1 卷积）输出与 *P*_{l} 同尺寸（*H*_{l}×*W*_{l}）的分数图 *s*_{l}。全局 top-k 容易集中在少数显著区域，而航拍场景中目标分散，因此我们把 *s*_{l} 划分为 *G*_{l}×*G*_{l} 个区域 {*R*_{r}}，在每个区域内取前 *k*_{l} 个位置："));
C.push(EQ([sb("Ω", "l"), m("="), sum("r", [m("TopK"), br([sb("k", "l"), m(",{"), sb("s", "l"), m("(p)|p∈"), sb("R", "r"), m("}")])]), m(",   "), m("|"), sb("Ω", "l"), m("|="), sp(sb("G", "l"), "2"), sb("k", "l")]));
C.push(P("P3、P4、P5 的网格与每区域数量分别为 4×4×2、4×4×1、2×2×2，对应 32、16、8 个 token，共 *N* = 56。*k*_{l} 在编译期确定，路由不含数据相关的控制流。top-k 只提供索引，本身不可导；为使检测损失能够影响打分器，被选中的特征乘以分数门控："));
C.push(EQ([sb("t", "i"), m("=σ"), br([sb("s", "l"), m("("), sb("p", "i"), m(")")]), m("·"), sb("P", "l"), m("("), sb("p", "i"), m("),   "), sb("c", "i"), m("="), br([fr([sb("x", "i"), m("+0.5")], sb("W", "l")), m(","), fr([sb("y", "i"), m("+0.5")], sb("H", "l"))])]));
C.push(P("其中 *c*_{i} 为 token 源位置的归一化中心，供后续模块使用。"));

C.push(H2("4.4  Token 混合"));
C.push(P("56 个 token 拼接后加入坐标嵌入与层级嵌入，经一层单头自注意力与 FFN（扩展比 2）建模全局关系："));
C.push(EQ([sb("z", "i"), m("="), sb("t", "i"), m("+"), sb("W", "c"), sb("c", "i"), m("+"), sb("e", "l(i)"), m(",   "), m("Z′=Z+MHSA(LN Z),   "), m("Z″=Z′+FFN(LN Z′)")]));
C.push(P("由于 *N* 很小，这一步的代价与特征图分辨率无关。"));

C.push(H2("4.5  几何感知写回"));
C.push(P("一个 token 概括的是特定尺度上的特定区域。若把它均匀广播或只按内容相似度写回，描述一辆 12 像素汽车的 token 会扩散到整个特征图。我们让每个 token 预测自身在两个轴上的范围："));
C.push(EQ([sb("σ", "i"), m("=softplus"), br([sb("W", "s"), sb("z", "i")]), m("+"), sb("σ", "min"), m(",   "), sb("σ", "i"), m("∈"), sp("ℝ", "2")]));
C.push(P("对目标层上归一化坐标为 *q*_{p} 的位置 *p*，写回注意力由内容项与高斯几何先验相加得到："));
C.push(EQ([m("C(p,i)="), fr([m("⟨"), sb("W", "q"), sb("f", "p"), m(","), sb("W", "k"), sb("z", "i"), m("⟩")], m("√d")), m(",   "),
  m("G(p,i)=−"), fr("1", "2"), sum("a∈{x,y}", fr(sp(br([sb("q", "p,a"), m("−"), sb("c", "i,a")]), "2"), sp(sb("σ", "i,a"), "2")))]));
C.push(EQ([m("A(p,i)="), sb("softmax", "i"), br(m("C(p,i)+G(p,i)")), m(",   "),
  sb("f′", "p"), m("="), sb("f", "p"), m("+γ "), sb("W", "o"), sum("i=1", [m("A(p,i) "), sb("W", "v"), sb("z", "i")], "N")]));
C.push(P("γ 为可学习标量并初始化为 0，训练从纯局部卷积的解出发，逐步引入全局路径；残差形式保证小目标的局部细节不会被替换。σ_{min} = 0.02（归一化坐标）。写回作用于 P3–P5，各层权重独立。去掉 *G* 即退化为内容注意力，这是 5.4 节检验几何先验的消融。"));
C.push(H2("4.6  路由监督"));
C.push(P("仅靠分数门控，打分器得不到“该选哪里”的明确信号：门控只是逐 token 的缩放，后续层可以吸收；一致性约束（4.7 节）在两个分数图相同时即为零，平坦分数图也满足这一条件；打分器输出层的权重衰减又持续把分数拉向常数。实测中，未加监督的模型训练到第 135 轮时 P3 分数图的归一化熵为 0.999997（分数标准差约 0.01），且随训练单调上升，路由退化为随机选择。为此，我们用真值中心热力图监督每层分数图："));
C.push(EQ([sb("H", "l"), m("(p)="), sb("max", "j"), m(" exp"), br([m("−"), fr(sp(br([sb("x", "p"), m("−"), sb("x", "j")]), "2"), [m("2"), sp(sb("σ", "j,x"), "2")]), m("−"), fr(sp(br([sb("y", "p"), m("−"), sb("y", "j")]), "2"), [m("2"), sp(sb("σ", "j,y"), "2")])]), m(",   "),
  sb("σ", "j,x"), m("=max"), br([m("ρ"), fr(sb("w", "j"), sb("s", "l")), m(","), sb("σ", "min")])]));
C.push(P("(*x*_{j}, *y*_{j}) 为第 *j* 个真值框中心在该层网格上的坐标，ρ = 1/6，σ_{min} = 0.5 个单元，σ_{j,y} 同理；中心所在单元置 1 作为正样本。每层都监督全部目标，极小目标在粗层上也保有最小半径，因此路由由目标密度驱动。损失采用 CenterNet 的惩罚衰减焦点损失，*ŝ* = σ(*s*_{l})："));
C.push(EQ([sb("L", "route"), m("=−"), fr("1", sb("N", "pos")), sum("p", [m("{"), sp(br("1−ŝ"), "α"), m("log ŝ,  H(p)=1;   "), sp(br("1−H(p)"), "β"), sp("ŝ", "α"), m("log(1−ŝ),  否则}")])]));
C.push(P("取 α = 2、β = 4。同时，打分器输出层的 1×1 卷积不施加权重衰减。该监督只在训练时计算，推理图与参数量不变。"));

C.push(H2("4.7  光照一致的 EMA 路由"));
C.push(P("DroneVehicle 等数据含大量夜间与低照度图像，我们希望 token 的位置不随光照变化。EMA 教师为打分器的滑动平均副本，动量在前 1000 步内预热："));
C.push(EQ([sb("θ", "T"), m("←"), sb("m", "n"), sb("θ", "T"), m("+"), br([m("1−"), sb("m", "n")]), sb("θ", "S"), m(",   "), sb("m", "n"), m("=min"), br([m("m,"), fr("1+n", "10+n")]), m(",   m=0.999")]));
C.push(P("学生使用原图，教师使用同一批图像的光照扰动视角：逐样本亮度 *b* ~ U(0.6, 1.4)、对比度 *c* ~ U(0.6, 1.4)、gamma *g* ~ U(0.7, 1.5)、高斯噪声标准差 ~ U(0, 0.03)，几何不变，因此两张分数图逐像素对齐："));
C.push(EQ([m("Ĩ=clip"), br([sp(br([m("c(bI−μ)+μ")]), "g"), m("+ε")]), m(",   "),
  sb("L", "cons"), m("="), fr("1", "|L|"), sum("l", [m("KL"), br([m("softmax("), ss("s", "l", "T"), m(")‖softmax("), ss("s", "l", "S"), m(")")])])]));
C.push(P("教师分支只做一次额外前向，不反传；BN 在该分支中使用批统计但不更新滑动统计量，扰动图像不会影响验证时的统计。有了 4.6 节的监督，分数图不再平坦，“光照变化下保持一致”不再能被平凡满足。"));

C.push(H2("4.8  训练目标"));
C.push(EQ([m("L="), sb("L", "QFL"), m("+2"), sb("L", "GIoU"), m("+0.25"), sb("L", "DFL"), m("+"), sb("λ", "c"), sb("L", "cons"), m("+"), sb("λ", "r"), sb("L", "route"), m(",   "), sb("λ", "c"), m("="), sb("λ", "r"), m("=0.5")]));

C.push(H2("4.9  复杂度与部署"));
C.push(P("P3 在 640 输入下有 6400 个位置。对其做全局自注意力的代价为 O((HW)^{2}C)；LiteGTR 中 token 混合为 O(*N*^{2}*C*)，写回为 O(*N*·HW·*C*)，*N* = 56。路由的 *k*、网格划分与所有张量形状均为常量，ONNX 以静态形状导出。LiteGTR 共 2.32M 参数、4.93G MACs（≈9.9 GFLOPs，另有注意力矩阵乘约 0.06G）：骨干 2.03M，检测头 74K，打分器 56K，写回 50K，金字塔投影 47K，局部路径 34K，混合器 34K；仅训练使用的 EMA 教师（56K）不计入。LiteGTR-S 将骨干缩为 [24, 48, 96, 160] × [2, 3, 6, 2]、token 预算改为 16 / 16 / 8，共 1.22M 参数、2.49G MACs。"));
C.push(H1("5  实验"));
C.push(H2("5.1  数据集与评测"));
C.push(P("**VisDrone2019-DET**：6471 张训练、548 张验证图像，10 类。ignored 区域以填充值屏蔽，others 类剔除。所有消融在 val 上报告。**DroneVehicle**：使用 RGB 模态、水平框，按亮度把图像标注为 day / night / dark，用于跨光照分析。评测使用 COCO API（maxDets = 100），另按 3.1 节分档报告 AP_{vt}、AP_{t}。同量级模型的外部结果评测协议不一，单独标注。"));
C.push(H2("5.2  实现细节"));
C.push(P("所有模型（含基线）从零训练 200 轮，输入 640，batch 16，AdamW（lr = 1×10^{−3}，weight decay = 0.05），梯度裁剪 10。学习率先线性预热 3 轮，保持峰值至第 100 轮，再余弦衰减至 1%。Mosaic 概率 0.5，最后 10 轮关闭。模型权重 EMA 衰减 0.9999，验证与保存均使用 EMA 权重。混合精度训练，单卡 [GPU 型号待补]。基线只替换骨干，复用相同的 neck、检测头、损失、分配器、数据增强与训练日程，并按参数量与 LiteGTR 配对。"));

C.push(H2("5.3  与现有方法的比较"));
C.push(...TABLE("VisDrone2019-DET val 上的比较。上半部分为引用的已发表结果（协议与训练轮数各不相同，仅供参考）；下半部分在统一协议下训练与评测。AP_{vt} / AP_{t} 按边长 [2,8) / [8,16) 像素统计。",
  ["方法", "来源", "轮数", "Params(M)", "GFLOPs", "mAP_{50:95}", "mAP_{50}", "AP_{S}", "AP_{vt}", "AP_{t}", "延迟(ms)"],
  [{ group: "引用结果†" },
   ["LEAF-YOLO-N", "ISWA'25", "—", "1.2", "5.6", "21.9", "39.7", "14.0", "‡", "‡", "—"],
   ["EDNet-Tiny", "UIC'24", "—", "1.8", "—", "19.5", "34.1", "—", "—", "—", "—"],
   ["YOLO26n-P2", "DHF-YOLO", "300", "2.40", "6.5", "19.2", "33.1", "—", "—", "—", "—"],
   ["DHF-YOLO", "2026", "300", "2.63", "9.9", "20.4", "34.9", "—", "—", "—", "—"],
   ["RemDet-Tiny", "AAAI'25", "300", "3.2", "4.6^{a}", "21.8", "37.1", "12.7", "‡", "‡", "—"],
   ["LEAF-YOLO", "ISWA'25", "—", "4.28", "20.9", "28.2", "48.3", "20.0", "—", "—", "—"],
   ["EdgeYOLO-Tiny", "—", "—", "5.5", "27.2", "21.8", "38.5", "12.4^{b}", "—", "—", "—"],
   ["PDWT-YOLO", "—", "—", "6.44", "24.5", "24.3", "42.6", "15.9^{b}", "—", "—", "—"],
   { group: "统一协议（本文训练，200 轮，从零）" },
   ["YOLOv8n", "复现", "200", "—", "—", "", "", "", "", "", ""],
   ["YOLO11n", "复现", "200", "—", "—", "", "", "", "", "", ""],
   ["CSP-T", "基线", "200", "1.17", "≈4.4", "", "", "", "", "", ""],
   ["CSP-N", "基线", "200", "2.25", "≈7.1", "", "", "", "", "", ""],
   { cells: ["LiteGTR-S（本文）", "—", "200", "1.22", "≈5.0", "", "", "", "", "", ""], bold: true },
   { cells: ["LiteGTR（本文）", "—", "200", "2.32", "≈9.9", "", "", "", "", "", ""], bold: true }],
  [17, 10, 6, 9, 8, 9, 8, 7, 7, 7, 9],
  "† 数据取自作者公开的代码仓库 README 或训练日志。‡ 用公开权重以本文评测脚本重新计算。a：mmdet 统计口径，可能为 MACs；b：由 LEAF-YOLO 作者测得。延迟为 [设备] 上 TensorRT FP16、batch 1、640×640，由 tools/benchmark_latency.py 测得。"));
C.push(TODO("结果分析：与参数量匹配基线的差距；与引用结果的比较需说明协议差异。"));

C.push(H2("5.4  消融实验"));
C.push(...TABLE("消融实验（VisDrone val）。每行相对完整模型只改变一项。",
  ["配置", "GFLOPs", "mAP_{50:95}", "mAP_{50}", "AP_{S}", "AP_{vt}", "AP_{M}", "AP_{L}"],
  [{ cells: ["LiteGTR（完整）", "≈9.9", "", "", "", "", "", ""], bold: true },
   { cells: ["− Token 路径（仅局部卷积）", "", "", "", "", "", "", ""], sep: true },
   ["写回：均匀广播", "", "", "", "", "", "", ""],
   ["写回：仅内容注意力（− 几何先验）", "", "", "", "", "", "", ""],
   ["− 路由监督", "≈9.9", "", "", "", "", "", ""],
   ["− 光照一致 EMA", "≈9.9", "", "", "", "", "", ""],
   { cells: ["Token 预算 56 → 256", "", "", "", "", "", "", ""], sep: true }],
  [36, 9, 10, 9, 9, 9, 9, 9],
  "注：AP_{M} 与 AP_{L} 用于量化面向小目标设计对中、大目标的影响；VisDrone val 中大目标仅约占 2.8%。"));
C.push(...TABLE("路由行为分析（P3）。命中率为被选中 token 落在真值框内的比例；随机水平为目标面积占比。",
  ["路由方式", "分数图熵 ↓", "Token 命中率 ↑", "随机水平", "EMA 一致性", "AP_{S}"],
  [["随机路由（无门控、无监督）", "—", "", "", "—", ""],
   ["门控 + EMA（无路由监督）", "0.99999^{*}", "", "", "0.999^{*}", ""],
   { cells: ["门控 + EMA + 路由监督（本文）", "", "", "", "", ""], bold: true }],
  [34, 15, 14, 11, 12, 10],
  "注：带星号的数值为无路由监督那次训练第 135 轮的实测值。熵接近 1 表示分数图平坦；EMA 一致性接近 1 是因为教师与学生的分数图都平坦，并非学到了光照不变性。"));
C.push(TODO("消融分析。重点：− token 路径时 AP_{S} 与 AP_{L} 的变化方向；无监督路由的命中率是否接近随机。"));
C.push(...FIG("Token 分数图与被选中位置（VisDrone val）。上：无路由监督，分数图平坦，token 近似随机分布；下：本文，token 集中在目标上。颜色为 σ(s)（0–1 绝对刻度），白点为被选中的 token。", [
  "由 tools/visualize_tokens.py 生成：输入+GT | P3 | P4 | P5 分数图。选 2 张目标密集、小目标多的图像。",
  "可在右下角嵌入小图：两次训练的 P3 分数图熵随 epoch 的变化（数据来自 token_stats.csv）。"], 2600));

C.push(H2("5.5  跨光照分析"));
C.push(...TABLE("DroneVehicle（RGB）上按光照条件拆分的 mAP_{50:95}。",
  ["配置", "day", "night", "dark", "Δ(night − day)"],
  [{ cells: ["LiteGTR", "", "", "", ""], bold: true }, ["− Token 路径", "", "", "", ""], ["− 光照一致 EMA", "", "", "", ""]],
  [34, 16, 16, 16, 18], "注：图像数 day / night / dark = [待补]。"));
C.push(H1("6  结论"));
C.push(TODO("实验完成后撰写。"));
C.push(H1("参考文献"));
C.push(TODO("RemDet (AAAI 2025)；LEAF-YOLO (ISWA 2025)；FBRT-YOLO (AAAI 2025)；EDNet (UIC 2024)；GFL；ConvNeXt；QueryDet (CVPR 2022)；CEASC (CVPR 2023)；CenterNet；NWD / AI-TOD；DQ-DETR (ECCV 2024)；SimD (IROS 2024)；VisDrone；DroneVehicle；DEIM (CVPR 2025)。"));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: BODY } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SMALL })] })] }) },
    children: C }] });
Packer.toBuffer(doc).then(b => { fs.writeFileSync(process.argv[2] || "LiteGTR_draft.docx", b); console.log("ok"); });
