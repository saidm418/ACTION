importScripts('https://cdn.jsdelivr.net/npm/jstat@latest/dist/jstat.min.js');

// تحليل نص معدلات وتحويله لمصفوفة أرقام صحيحة (ليست ضرورية هنا لأن البيانات تصل جاهزة)
// function parseData(text) {...}  // غير مستخدمة هنا

function testNormality(data) {
  if (data.length < 3) return { normal: false, pValue: NaN };

  let mean = jStat.mean(data);
  let sd = jStat.stdev(data, true);
  let n = data.length;

  let sorted = data.slice().sort((a,b) => a-b);

  let d = 0;
  for(let i=0; i<n; i++) {
    let Fi = (i+1)/n;
    let cdf = jStat.normal.cdf(sorted[i], mean, sd);
    let diff1 = Math.abs(Fi - cdf);
    let diff2 = Math.abs(cdf - i/n);
    d = Math.max(d, diff1, diff2);
  }

  let en = Math.sqrt(n);
  let lambda = (en + 0.12 + 0.11/en) * d;
  let pValue = kolmogorovSmirnovPvalue(lambda);

  return {normal: pValue > 0.05, pValue};
}

function kolmogorovSmirnovPvalue(lambda) {
  if (lambda < 1.18) {
    let sum = 0;
    for (let k = -20; k <= 20; k++) {
      let term = Math.pow(-1, k) * Math.exp(-2 * k * k * lambda * lambda);
      sum += term;
    }
    return Math.min(Math.max(2*sum, 0),1);
  } else {
    return 2 * Math.exp(-2 * lambda * lambda);
  }
}

function leveneTest(group1, group2) {
  let m1 = jStat.mean(group1);
  let m2 = jStat.mean(group2);

  let absDev1 = group1.map(x => Math.abs(x - m1));
  let absDev2 = group2.map(x => Math.abs(x - m2));

  // تقريبا اختبار t على الفروق المطلقة (تقديري)
  let n1 = absDev1.length;
  let n2 = absDev2.length;
  let mean1 = jStat.mean(absDev1);
  let mean2 = jStat.mean(absDev2);
  let var1 = jStat.variance(absDev1);
  let var2 = jStat.variance(absDev2);

  let s_pooled = Math.sqrt((var1/n1) + (var2/n2));
  if(s_pooled === 0) return {homogeneous: true, pValue: 1}; // لا اختلاف

  let t = (mean1 - mean2) / s_pooled;
  let df = n1 + n2 - 2;
  let p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), df));

  return {homogeneous: p > 0.05, pValue: p};
}

function cohensD(group1, group2) {
  let m1 = jStat.mean(group1);
  let m2 = jStat.mean(group2);
  let s1 = jStat.stdev(group1, true);
  let s2 = jStat.stdev(group2, true);
  let n1 = group1.length;
  let n2 = group2.length;
  let s_pooled = Math.sqrt( ((n1 -1)*s1*s1 + (n2 -1)*s2*s2) / (n1 + n2 - 2) );
  return (m1 - m2) / s_pooled;
}

function effectSizeR(U, n1, n2) {
  let z = (U - (n1*n2)/2) / Math.sqrt(n1*n2*(n1+n2+1)/12);
  return Math.abs(z) / Math.sqrt(n1 + n2);
}

function interpretCohensD(d) {
  let absd = Math.abs(d);
  if(absd < 0.2) return "صغير جدًا";
  if(absd < 0.5) return "صغير";
  if(absd < 0.8) return "متوسط";
  return "كبير";
}

function interpretR(r) {
  if(r < 0.1) return "صغير جدًا";
  if(r < 0.3) return "صغير";
  if(r < 0.5) return "متوسط";
  return "كبير";
}

function pairedTTest(group1, group2) {
  let diffs = group1.map((v,i) => v - group2[i]);
  let meanDiff = jStat.mean(diffs);
  let sdDiff = jStat.stdev(diffs, true);
  let n = diffs.length;
  let t = meanDiff / (sdDiff / Math.sqrt(n));
  let df = n - 1;
  let p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), df));
  return {t, p, df};
}

function mannWhitneyU(group1, group2) {
  let combined = group1.concat(group2);
  let ranks = rankArray(combined);
  let ranks1 = ranks.slice(0, group1.length);
  let ranks2 = ranks.slice(group1.length);

  let R1 = ranks1.reduce((a,b)=>a+b, 0);
  let R2 = ranks2.reduce((a,b)=>a+b, 0);

  let n1 = group1.length;
  let n2 = group2.length;

  let U1 = n1*n2 + (n1*(n1+1))/2 - R1;
  let U2 = n1*n2 + (n2*(n2+1))/2 - R2;

  let U = Math.min(U1, U2);

  let mu = n1*n2/2;
  let sigma = Math.sqrt(n1*n2*(n1+n2+1)/12);
  let z = (U - mu) / sigma;

  let p = 2 * (1 - jStat.normal.cdf(Math.abs(z), 0, 1));
  return {U, z, p, n1, n2};
}

function wilcoxonSignedRank(group1, group2) {
  let diffs = group1.map((v,i) => v - group2[i]);
  let absDiffs = diffs.map(Math.abs);
  let ranks = rankArray(absDiffs);
  let Wplus = 0;
  let Wminus = 0;
  for(let i=0; i<diffs.length; i++) {
    if(diffs[i] > 0) Wplus += ranks[i];
    else if(diffs[i] < 0) Wminus += ranks[i];
  }
  let W = Math.min(Wplus, Wminus);

  let n = diffs.filter(d=>d!=0).length;
  let meanW = n*(n+1)/4;
  let sdW = Math.sqrt(n*(n+1)*(2*n+1)/24);
  let z = (W - meanW) / sdW;
  let p = 2 * (1 - jStat.normal.cdf(Math.abs(z), 0, 1));
  return {W, z, p, n};
}

function rankArray(arr) {
  let sorted = arr.slice().map((v,i) => ({v,i})).sort((a,b)=>a.v - b.v);
  let ranks = [];
  for(let i=0; i<arr.length; i++) ranks.push(0);

  let rank = 1;
  for(let i=0; i<sorted.length; i++) {
    if(i > 0 && sorted[i].v === sorted[i-1].v) {
      ranks[sorted[i].i] = ranks[sorted[i-1].i];
    } else {
      ranks[sorted[i].i] = rank;
    }
    rank++;
  }
  return ranks;
}

onmessage = function(e){
  const {group1, group2, studyType} = e.data;

  let mean1 = jStat.mean(group1).toFixed(3);
  let mean2 = jStat.mean(group2).toFixed(3);
  let sd1 = jStat.stdev(group1, true).toFixed(3);
  let sd2 = jStat.stdev(group2, true).toFixed(3);

  let norm1 = testNormality(group1);
  let norm2 = testNormality(group2);

  let homogeneity = {homogeneous: true, pValue: NaN};
  if(studyType === "independent") {
    homogeneity = leveneTest(group1, group2);
  }

  let count14_1 = group1.filter(x => x >= 14).length;
  let count14_2 = group2.filter(x => x >= 14).length;

  let results = 
    `المجموعة الأولى:\n` +
    `- عدد التلاميذ: ${group1.length}\n` +
    `- المتوسط الحسابي: ${mean1}\n` +
    `- الانحراف المعياري: ${sd1}\n` +
    `- اختبار الاعتدالية: p = ${norm1.pValue.toFixed(3)} → ${norm1.normal ? "توزيع طبيعي" : "ليس طبيعيًا"}\n` +
    `- عدد التلاميذ بمعدل ≥ 14: ${count14_1}\n\n` +

    `المجموعة الثانية:\n` +
    `- عدد التلاميذ: ${group2.length}\n` +
    `- المتوسط الحسابي: ${mean2}\n` +
    `- الانحراف المعياري: ${sd2}\n` +
    `- اختبار الاعتدالية: p = ${norm2.pValue.toFixed(3)} → ${norm2.normal ? "توزيع طبيعي" : "ليس طبيعيًا"}\n` +
    `- عدد التلاميذ بمعدل ≥ 14: ${count14_2}\n\n`;

  let finalAnalysis = "";
  if(studyType === "independent") {
    finalAnalysis += `اختبار تجانس التباين (ليفين): p = ${homogeneity.pValue.toFixed(3)} → ${homogeneity.homogeneous ? "تجانس التباين متحقق" : "تجانس التباين غير متحقق"}\n\n`;
    if(norm1.normal && norm2.normal && homogeneity.homogeneous) {
      let ttest = jStat.ttestTwoSample(group1, group2, 0);
      let df = group1.length + group2.length - 2;
      let t = ttest;
      let p = 2 * (1 - jStat.studentt.cdf(Math.abs(t), df));
      let d = cohensD(group1, group2);
      let size = interpretCohensD(d);
      finalAnalysis +=
        `تم استخدام اختبار t للمجموعتين المستقلتين:\n` +
        `- t = ${t.toFixed(3)}, درجات الحرية = ${df}, p = ${p.toFixed(4)}\n` +
        `${p < 0.05 ? "يوجد فرق معنوي إحصائي بين المجموعتين." : "لا يوجد فرق معنوي إحصائي بين المجموعتين."}\n` +
        `حجم التأثير (Cohen's d): ${d.toFixed(3)} → ${size}\n`;
    } else {
      let mw = mannWhitneyU(group1, group2);
      let r = effectSizeR(mw.U, mw.n1, mw.n2);
      let size = interpretR(r);
      finalAnalysis +=
        `لم تتحقق شروط الاختبار المعلمي، لذلك تم استخدام اختبار مان-ويتني (لامعلمي):\n` +
        `- U = ${mw.U.toFixed(3)}, z = ${mw.z.toFixed(3)}, p = ${mw.p.toFixed(4)}\n` +
        `${mw.p < 0.05 ? "يوجد فرق معنوي إحصائي بين المجموعتين." : "لا يوجد فرق معنوي إحصائي بين المجموعتين."}\n` +
        `حجم التأثير (r): ${r.toFixed(3)} → ${size}\n`;
    }
  } else {
    let diffs = group1.map((v,i) => v - group2[i]);
    let normDiff = testNormality(diffs);
    finalAnalysis +=
      `اختبار الاعتدالية لتوزيع الفروق (قبل - بعد): p = ${normDiff.pValue.toFixed(3)} → ${normDiff.normal ? "توزيع طبيعي" : "ليس طبيعيًا"}\n\n`;
    if(normDiff.normal) {
      let pt = pairedTTest(group1, group2);
      let d = cohensD(group1, group2);
      let size = interpretCohensD(d);
      finalAnalysis +=
        `تم استخدام اختبار t للعينات المرتبطة:\n` +
        `- t = ${pt.t.toFixed(3)}, درجات الحرية = ${pt.df}, p = ${pt.p.toFixed(4)}\n` +
        `${pt.p < 0.05 ? "يوجد فرق معنوي إحصائي بين القياسين." : "لا يوجد فرق معنوي إحصائي بين القياسين."}\n` +
        `حجم التأثير (Cohen's d): ${d.toFixed(3)} → ${size}\n`;
    } else {
      let w = wilcoxonSignedRank(group1, group2);
      let r = Math.abs(w.z) / Math.sqrt(w.n);
      let size = interpretR(r);
      finalAnalysis +=
        `لم يتحقق توزيع طبيعي للفروق، تم استخدام اختبار ويلكوكسون:\n` +
        `- W = ${w.W.toFixed(3)}, z = ${w.z.toFixed(3)}, p = ${w.p.toFixed(4)}\n` +
        `${w.p < 0.05 ? "يوجد فرق معنوي إحصائي بين القياسين." : "لا يوجد فرق معنوي إحصائي بين القياسين."}\n` +
        `حجم التأثير (r): ${r.toFixed(3)} → ${size}\n`;
    }
  }

  postMessage({resultText: results + finalAnalysis});
};
