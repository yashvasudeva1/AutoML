import { ColumnInfo, StatisticalTest, ColumnType, OutlierResult } from "../types";

// Helper mathematical distributions approximations for p-values

// Normal CDF
function normalCDF(z: number): number {
  const b1 =  0.319381530;
  const b2 = -0.356563782;
  const b3 =  1.781477937;
  const b4 = -1.821255978;
  const b5 =  1.330274429;
  const p  =  0.2316419;
  const c  =  0.39894228;

  if (z >= 0.0) {
    const t = 1.0 / (1.0 + p * z);
    return (1.0 - c * Math.exp(-z * z / 2.0) * t *
      (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
  } else {
    const t = 1.0 / (1.0 - p * z);
    return (c * Math.exp(-z * z / 2.0) * t *
      (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1));
  }
}

// Student's T distribution CDF approximation (two-tailed p-value)
export function getStudentT_pValue(t: number, df: number): number {
  t = Math.abs(t);
  if (df <= 0) return 1.0;
  
  // High degrees of freedom -> normal approximation
  if (df > 100) {
    return 2.0 * (1.0 - normalCDF(t));
  }

  // Polynomial approximation for Student-T CDF
  const x = df / (df + t * t);
  let p = 0;
  if (df % 2 === 0) {
    let sum = 1.0;
    let term = 1.0;
    for (let i = 2; i <= df - 2; i += 2) {
      term = term * x * (i - 1) / i;
      sum += term;
    }
    p = 1.0 - Math.sqrt(1.0 - x) * sum;
  } else {
    let sum = 0.0;
    let term = 1.0;
    if (df > 1) {
      sum = 1.0;
      for (let i = 3; i <= df - 2; i += 2) {
        term = term * x * (i - 2) / (i - 1);
        sum += term;
      }
    }
    const theta = Math.atan(t / Math.sqrt(df));
    p = 1.0 - (theta + Math.sin(theta) * Math.cos(theta) * sum) / (Math.PI / 2.0);
  }
  return Math.max(0.0, Math.min(1.0, p));
}

// Chi-Square distribution CDF approximation (p-value)
export function getChiSquare_pValue(chiSq: number, df: number): number {
  if (chiSq < 0 || df <= 0) return 1.0;
  
  // Wilson-Hilferty transformation for Chi-Square to Normal approximation
  const a = 1.0 - 2.0 / (9.0 * df);
  const b = Math.pow(chiSq / df, 1 / 3);
  const z = (b - a) / Math.sqrt(2.0 / (9.0 * df));
  
  const p = 1.0 - normalCDF(z);
  return Math.max(0.0, Math.min(1.0, p));
}

// F-distribution CDF approximation (p-value) for ANOVA
export function getFDistribution_pValue(F: number, df1: number, df2: number): number {
  if (F < 0 || df1 <= 0 || df2 <= 0) return 1.0;
  
  // Paulson's approximation
  const x = F;
  const a = 2.0 / (9.0 * df1);
  const b = 2.0 / (9.0 * df2);
  const numerator = (1.0 - b) * Math.pow(x, 1 / 3) - (1.0 - a);
  const denominator = Math.sqrt(b * Math.pow(x, 2 / 3) + a);
  const z = numerator / denominator;
  
  const p = 1.0 - normalCDF(z);
  return Math.max(0.0, Math.min(1.0, p));
}

// Main Statistics Computations

export function computeColumnStats(values: any[], type: ColumnType): any {
  const total = values.length;
  const validValues = values.filter(v => v !== undefined && v !== null && v !== "");
  const missingCount = total - validValues.length;
  const nonNullCount = validValues.length;

  const uniqueSet = new Set(validValues);
  const uniqueCount = uniqueSet.size;
  const uniquePct = Number(((uniqueCount / Math.max(1, nonNullCount)) * 100).toFixed(2));

  if (type === "Numeric") {
    const numValues = validValues.map(v => Number(v)).filter(v => !isNaN(v));
    if (numValues.length === 0) {
      return { 
        missingCount, 
        uniqueCount, 
        nonNullCount, 
        uniquePct, 
        mean: 0, 
        min: 0, 
        max: 0,
        median: 0,
        stdDev: 0,
        variance: 0,
        kurtosis: 0,
        q25: 0,
        q75: 0,
        range: 0,
        skewness: 0
      };
    }

    numValues.sort((a, b) => a - b);
    const min = numValues[0];
    const max = numValues[numValues.length - 1];
    const range = max - min;
    
    // Mean
    const sum = numValues.reduce((acc, v) => acc + v, 0);
    const mean = sum / numValues.length;

    // Median
    const mid = Math.floor(numValues.length / 2);
    const median = numValues.length % 2 !== 0 
      ? numValues[mid] 
      : (numValues[mid - 1] + numValues[mid]) / 2;

    // Percentiles (Q25 and Q75)
    const getPercentile = (arr: number[], q: number): number => {
      const pos = (arr.length - 1) * q;
      const base = Math.floor(pos);
      const rest = pos - base;
      if (arr[base + 1] !== undefined) {
        return arr[base] + rest * (arr[base + 1] - arr[base]);
      } else {
        return arr[base];
      }
    };
    const q25 = getPercentile(numValues, 0.25);
    const q75 = getPercentile(numValues, 0.75);

    // Standard Deviation, Skewness, Kurtosis
    let sqSum = 0;
    let cubedSum = 0;
    let fourthSum = 0;
    for (const v of numValues) {
      const diff = v - mean;
      sqSum += diff * diff;
      cubedSum += diff * diff * diff;
      fourthSum += diff * diff * diff * diff;
    }
    const variance = sqSum / Math.max(1, numValues.length - 1);
    const stdDev = Math.sqrt(variance);

    // Skewness
    let skewness = 0;
    if (stdDev > 0 && numValues.length > 2) {
      const n = numValues.length;
      const m3 = cubedSum / n;
      const m2 = sqSum / n;
      skewness = (m3 / Math.pow(m2, 1.5)) * Math.sqrt(n * (n - 1)) / (n - 2);
    }

    // Kurtosis (excess kurtosis)
    let kurtosis = 0;
    if (variance > 0 && numValues.length > 3) {
      const n = numValues.length;
      const m4 = fourthSum / n;
      const m2 = sqSum / n;
      kurtosis = (m4 / (m2 * m2)) - 3;
    }

    return {
      mean: Number(mean.toFixed(4)),
      median: Number(median.toFixed(4)),
      min: Number(min.toFixed(4)),
      max: Number(max.toFixed(4)),
      range: Number(range.toFixed(4)),
      q25: Number(q25.toFixed(4)),
      q75: Number(q75.toFixed(4)),
      stdDev: Number(stdDev.toFixed(4)),
      variance: Number(variance.toFixed(4)),
      skewness: Number(skewness.toFixed(4)),
      kurtosis: Number(kurtosis.toFixed(4)),
      missingCount,
      uniqueCount,
      nonNullCount,
      uniquePct,
    };
  } else {
    // Categorical, Text, or Date
    const freqs: { [key: string]: number } = {};
    for (const v of validValues) {
      const strVal = String(v).trim();
      freqs[strVal] = (freqs[strVal] || 0) + 1;
    }

    // Top 20 frequencies for clean rendering and storage
    const sortedFreqs = Object.entries(freqs)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);

    const freqObj: { [key: string]: number } = {};
    for (const [key, count] of sortedFreqs) {
      freqObj[key] = count;
    }

    return {
      missingCount,
      uniqueCount,
      nonNullCount,
      uniquePct,
      frequencies: freqObj,
    };
  }
}

// Compute Pearson Correlation Matrix for continuous variables
export function computeCorrelationMatrix(
  data: any[],
  numericCols: string[]
): { [key: string]: { [key: string]: number } } {
  const matrix: { [key: string]: { [key: string]: number } } = {};

  for (const c1 of numericCols) {
    matrix[c1] = {};
    for (const c2 of numericCols) {
      if (c1 === c2) {
        matrix[c1][c2] = 1.0;
        continue;
      }

      // Compute pairwise Pearson correlation
      const pairs: [number, number][] = [];
      for (const row of data) {
        const v1 = Number(row[c1]);
        const v2 = Number(row[c2]);
        if (!isNaN(v1) && !isNaN(v2) && row[c1] !== null && row[c2] !== null) {
          pairs.push([v1, v2]);
        }
      }

      if (pairs.length < 3) {
        matrix[c1][c2] = 0;
        continue;
      }

      const mean1 = pairs.reduce((sum, p) => sum + p[0], 0) / pairs.length;
      const mean2 = pairs.reduce((sum, p) => sum + p[1], 0) / pairs.length;

      let num = 0;
      let den1 = 0;
      let den2 = 0;

      for (const [x, y] of pairs) {
        const dx = x - mean1;
        const dy = y - mean2;
        num += dx * dy;
        den1 += dx * dx;
        den2 += dy * dy;
      }

      const r = den1 * den2 === 0 ? 0 : num / Math.sqrt(den1 * den2);
      matrix[c1][c2] = Number(r.toFixed(4));
    }
  }

  return matrix;
}

// Automated statistical tests selector and runner
export function runStatisticalTests(
  data: any[],
  columns: ColumnInfo[]
): StatisticalTest[] {
  const tests: StatisticalTest[] = [];

  const numericCols = columns.filter(c => c.type === "Numeric").map(c => c.name);
  const categoricalCols = columns.filter(c => c.type === "Categorical" && c.stats.uniqueCount <= 10).map(c => c.name);

  // 1. T-Test or ANOVA (Compare Numeric column across Categorical column groups)
  if (numericCols.length > 0 && categoricalCols.length > 0) {
    const numCol = numericCols[0];
    const catCol = categoricalCols[0];

    // Group the numeric values
    const groups: { [key: string]: number[] } = {};
    for (const row of data) {
      const catVal = String(row[catCol]).trim();
      const numVal = Number(row[numCol]);
      if (catVal !== undefined && catVal !== "null" && !isNaN(numVal) && row[catCol] !== null && row[numCol] !== null) {
        if (!groups[catVal]) groups[catVal] = [];
        groups[catVal].push(numVal);
      }
    }

    const groupKeys = Object.keys(groups).filter(k => groups[k].length >= 3);

    if (groupKeys.length === 2) {
      // 2-sample Independent T-Test (Welch's T-Test)
      const groupA = groups[groupKeys[0]];
      const groupB = groups[groupKeys[1]];

      const meanA = groupA.reduce((sum, v) => sum + v, 0) / groupA.length;
      const meanB = groupB.reduce((sum, v) => sum + v, 0) / groupB.length;

      const varA = groupA.reduce((sum, v) => sum + Math.pow(v - meanA, 2), 0) / (groupA.length - 1);
      const varB = groupB.reduce((sum, v) => sum + Math.pow(v - meanB, 2), 0) / (groupB.length - 1);

      const se = Math.sqrt((varA / groupA.length) + (varB / groupB.length));
      if (se > 0) {
        const tStat = (meanA - meanB) / se;
        // Welch-Satterthwaite DF
        const dfNumerator = Math.pow((varA / groupA.length) + (varB / groupB.length), 2);
        const dfDenominator = (Math.pow(varA / groupA.length, 2) / (groupA.length - 1)) +
                              (Math.pow(varB / groupB.length, 2) / (groupB.length - 1));
        const df = dfDenominator > 0 ? Math.round(dfNumerator / dfDenominator) : groupA.length + groupB.length - 2;

        const pVal = getStudentT_pValue(tStat, df);
        const significance = pVal < 0.05 ? "Significant" : "Not significant";

        tests.push({
          type: "T-Test",
          variable1: numCol,
          variable2: catCol,
          statisticValue: Number(tStat.toFixed(4)),
          pValue: Number(pVal.toFixed(6)),
          degreesOfFreedom: df,
          interpretation: `Compared means of '${numCol}' across binary categories of '${catCol}' ('${groupKeys[0]}' vs '${groupKeys[1]}'). Result is ${significance} (p = ${pVal.toFixed(4)}) with an average difference of ${(meanA - meanB).toFixed(2)}.`
        });
      }
    } else if (groupKeys.length > 2) {
      // One-way ANOVA
      const k = groupKeys.length;
      const nTotal = groupKeys.reduce((sum, k) => sum + groups[k].length, 0);

      // Global Mean
      const grandSum = groupKeys.reduce((sum, k) => sum + groups[k].reduce((s, v) => s + v, 0), 0);
      const grandMean = grandSum / nTotal;

      // Sum of Squares Between (SSB)
      let ssb = 0;
      for (const key of groupKeys) {
        const n_g = groups[key].length;
        const mean_g = groups[key].reduce((s, v) => s + v, 0) / n_g;
        ssb += n_g * Math.pow(mean_g - grandMean, 2);
      }

      // Sum of Squares Within (SSW)
      let ssw = 0;
      for (const key of groupKeys) {
        const groupVals = groups[key];
        const mean_g = groupVals.reduce((s, v) => s + v, 0) / groupVals.length;
        for (const val of groupVals) {
          ssw += Math.pow(val - mean_g, 2);
        }
      }

      const dfBetween = k - 1;
      const dfWithin = nTotal - k;

      const msb = ssb / dfBetween;
      const msw = ssw / dfWithin;

      if (msw > 0 && dfWithin > 0) {
        const F = msb / msw;
        const pVal = getFDistribution_pValue(F, dfBetween, dfWithin);
        const significance = pVal < 0.05 ? "Significant" : "Not significant";

        tests.push({
          type: "ANOVA",
          variable1: numCol,
          variable2: catCol,
          statisticValue: Number(F.toFixed(4)),
          pValue: Number(pVal.toFixed(6)),
          degreesOfFreedom: dfBetween + dfWithin,
          interpretation: `One-way ANOVA conducted on '${numCol}' grouped by '${catCol}' (${k} groups). Result is ${significance} (F = ${F.toFixed(2)}, p = ${pVal.toFixed(4)}), indicating that the means across groups ${pVal < 0.05 ? 'differ significantly' : 'do not differ significantly'}.`
        });
      }
    }
  }

  // 2. Chi-Square Test of Independence (Between two categorical columns)
  if (categoricalCols.length >= 2) {
    const colA = categoricalCols[0];
    const colB = categoricalCols[1];

    // Find unique levels
    const levelsA = Array.from(new Set(data.map(row => String(row[colA]).trim()).filter(v => v !== "null" && v !== "")));
    const levelsB = Array.from(new Set(data.map(row => String(row[colB]).trim()).filter(v => v !== "null" && v !== "")));

    if (levelsA.length >= 2 && levelsA.length <= 10 && levelsB.length >= 2 && levelsB.length <= 10) {
      // Build Observed contingency table
      const obsTable: { [key: string]: { [key: string]: number } } = {};
      const rowTotals: { [key: string]: number } = {};
      const colTotals: { [key: string]: number } = {};
      let totalValid = 0;

      for (const r of levelsA) {
        obsTable[r] = {};
        for (const c of levelsB) {
          obsTable[r][c] = 0;
        }
        rowTotals[r] = 0;
      }
      for (const c of levelsB) {
        colTotals[c] = 0;
      }

      for (const row of data) {
        const valA = String(row[colA]).trim();
        const valB = String(row[colB]).trim();
        if (valA && valB && obsTable[valA] && obsTable[valA][valB] !== undefined) {
          obsTable[valA][valB]++;
          rowTotals[valA]++;
          colTotals[valB]++;
          totalValid++;
        }
      }

      // Calculate Chi-Square statistic
      let chiSquare = 0;
      for (const r of levelsA) {
        for (const c of levelsB) {
          const expected = (rowTotals[r] * colTotals[c]) / totalValid;
          if (expected > 0) {
            const observed = obsTable[r][c];
            chiSquare += Math.pow(observed - expected, 2) / expected;
          }
        }
      }

      const df = (levelsA.length - 1) * (levelsB.length - 1);
      if (df > 0 && totalValid > 0) {
        const pVal = getChiSquare_pValue(chiSquare, df);
        const significance = pVal < 0.05 ? "Significant relationship" : "Independent (Not significant)";

        tests.push({
          type: "Chi-Square",
          variable1: colA,
          variable2: colB,
          statisticValue: Number(chiSquare.toFixed(4)),
          pValue: Number(pVal.toFixed(6)),
          degreesOfFreedom: df,
          interpretation: `Chi-Square test of independence between categorical variables '${colA}' (${levelsA.length} categories) and '${colB}' (${levelsB.length} categories). Result is ${significance} (Chi-Sq = ${chiSquare.toFixed(2)}, p = ${pVal.toFixed(4)}).`
        });
      }
    }
  }

  // 3. Pearson Correlation significance (Between top two correlated numerical columns)
  if (numericCols.length >= 2) {
    const c1 = numericCols[0];
    const c2 = numericCols[1];

    const pairs: [number, number][] = [];
    for (const row of data) {
      const v1 = Number(row[c1]);
      const v2 = Number(row[c2]);
      if (!isNaN(v1) && !isNaN(v2) && row[c1] !== null && row[c2] !== null) {
        pairs.push([v1, v2]);
      }
    }

    if (pairs.length >= 5) {
      const mean1 = pairs.reduce((sum, p) => sum + p[0], 0) / pairs.length;
      const mean2 = pairs.reduce((sum, p) => sum + p[1], 0) / pairs.length;

      let num = 0;
      let den1 = 0;
      let den2 = 0;

      for (const [x, y] of pairs) {
        const dx = x - mean1;
        const dy = y - mean2;
        num += dx * dy;
        den1 += dx * dx;
        den2 += dy * dy;
      }

      const r = den1 * den2 === 0 ? 0 : num / Math.sqrt(den1 * den2);
      const df = pairs.length - 2;

      // t statistic for correlation coefficient
      let tStat = 0;
      if (Math.abs(r) < 1.0) {
        tStat = r * Math.sqrt(df / (1 - r * r));
      } else {
        tStat = 999 * Math.sign(r);
      }

      const pVal = getStudentT_pValue(tStat, df);
      const strength = Math.abs(r) > 0.7 ? "Strong" : Math.abs(r) > 0.4 ? "Moderate" : "Weak";
      const direct = r > 0 ? "positive" : "negative";

      tests.push({
        type: "Pearson Correlation",
        variable1: c1,
        variable2: c2,
        statisticValue: Number(r.toFixed(4)),
        pValue: Number(pVal.toFixed(6)),
        degreesOfFreedom: df,
        interpretation: `Correlation test between continuous variables '${c1}' and '${c2}'. Found a ${strength} ${direct} correlation (r = ${r.toFixed(2)}, p = ${pVal.toFixed(4)}).`
      });
    }
  }

  return tests;
}

/**
 * Detects and highlights outliers in numerical columns of a dataset.
 * Uses the Interquartile Range (IQR) method as a default, falling back
 * to standard Z-scores if the IQR is 0 (due to many identical values).
 */
export function detectDatasetOutliers(data: any[], columns: ColumnInfo[]): OutlierResult[] {
  const results: OutlierResult[] = [];

  for (const col of columns) {
    if (col.type !== "Numeric") continue;

    const stats = col.stats;
    const mean = stats.mean ?? 0;
    const stdDev = stats.stdDev ?? 0;
    const q25 = stats.q25 ?? 0;
    const q75 = stats.q75 ?? 0;

    const iqr = q75 - q25;
    let lowerBound = q25 - 1.5 * iqr;
    let upperBound = q75 + 1.5 * iqr;
    let method = "IQR";

    if (iqr === 0) {
      if (stdDev > 0) {
        lowerBound = mean - 3 * stdDev;
        upperBound = mean + 3 * stdDev;
        method = "Z-Score";
      } else {
        // All values are identical, no outliers possible
        continue;
      }
    }

    const outlierIndices: number[] = [];
    const outlierValues: number[] = [];

    data.forEach((row, idx) => {
      const val = row[col.name];
      if (val !== undefined && val !== null && val !== "") {
        const numVal = Number(val);
        if (!isNaN(numVal)) {
          if (numVal < lowerBound || numVal > upperBound) {
            outlierIndices.push(idx);
            outlierValues.push(numVal);
          }
        }
      }
    });

    const totalValid = col.stats.nonNullCount || 1;
    const outlierCount = outlierIndices.length;
    const outlierPercentage = Number(((outlierCount / totalValid) * 100).toFixed(2));

    // Signify statistically significant anomaly if there are outliers and either:
    // - Outlier count is more than 2% of non-null records (indicating high frequency of anomalies)
    // - There is an extreme outlier that resides more than 3.5 standard deviations away from the mean
    const isSignificant = outlierCount > 0 && (
      outlierPercentage > 2.0 || 
      outlierValues.some(v => Math.abs((v - mean) / (stdDev || 1)) > 3.5)
    );

    // Sort outlier values by absolute distance from mean descending
    const sortedOutliers = [...outlierValues].sort((a, b) => {
      const distA = Math.abs(a - mean);
      const distB = Math.abs(b - mean);
      return distB - distA;
    });

    results.push({
      columnName: col.name,
      outlierCount,
      outlierPercentage,
      lowerBound: Number(lowerBound.toFixed(4)),
      upperBound: Number(upperBound.toFixed(4)),
      isSignificant,
      method,
      extremeValues: sortedOutliers.slice(0, 5).map(v => Number(v.toFixed(4))),
      indices: outlierIndices
    });
  }

  return results;
}
