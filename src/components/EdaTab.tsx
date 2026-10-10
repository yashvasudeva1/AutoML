import React, { useState, useMemo, useEffect, useRef } from "react";
import { DatasetSummary, ColumnInfo } from "../types";
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  Cell, PieChart, Pie, Legend,
  ComposedChart, Scatter, Line, CartesianGrid, ZAxis
} from "recharts";
import { 
  BarChart3, PieChartIcon, Grid3X3, ArrowRight, Activity,
  Sparkles, Pin, HelpCircle, Loader2, TrendingUp, AlertTriangle,
  Search, Sliders, Maximize2, X, Filter, BarChart2, Check, ShieldAlert,
  ChevronRight, ArrowUpDown, Layers
} from "lucide-react";

interface EdaTabProps {
  summary: DatasetSummary;
  rawData?: any[] | null;
}

export interface HistogramBin {
  binIndex: number;
  minVal: number;
  maxVal: number;
  binLabel: string;
  count: number;
  percentage: number;
}

export interface ColumnHistogramProfile {
  column: ColumnInfo;
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
  skewness: number;
  q25: number;
  q75: number;
  p10: number;
  p90: number;
  iqr: number;
  lowerBound: number;
  upperBound: number;
  outlierCount: number;
  missingCount: number;
  missingPercentage: number;
  validCount: number;
  distributionType: "Normal" | "Right-Skewed" | "Left-Skewed" | "Uniform" | "Constant";
  bins: HistogramBin[];
}

export default function EdaTab({ summary, rawData }: EdaTabProps) {
  const [selectedColName, setSelectedColName] = useState<string>(
    summary.columns[0]?.name || ""
  );

  // Deep Profiling state for numeric columns histograms
  const [profileBinCount, setProfileBinCount] = useState<number>(10);
  const [profileSearchQuery, setProfileSearchQuery] = useState<string>("");
  const [profileSortBy, setProfileSortBy] = useState<"default" | "skewness" | "stdDev" | "outliers" | "missing">("default");
  const [profileGridCols, setProfileGridCols] = useState<2 | 3>(2);
  const [inspectedColumnName, setInspectedColumnName] = useState<string | null>(null);
  const [activeEdaSection, setActiveEdaSection] = useState<"profiling" | "single" | "scatter" | "correlation">("profiling");

  const selectedCol = useMemo(() => {
    return summary.columns.find(c => c.name === selectedColName) || null;
  }, [selectedColName, summary.columns]);

  // Scatter & Trend segments explorer state
  const numericCols = useMemo(() => {
    return summary.columns.filter(c => c.type === "Numeric").map(c => c.name);
  }, [summary.columns]);

  const [selectedX, setSelectedX] = useState<string>("");
  const [selectedY, setSelectedY] = useState<string>("");

  const numericColsKey = numericCols.join(",");
  useEffect(() => {
    if (numericCols.length > 0) {
      setSelectedX(prev => prev && numericCols.includes(prev) ? prev : numericCols[0]);
      setSelectedY(prev => prev && numericCols.includes(prev) ? prev : (numericCols[1] || numericCols[0]));
    }
  }, [numericColsKey]);

  const activeX = useMemo(() => {
    return numericCols.includes(selectedX) ? selectedX : (numericCols[0] || "");
  }, [selectedX, numericCols]);

  const activeY = useMemo(() => {
    return numericCols.includes(selectedY) ? selectedY : (numericCols[1] || numericCols[0] || "");
  }, [selectedY, numericCols]);

  const plotRows = useMemo(() => {
    return rawData || summary.sampleData || [];
  }, [rawData, summary.sampleData]);

  // Deep Profiling histogram calculation for every numeric column
  const numericColumnProfiles = useMemo<ColumnHistogramProfile[]>(() => {
    const numericColumns = summary.columns.filter(c => c.type === "Numeric");
    if (numericColumns.length === 0 || plotRows.length === 0) return [];

    return numericColumns.map(col => {
      // Gather all valid numeric values for this column
      const rawValues = plotRows
        .map(r => Number(r[col.name]))
        .filter(v => !isNaN(v) && v !== null && v !== undefined);

      const validCount = rawValues.length;
      const missingCount = plotRows.length - validCount;
      const missingPercentage = plotRows.length > 0 ? (missingCount / plotRows.length) * 100 : 0;

      if (validCount === 0) {
        return {
          column: col,
          min: 0,
          max: 0,
          mean: 0,
          median: 0,
          stdDev: 0,
          skewness: 0,
          q25: 0,
          q75: 0,
          p10: 0,
          p90: 0,
          iqr: 0,
          lowerBound: 0,
          upperBound: 0,
          outlierCount: 0,
          missingCount,
          missingPercentage,
          validCount: 0,
          distributionType: "Constant" as const,
          bins: []
        };
      }

      // Sort values for quantiles and binning
      const sorted = [...rawValues].sort((a, b) => a - b);
      const min = sorted[0];
      const max = sorted[sorted.length - 1];
      const sum = sorted.reduce((acc, v) => acc + v, 0);
      const mean = sum / validCount;

      // Median
      const mid = Math.floor(validCount / 2);
      const median = validCount % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;

      // Quantiles
      const p10 = sorted[Math.floor(validCount * 0.10)] ?? min;
      const q25 = sorted[Math.floor(validCount * 0.25)] ?? min;
      const q75 = sorted[Math.floor(validCount * 0.75)] ?? max;
      const p90 = sorted[Math.floor(validCount * 0.90)] ?? max;
      const iqr = q75 - q25;
      const lowerBound = q25 - 1.5 * iqr;
      const upperBound = q75 + 1.5 * iqr;

      // Variance & StdDev
      const variance = validCount > 1 
        ? sorted.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / (validCount - 1)
        : 0;
      const stdDev = Math.sqrt(variance);

      // Skewness calculation (Fisher-Pearson coefficient)
      let skewness = 0;
      if (stdDev > 0 && validCount > 2) {
        const m3 = sorted.reduce((acc, v) => acc + Math.pow((v - mean) / stdDev, 3), 0);
        skewness = (validCount / ((validCount - 1) * (validCount - 2))) * m3;
      }

      // Distribution type classification
      let distributionType: "Normal" | "Right-Skewed" | "Left-Skewed" | "Uniform" | "Constant" = "Normal";
      if (min === max || stdDev === 0) {
        distributionType = "Constant";
      } else if (skewness > 0.6) {
        distributionType = "Right-Skewed";
      } else if (skewness < -0.6) {
        distributionType = "Left-Skewed";
      } else {
        distributionType = "Normal";
      }

      // Outlier count
      const foundOutliers = summary.outliers?.find(o => o.columnName === col.name);
      const outlierCount = foundOutliers ? foundOutliers.outlierCount : sorted.filter(v => v < lowerBound || v > upperBound).length;

      // Compute histogram bins
      const bins: HistogramBin[] = [];
      const numBins = Math.max(4, Math.min(25, profileBinCount));

      if (min === max) {
        bins.push({
          binIndex: 0,
          minVal: min,
          maxVal: max,
          binLabel: `${min}`,
          count: validCount,
          percentage: 100
        });
      } else {
        const step = (max - min) / numBins;
        const isInteger = sorted.every(v => Number.isInteger(v));

        for (let b = 0; b < numBins; b++) {
          const binMin = min + b * step;
          const binMax = min + (b + 1) * step;
          
          let label = "";
          if (isInteger && step >= 1) {
            label = `${Math.round(binMin)} - ${Math.round(binMax)}`;
          } else if (Math.abs(binMin) < 10 && Math.abs(binMax) < 10) {
            label = `${binMin.toFixed(2)} - ${binMax.toFixed(2)}`;
          } else {
            label = `${binMin.toFixed(1)} - ${binMax.toFixed(1)}`;
          }

          bins.push({
            binIndex: b,
            minVal: Number(binMin.toFixed(4)),
            maxVal: Number(binMax.toFixed(4)),
            binLabel: label,
            count: 0,
            percentage: 0
          });
        }

        // Fill counts
        for (const val of sorted) {
          let placed = false;
          for (let b = 0; b < numBins; b++) {
            if (val >= bins[b].minVal && val < bins[b].maxVal) {
              bins[b].count++;
              placed = true;
              break;
            }
          }
          if (!placed && val === max) {
            bins[numBins - 1].count++;
          }
        }

        bins.forEach(b => {
          b.percentage = Number(((b.count / validCount) * 100).toFixed(1));
        });
      }

      return {
        column: col,
        min: Number(min.toFixed(3)),
        max: Number(max.toFixed(3)),
        mean: Number(mean.toFixed(3)),
        median: Number(median.toFixed(3)),
        stdDev: Number(stdDev.toFixed(3)),
        skewness: Number(skewness.toFixed(3)),
        q25: Number(q25.toFixed(3)),
        q75: Number(q75.toFixed(3)),
        p10: Number(p10.toFixed(3)),
        p90: Number(p90.toFixed(3)),
        iqr: Number(iqr.toFixed(3)),
        lowerBound: Number(lowerBound.toFixed(3)),
        upperBound: Number(upperBound.toFixed(3)),
        outlierCount,
        missingCount,
        missingPercentage: Number(missingPercentage.toFixed(1)),
        validCount,
        distributionType,
        bins
      };
    });
  }, [summary.columns, summary.outliers, plotRows, profileBinCount]);

  const filteredProfiles = useMemo(() => {
    let list = numericColumnProfiles;
    if (profileSearchQuery.trim()) {
      const q = profileSearchQuery.toLowerCase();
      list = list.filter(p => p.column.name.toLowerCase().includes(q));
    }

    if (profileSortBy === "skewness") {
      return [...list].sort((a, b) => Math.abs(b.skewness) - Math.abs(a.skewness));
    } else if (profileSortBy === "stdDev") {
      return [...list].sort((a, b) => b.stdDev - a.stdDev);
    } else if (profileSortBy === "outliers") {
      return [...list].sort((a, b) => b.outlierCount - a.outlierCount);
    } else if (profileSortBy === "missing") {
      return [...list].sort((a, b) => b.missingCount - a.missingCount);
    }
    return list;
  }, [numericColumnProfiles, profileSearchQuery, profileSortBy]);

  const inspectedProfile = useMemo(() => {
    if (!inspectedColumnName) return null;
    return numericColumnProfiles.find(p => p.column.name === inspectedColumnName) || null;
  }, [numericColumnProfiles, inspectedColumnName]);

  // Outlier filter view modes:
  const [outlierViewMode, setOutlierViewMode] = useState<"all" | "clean-active" | "clean-all">("all");

  const xOutliers = useMemo(() => {
    return summary.outliers?.find(o => o.columnName === activeX)?.indices || [];
  }, [summary.outliers, activeX]);

  const yOutliers = useMemo(() => {
    return summary.outliers?.find(o => o.columnName === activeY)?.indices || [];
  }, [summary.outliers, activeY]);

  const allOutliers = useMemo(() => {
    const set = new Set<number>();
    summary.outliers?.forEach(o => {
      o.indices.forEach(idx => set.add(idx));
    });
    return Array.from(set);
  }, [summary.outliers]);

  const xOutlierSet = useMemo(() => new Set(xOutliers), [xOutliers]);
  const yOutlierSet = useMemo(() => new Set(yOutliers), [yOutliers]);
  const allOutliersSet = useMemo(() => {
    const set = new Set<number>();
    summary.outliers?.forEach(o => {
      o.indices.forEach(idx => set.add(idx));
    });
    return set;
  }, [summary.outliers]);

  const parsedPoints = useMemo(() => {
    if (!activeX || !activeY) return [];
    return plotRows.map((row, idx) => {
      const isXOutlier = xOutlierSet.has(idx);
      const isYOutlier = yOutlierSet.has(idx);
      return {
        x: Number(row[activeX]),
        y: Number(row[activeY]),
        id: idx,
        isXOutlier,
        isYOutlier,
        isOutlier: isXOutlier || isYOutlier,
        isAnyOutlier: allOutliersSet.has(idx),
        payload: row
      };
    }).filter(pt => !isNaN(pt.x) && !isNaN(pt.y));
  }, [plotRows, activeX, activeY, xOutlierSet, yOutlierSet, allOutliersSet]);

  const scatterData = useMemo(() => {
    if (outlierViewMode === "clean-active") {
      return parsedPoints.filter(p => !p.isOutlier);
    }
    if (outlierViewMode === "clean-all") {
      return parsedPoints.filter(p => !p.isAnyOutlier);
    }
    return parsedPoints;
  }, [parsedPoints, outlierViewMode]);

  // Linear Regression & Correlation Computation
  const regression = useMemo(() => {
    const pts = scatterData;
    const n = pts.length;
    if (n < 2) return null;

    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumX2 = 0;
    let sumY2 = 0;

    let minX = Infinity;
    let maxX = -Infinity;

    for (const p of pts) {
      sumX += p.x;
      sumY += p.y;
      sumXY += p.x * p.y;
      sumX2 += p.x * p.x;
      sumY2 += p.y * p.y;
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
    }

    const denom = n * sumX2 - sumX * sumX;
    if (denom === 0) return null;

    const slope = (n * sumXY - sumX * sumY) / denom;
    const intercept = (sumY - slope * sumX) / n;

    const numCorr = n * sumXY - sumX * sumY;
    const denCorr = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
    const correlation = denCorr !== 0 ? numCorr / denCorr : 0;

    const points = [
      { x: minX, y: slope * minX + intercept },
      { x: maxX, y: slope * maxX + intercept }
    ];

    return { slope, intercept, correlation, points };
  }, [scatterData]);

  const trendData = useMemo(() => {
    if (!regression) return [];
    return regression.points;
  }, [regression]);

  // Hovered Point & Pinned Explanation State
  const [hoveredPoint, setHoveredPoint] = useState<{ x: number; y: number; isTrendLine?: boolean } | null>(null);
  const [pinnedPoint, setPinnedPoint] = useState<{ x: number; y: number; isTrendLine?: boolean } | null>(null);
  const [explanation, setExplanation] = useState<string | null>(null);
  const [isLoadingExpl, setIsLoadingExpl] = useState<boolean>(false);
  const [explCache, setExplCache] = useState<{ [key: string]: string }>({});

  const hoverTimeoutRef = useRef<any>(null);

  const activeExplorerPoint = pinnedPoint || hoveredPoint;

  const fetchExplanation = async (x: number, y: number, isTrendLine = false) => {
    if (!activeX || !activeY || !regression) return;

    const segment = isTrendLine
      ? "Directly along linear trend line"
      : regression.slope > 0
        ? y > (regression.slope * x + regression.intercept) ? "Above positive trend line" : "Below positive trend line"
        : y > (regression.slope * x + regression.intercept) ? "Above negative trend line" : "Below negative trend line";

    const cacheKey = `${activeX}_${activeY}_${x.toFixed(2)}_${y.toFixed(2)}_${isTrendLine}`;
    if (explCache[cacheKey]) {
      setExplanation(explCache[cacheKey]);
      return;
    }

    setIsLoadingExpl(true);
    setExplanation(null);

    try {
      const response = await fetch("/api/eda/explain-segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          xCol: activeX,
          yCol: activeY,
          xVal: x,
          yVal: y,
          correlation: regression.correlation,
          slope: regression.slope,
          segment,
          fileName: summary.fileName
        })
      });

      if (response.ok) {
        const data = await response.json();
        const explText = data.explanation || "No explanation returned.";
        setExplanation(explText);
        setExplCache(prev => ({ ...prev, [cacheKey]: explText }));
      } else {
        setExplanation("Unable to generate real-time insight for this segment.");
      }
    } catch (err) {
      console.error("Failed to fetch segment explanation:", err);
      setExplanation("Offline or connection error. Unable to analyze trend segment.");
    } finally {
      setIsLoadingExpl(false);
    }
  };

  const handlePointHover = (data: any) => {
    if (!data) return;
    const x = data.x !== undefined ? data.x : data.payload?.x;
    const y = data.y !== undefined ? data.y : data.payload?.y;
    
    if (x === undefined || y === undefined) return;

    setHoveredPoint({ x, y, isTrendLine: false });

    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }

    if (!pinnedPoint) {
      hoverTimeoutRef.current = setTimeout(() => {
        fetchExplanation(x, y, false);
      }, 550);
    }
  };

  const handleTrendLineHover = () => {
    if (!regression) return;
    const x = (regression.points[0]?.x + regression.points[1]?.x) / 2 || 0;
    const y = (regression.points[0]?.y + regression.points[1]?.y) / 2 || 0;

    setHoveredPoint({ x, y, isTrendLine: true });

    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }

    if (!pinnedPoint) {
      hoverTimeoutRef.current = setTimeout(() => {
        fetchExplanation(x, y, true);
      }, 550);
    }
  };

  const handlePointLeave = () => {
    if (hoverTimeoutRef.current) {
      clearTimeout(hoverTimeoutRef.current);
    }
    setHoveredPoint(null);
  };

  const handlePointClick = (data: any) => {
    if (!data) return;
    const x = data.x !== undefined ? data.x : data.payload?.x;
    const y = data.y !== undefined ? data.y : data.payload?.y;
    if (x === undefined || y === undefined) return;

    const clicked = { x, y, isTrendLine: false };
    setPinnedPoint(clicked);
    fetchExplanation(x, y, false);
  };

  const handleTogglePin = () => {
    if (pinnedPoint) {
      setPinnedPoint(null);
    } else if (hoveredPoint) {
      setPinnedPoint(hoveredPoint);
    }
  };

  // Histogram Binning for Numeric Columns
  const distributionData = useMemo(() => {
    if (!selectedCol || selectedCol.type !== "Numeric" || summary.sampleData.length === 0) {
      return [];
    }

    const { min, max } = selectedCol.stats;
    if (min === undefined || max === undefined || min === max) {
      return [{ binLabel: `${min}`, count: summary.sampleData.length }];
    }

    const numBins = 8;
    const binWidth = (max - min) / numBins;
    const bins = Array(numBins).fill(0).map((_, i) => ({
      minVal: min + i * binWidth,
      maxVal: min + (i + 1) * binWidth,
      binLabel: `${(min + i * binWidth).toFixed(1)} - ${(min + (i + 1) * binWidth).toFixed(1)}`,
      count: 0
    }));

    for (const row of summary.sampleData) {
      const val = Number(row[selectedCol.name]);
      if (!isNaN(val) && val !== null && row[selectedCol.name] !== "") {
        let placed = false;
        for (let i = 0; i < numBins; i++) {
          if (val >= bins[i].minVal && val < bins[i].maxVal) {
            bins[i].count++;
            placed = true;
            break;
          }
        }
        if (!placed && val === max) {
          bins[numBins - 1].count++;
        }
      }
    }

    return bins;
  }, [selectedCol, summary.sampleData]);

  // Frequency mapping for Categorical Columns
  const categoricalData = useMemo(() => {
    if (!selectedCol || selectedCol.type === "Numeric" || !selectedCol.stats.frequencies) {
      return [];
    }

    return Object.entries(selectedCol.stats.frequencies).map(([key, count]) => ({
      name: key || "(Blank)",
      value: count as number
    })).sort((a, b) => (b.value as number) - (a.value as number));
  }, [selectedCol]);

  // Warm Claude-style palette for charts
  const PIE_COLORS = ["#0284C7", "#38BDF8", "#8C7E72", "#635850", "#7DD3FC", "#0369A1", "#5E503F", "#9E9D96"];

  // Heatmap styling helper
  const getHeatmapBg = (val: number) => {
    const abs = Math.abs(val);
    if (val > 0) {
      if (abs > 0.8) return "bg-[#0284C7] text-white font-mono font-bold";
      if (abs > 0.6) return "bg-[#0284C7]/70 text-white font-mono";
      if (abs > 0.4) return "bg-[#0284C7]/35 text-[#0284C7] font-mono";
      if (abs > 0.2) return "bg-[#0284C7]/10 text-[#0284C7] font-mono";
      return "text-[#73726C] font-mono";
    } else {
      if (abs > 0.8) return "bg-[#73726C] text-white font-mono font-bold";
      if (abs > 0.6) return "bg-[#73726C]/70 text-white font-mono";
      if (abs > 0.4) return "bg-[#73726C]/35 text-[#73726C] font-mono";
      if (abs > 0.2) return "bg-[#73726C]/10 text-[#73726C] font-mono";
      return "text-[#73726C] font-mono";
    }
  };

  return (
    <div id="eda-tab-content" className="space-y-6">
      
      {/* 1. Deep Profiling: All Numeric Feature Histograms */}
      <div id="deep-profiling-section" className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-4 shadow-xs">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
          <div>
            <div className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4 text-[#0284C7]" />
              <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                Deep Profiling: Numeric Feature Histograms
              </h3>
            </div>
            <p className="text-xs text-[#73726C] mt-0.5">
              Comprehensive frequency distribution histograms, central tendencies, and distribution skewness for every numeric column in the dataset.
            </p>
          </div>

          {/* Quick Metrics Badges */}
          <div className="flex items-center gap-2 flex-wrap text-[11px] font-mono">
            <span className="px-2.5 py-1 rounded-lg bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] font-medium">
              {numericColumnProfiles.length} Numeric Features
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-emerald-50 border border-emerald-200 text-emerald-800">
              {numericColumnProfiles.filter(p => p.distributionType === "Normal").length} Symmetric
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-amber-50 border border-amber-200 text-amber-800">
              {numericColumnProfiles.filter(p => p.distributionType.includes("Skewed")).length} Skewed
            </span>
            <span className="px-2.5 py-1 rounded-lg bg-sky-50 border border-[#BAE6FD] text-[#0284C7]">
              {plotRows.length.toLocaleString()} Records Sampled
            </span>
          </div>
        </div>

        {/* Profiling Controls Toolbar */}
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-[#FAF9F5] p-2.5 rounded-xl border border-[#E5E2D9] text-xs">
          
          {/* Left: Search input */}
          <div className="relative flex-1 max-w-xs">
            <Search className="h-3.5 w-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-[#73726C]" />
            <input
              type="text"
              placeholder="Search numeric column..."
              value={profileSearchQuery}
              onChange={(e) => setProfileSearchQuery(e.target.value)}
              className="w-full bg-white border border-[#E5E2D9] rounded-lg pl-8 pr-3 py-1.5 text-xs text-[#1F1E1D] placeholder-[#9E9D96] focus:border-[#0284C7] outline-none"
            />
            {profileSearchQuery && (
              <button 
                onClick={() => setProfileSearchQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[#73726C] hover:text-[#1F1E1D] cursor-pointer"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>

          {/* Right: Controls & Toggles */}
          <div className="flex items-center gap-2 flex-wrap">
            
            {/* Bin Count Selector */}
            <div className="flex items-center gap-1 font-mono text-[11px]">
              <span className="text-[#73726C] mr-0.5">Bins:</span>
              {[8, 10, 15, 20].map((b) => (
                <button
                  key={b}
                  onClick={() => setProfileBinCount(b)}
                  className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
                    profileBinCount === b
                      ? "bg-[#0284C7] text-white border-[#0284C7] font-semibold"
                      : "bg-white text-[#73726C] border-[#E5E2D9] hover:text-[#1F1E1D]"
                  }`}
                >
                  {b}
                </button>
              ))}
            </div>

            {/* Sort Dropdown */}
            <div className="flex items-center space-x-1 text-xs">
              <span className="text-[#73726C] text-[11px]">Sort:</span>
              <select
                value={profileSortBy}
                onChange={(e: any) => setProfileSortBy(e.target.value)}
                className="bg-white border border-[#E5E2D9] text-[#1F1E1D] rounded-lg px-2 py-1 text-xs outline-none cursor-pointer"
              >
                <option value="default">Schema Order</option>
                <option value="skewness">Highest Skewness</option>
                <option value="stdDev">Highest Std Dev (Variance)</option>
                <option value="outliers">Most Outliers</option>
                <option value="missing">Most Missing Values</option>
              </select>
            </div>

            {/* Grid Density Toggle */}
            <div className="flex items-center gap-1 font-mono text-[11px] border-l border-[#E5E2D9] pl-2">
              <button
                onClick={() => setProfileGridCols(2)}
                className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
                  profileGridCols === 2
                    ? "bg-white text-[#0284C7] border-[#0284C7] font-semibold shadow-2xs"
                    : "bg-[#FAF9F5] text-[#73726C] border-[#E5E2D9] hover:text-[#1F1E1D]"
                }`}
                title="2 Columns Grid"
              >
                2-Col
              </button>
              <button
                onClick={() => setProfileGridCols(3)}
                className={`px-2 py-0.5 rounded border transition-colors cursor-pointer ${
                  profileGridCols === 3
                    ? "bg-white text-[#0284C7] border-[#0284C7] font-semibold shadow-2xs"
                    : "bg-[#FAF9F5] text-[#73726C] border-[#E5E2D9] hover:text-[#1F1E1D]"
                }`}
                title="3 Columns Grid"
              >
                3-Col
              </button>
            </div>
          </div>
        </div>

        {/* Histograms Multi-Column Grid */}
        {numericColumnProfiles.length === 0 ? (
          <div className="p-8 text-center bg-[#FAF9F5] border border-dashed border-[#E5E2D9] rounded-xl text-xs text-[#73726C] space-y-1">
            <BarChart3 className="h-6 w-6 mx-auto opacity-40 text-[#0284C7]" />
            <p className="font-medium text-[#1F1E1D]">No Numerical Features Detected</p>
            <p>Deep profiling histograms require numeric continuous or integer columns.</p>
          </div>
        ) : filteredProfiles.length === 0 ? (
          <div className="p-8 text-center bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl text-xs text-[#73726C] space-y-1">
            <Search className="h-5 w-5 mx-auto opacity-40" />
            <p className="font-medium text-[#1F1E1D]">No numeric columns match &quot;{profileSearchQuery}&quot;</p>
            <button
              onClick={() => setProfileSearchQuery("")}
              className="text-[#0284C7] underline hover:text-[#0369A1] cursor-pointer"
            >
              Clear search filter
            </button>
          </div>
        ) : (
          <div className={`grid gap-4 ${profileGridCols === 3 ? "md:grid-cols-2 xl:grid-cols-3" : "md:grid-cols-2"}`}>
            {filteredProfiles.map((prof) => {
              const maxBinCount = Math.max(...prof.bins.map(b => b.count), 1);
              const peakBin = prof.bins.find(b => b.count === maxBinCount);

              return (
                <div 
                  key={prof.column.name}
                  className="bg-white border border-[#E5E2D9] rounded-xl p-4 flex flex-col justify-between hover:border-[#0284C7]/50 transition-all shadow-2xs group"
                >
                  {/* Card Header */}
                  <div>
                    <div className="flex items-start justify-between gap-2 border-b border-[#E5E2D9] pb-2.5">
                      <div>
                        <div className="flex items-center space-x-1.5 flex-wrap gap-y-1">
                          <span className="font-mono font-semibold text-xs text-[#1F1E1D]">
                            {prof.column.name}
                          </span>
                          {prof.column.sensitivity?.isSensitive && (
                            <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-rose-50 text-rose-700 border border-rose-200 flex items-center gap-0.5">
                              <ShieldAlert className="h-2.5 w-2.5" />
                              <span>{prof.column.sensitivity.category || "PII"}</span>
                            </span>
                          )}
                        </div>
                        <div className="text-[10px] text-[#73726C] font-mono mt-0.5">
                          Range: [{prof.min} → {prof.max}] · Valid: {prof.validCount.toLocaleString()}
                        </div>
                      </div>

                      <div className="flex items-center space-x-1">
                        {/* Skewness badge */}
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono border font-medium ${
                          prof.distributionType === "Normal"
                            ? "bg-emerald-50 text-emerald-800 border-emerald-200"
                            : prof.distributionType === "Right-Skewed"
                              ? "bg-amber-50 text-amber-800 border-amber-200"
                              : prof.distributionType === "Left-Skewed"
                                ? "bg-purple-50 text-purple-800 border-purple-200"
                                : "bg-gray-100 text-gray-700 border-gray-200"
                        }`}>
                          {prof.distributionType} ({prof.skewness > 0 ? `+${prof.skewness}` : prof.skewness})
                        </span>

                        {/* Expand Modal Button */}
                        <button
                          onClick={() => setInspectedColumnName(prof.column.name)}
                          className="p-1 rounded text-[#73726C] hover:text-[#0284C7] hover:bg-[#FAF9F5] cursor-pointer transition-colors"
                          title="Inspect Detailed Profile & Quantiles"
                        >
                          <Maximize2 className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    </div>

                    {/* Statistical Moments Bar */}
                    <div className="grid grid-cols-4 gap-1 py-2 text-center text-xs font-mono border-b border-[#E5E2D9]/70 bg-[#FAF9F5] -mx-4 px-4 my-2">
                      <div className="border-r border-[#E5E2D9] last:border-0">
                        <span className="text-[9px] text-[#73726C] block uppercase">Mean (μ)</span>
                        <span className="font-medium text-[#1F1E1D] text-[11px]">{prof.mean}</span>
                      </div>
                      <div className="border-r border-[#E5E2D9] last:border-0">
                        <span className="text-[9px] text-[#73726C] block uppercase">Median</span>
                        <span className="font-medium text-[#1F1E1D] text-[11px]">{prof.median}</span>
                      </div>
                      <div className="border-r border-[#E5E2D9] last:border-0">
                        <span className="text-[9px] text-[#73726C] block uppercase">Std Dev (σ)</span>
                        <span className="font-medium text-[#1F1E1D] text-[11px]">{prof.stdDev}</span>
                      </div>
                      <div>
                        <span className="text-[9px] text-[#73726C] block uppercase">IQR</span>
                        <span className="font-medium text-[#1F1E1D] text-[11px]">{prof.iqr}</span>
                      </div>
                    </div>

                    {/* Recharts Histogram Chart */}
                    <div className="h-44 w-full pt-1">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart 
                          data={prof.bins} 
                          margin={{ top: 8, right: 10, left: -22, bottom: 22 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E2D9" />
                          <XAxis 
                            dataKey="binLabel" 
                            stroke="#73726C" 
                            fontSize={9} 
                            tickLine={false}
                            interval={0}
                            angle={-25}
                            textAnchor="end"
                            height={28}
                          />
                          <YAxis 
                            stroke="#73726C" 
                            fontSize={9} 
                            tickLine={false}
                            allowDecimals={false}
                          />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: "#FFFFFF", 
                              borderColor: "#E5E2D9", 
                              borderRadius: "8px", 
                              fontSize: "11px",
                              boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
                            }}
                            formatter={(value: any, name: any, item: any) => [
                              `${Number(value).toLocaleString()} rows (${item?.payload?.percentage}%)`,
                              "Frequency"
                            ]}
                            labelFormatter={(label) => `Bin Range: [${label}]`}
                          />
                          <Bar 
                            dataKey="count" 
                            fill="#0284C7" 
                            radius={[3, 3, 0, 0]}
                            name="Frequency"
                          >
                            {prof.bins.map((b, bIdx) => (
                              <Cell 
                                key={`cell-${bIdx}`}
                                fill={b.count === maxBinCount ? "#0284C7" : "#38BDF8"}
                                opacity={0.9}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  {/* Card Bottom Meta */}
                  <div className="pt-2 border-t border-[#E5E2D9] flex items-center justify-between text-[11px] font-mono text-[#73726C]">
                    <span className="truncate max-w-[200px]">
                      Peak: [{peakBin?.binLabel || "-"}] ({peakBin?.percentage || 0}%)
                    </span>

                    <div className="flex items-center space-x-1.5">
                      {prof.outlierCount > 0 ? (
                        <span className="text-amber-700 bg-amber-50 px-1.5 py-0.2 rounded border border-amber-200">
                          {prof.outlierCount} outliers
                        </span>
                      ) : (
                        <span className="text-emerald-700">0 outliers</span>
                      )}

                      {prof.missingCount > 0 && (
                        <span className="text-rose-700 bg-rose-50 px-1.5 py-0.2 rounded border border-rose-200">
                          {prof.missingCount} null
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Detailed Histogram Inspection Modal */}
      {inspectedProfile && (
        <div className="fixed inset-0 z-50 bg-black/40 backdrop-blur-2xs flex items-center justify-center p-4">
          <div className="bg-white border border-[#E5E2D9] rounded-2xl max-w-3xl w-full p-6 space-y-4 shadow-xl max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-3">
              <div>
                <div className="flex items-center space-x-2">
                  <BarChart3 className="h-4 w-4 text-[#0284C7]" />
                  <h3 className="font-serif text-base font-medium text-[#1F1E1D]">
                    Detailed Statistical Profile: {inspectedProfile.column.name}
                  </h3>
                </div>
                <p className="text-xs text-[#73726C] mt-0.5">
                  Complete quantile fences, shape skewness, and bin frequency intervals
                </p>
              </div>

              <button
                onClick={() => setInspectedColumnName(null)}
                className="p-1 rounded-lg text-[#73726C] hover:text-[#1F1E1D] hover:bg-[#FAF9F5] cursor-pointer"
              >
                <X className="h-5 w-5" />
              </button>
            </div>

            {/* Large Modal Histogram */}
            <div className="h-64 w-full bg-[#FAF9F5] p-3 rounded-xl border border-[#E5E2D9]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={inspectedProfile.bins} margin={{ top: 10, right: 15, left: -15, bottom: 25 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E5E2D9" />
                  <XAxis 
                    dataKey="binLabel" 
                    stroke="#73726C" 
                    fontSize={10} 
                    tickLine={false} 
                    interval={0}
                    angle={-20}
                    textAnchor="end"
                    height={32}
                  />
                  <YAxis stroke="#73726C" fontSize={10} tickLine={false} allowDecimals={false} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: "#FFFFFF", 
                      borderColor: "#E5E2D9", 
                      borderRadius: "8px", 
                      fontSize: "12px" 
                    }}
                    formatter={(val: any, _, item: any) => [
                      `${Number(val).toLocaleString()} records (${item?.payload?.percentage}%)`,
                      "Frequency"
                    ]}
                    labelFormatter={(l) => `Interval: [${l}]`}
                  />
                  <Bar dataKey="count" fill="#0284C7" radius={[4, 4, 0, 0]} name="Frequency">
                    {inspectedProfile.bins.map((b, idx) => (
                      <Cell 
                        key={`modal-cell-${idx}`}
                        fill={b.count === Math.max(...inspectedProfile.bins.map(bin => bin.count)) ? "#0284C7" : "#38BDF8"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Quantiles & Dispersion Table */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
              <div className="p-3 bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl space-y-1">
                <span className="text-[10px] text-[#73726C] block">Minimum (P0)</span>
                <span className="font-semibold text-[#1F1E1D] text-sm">{inspectedProfile.min}</span>
                <span className="text-[9px] text-[#9E9D96] block">P10: {inspectedProfile.p10}</span>
              </div>
              <div className="p-3 bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl space-y-1">
                <span className="text-[10px] text-[#73726C] block">Lower Quartile (Q1 / P25)</span>
                <span className="font-semibold text-[#1F1E1D] text-sm">{inspectedProfile.q25}</span>
                <span className="text-[9px] text-[#9E9D96] block">Fence: {inspectedProfile.lowerBound}</span>
              </div>
              <div className="p-3 bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl space-y-1">
                <span className="text-[10px] text-[#73726C] block">Median (Q2 / P50)</span>
                <span className="font-semibold text-[#0284C7] text-sm">{inspectedProfile.median}</span>
                <span className="text-[9px] text-[#9E9D96] block">Mean: {inspectedProfile.mean}</span>
              </div>
              <div className="p-3 bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl space-y-1">
                <span className="text-[10px] text-[#73726C] block">Upper Quartile (Q3 / P75)</span>
                <span className="font-semibold text-[#1F1E1D] text-sm">{inspectedProfile.q75}</span>
                <span className="text-[9px] text-[#9E9D96] block">Fence: {inspectedProfile.upperBound}</span>
              </div>
            </div>

            {/* All Bins Breakdown Table */}
            <div className="space-y-2">
              <span className="text-xs font-medium text-[#1F1E1D]">Interval Bin Breakdown</span>
              <div className="max-h-48 overflow-y-auto border border-[#E5E2D9] rounded-xl">
                <table className="w-full text-left border-collapse text-xs font-mono">
                  <thead>
                    <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[#73726C] text-[10px]">
                      <th className="py-2 px-3">Bin Index</th>
                      <th className="py-2 px-3">Range Interval</th>
                      <th className="py-2 px-3">Frequency Count</th>
                      <th className="py-2 px-3 text-right">Dataset Share</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#E5E2D9]/70 text-[#1F1E1D]">
                    {inspectedProfile.bins.map((b) => (
                      <tr key={b.binIndex} className="hover:bg-[#FAF9F5]">
                        <td className="py-1.5 px-3 text-[#73726C]">#{b.binIndex + 1}</td>
                        <td className="py-1.5 px-3 font-medium">[{b.binLabel}]</td>
                        <td className="py-1.5 px-3">{b.count.toLocaleString()}</td>
                        <td className="py-1.5 px-3 text-right font-medium text-[#0284C7]">{b.percentage}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 2. Feature Distribution Analyzer */}
      <div className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-4 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
          <div>
            <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
              Variable Distribution Analyzer
            </h3>
            <p className="text-xs text-[#73726C]">
              Distribution shape, variance, and frequency breakdown for each individual column.
            </p>
          </div>
          
          <div className="flex items-center space-x-2">
            <span className="text-xs font-medium text-[#73726C]">Feature:</span>
            <select
              value={selectedColName}
              onChange={(e) => setSelectedColName(e.target.value)}
              className="bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] text-xs font-medium rounded-lg px-3 py-1.5 focus:border-[#0284C7] outline-none cursor-pointer"
            >
              {summary.columns.map((col, idx) => (
                <option key={idx} value={col.name}>
                  {col.name} ({col.type})
                </option>
              ))}
            </select>
          </div>
        </div>

        {selectedCol && (
          <div className="grid lg:grid-cols-3 gap-5">
            {/* Left: Column Metrics Grid */}
            <div className="space-y-3">
              <span className="text-[10px] font-medium uppercase tracking-wider text-[#9E9D96]">
                Statistical Moments
              </span>
              
              <div className="grid grid-cols-2 lg:grid-cols-1 gap-2 text-xs">
                <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-2.5 flex flex-col justify-between">
                  <span className="text-[10px] text-[#73726C]">Inferred Type</span>
                  <span className="font-mono font-medium text-[#1F1E1D] mt-0.5">{selectedCol.type}</span>
                </div>

                <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-2.5 flex flex-col justify-between">
                  <span className="text-[10px] text-[#73726C]">Unique Values</span>
                  <span className="font-mono font-medium text-[#1F1E1D] mt-0.5">
                    {selectedCol.stats.uniqueCount.toLocaleString()} ({selectedCol.stats.uniquePct ?? "N/A"}%)
                  </span>
                </div>

                {selectedCol.type === "Numeric" ? (
                  <>
                    <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-2.5 flex flex-col justify-between">
                      <span className="text-[10px] text-[#73726C]">Mean Average (μ)</span>
                      <span className="font-mono font-medium text-[#1F1E1D] mt-0.5">{selectedCol.stats.mean}</span>
                    </div>

                    <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-2.5 flex flex-col justify-between">
                      <span className="text-[10px] text-[#73726C]">Standard Deviation (σ)</span>
                      <span className="font-mono font-medium text-[#1F1E1D] mt-0.5">{selectedCol.stats.stdDev}</span>
                    </div>

                    <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-2.5 flex flex-col justify-between">
                      <span className="text-[10px] text-[#73726C]">Skewness & Kurtosis</span>
                      <span className="font-mono font-medium text-[#1F1E1D] mt-0.5">
                        skew: {selectedCol.stats.skewness} | kurt: {selectedCol.stats.kurtosis}
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-2.5 flex flex-col justify-between">
                    <span className="text-[10px] text-[#73726C]">Mode / Most Frequent</span>
                    <span className="font-mono font-medium text-[#1F1E1D] mt-0.5 truncate">
                      {categoricalData[0]?.name || "None"} ({categoricalData[0]?.value || 0} hits)
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Distribution Chart */}
            <div className="lg:col-span-2 space-y-3">
              <span className="text-[10px] font-medium uppercase tracking-wider text-[#9E9D96]">
                Density / Frequency Histogram
              </span>

              <div className="h-64 bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl p-3.5">
                {selectedCol.type === "Numeric" ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
                      <XAxis dataKey="binLabel" fontSize={9} stroke="#73726C" tickLine={false} />
                      <YAxis fontSize={9} stroke="#73726C" tickLine={false} />
                      <Tooltip 
                        contentStyle={{ 
                          background: "#FFFFFF", 
                          border: "1px solid #E5E2D9", 
                          borderRadius: "10px",
                          boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
                          color: "#1F1E1D",
                          fontSize: "11px"
                        }}
                      />
                      <Bar dataKey="count" fill="#0284C7" radius={[4, 4, 0, 0]}>
                        {distributionData.map((_, index) => (
                          <Cell key={`cell-${index}`} fill={index % 2 === 0 ? "#0284C7" : "#38BDF8"} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : categoricalData.length > 0 ? (
                  <div className="grid md:grid-cols-2 h-full items-center">
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={categoricalData.slice(0, 8)}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            innerRadius={45}
                            outerRadius={70}
                            paddingAngle={2}
                          >
                            {categoricalData.slice(0, 8).map((_, index) => (
                              <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip 
                            contentStyle={{ 
                              background: "#FFFFFF", 
                              border: "1px solid #E5E2D9", 
                              borderRadius: "10px",
                              color: "#1F1E1D",
                              fontSize: "11px"
                            }} 
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="space-y-1.5 max-h-52 overflow-y-auto pr-2 text-xs">
                      {categoricalData.slice(0, 6).map((item, idx) => {
                        const percent = ((item.value / (summary.rowCount - selectedCol.stats.missingCount)) * 100).toFixed(1);
                        return (
                          <div key={idx} className="flex items-center justify-between border-b border-[#E5E2D9] pb-1 last:border-0">
                            <span className="font-mono truncate max-w-[120px]" title={item.name}>{item.name}</span>
                            <span className="text-[#73726C] font-mono">{item.value} ({percent}%)</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-xs text-[#73726C]">
                    No frequency distribution available.
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 2. Scatter Plot & Trend Explorer */}
      {numericCols.length >= 2 && (
        <div className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-4 shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
            <div>
              <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
                Bivariate Scatter & Trend Explorer
              </h3>
              <p className="text-xs text-[#73726C]">
                Investigate associations between two numeric features with fitted regression and outlier isolation.
              </p>
            </div>

            {/* Variable Selectors */}
            <div className="flex items-center gap-2 flex-wrap text-xs">
              <div className="flex items-center space-x-1">
                <span className="text-[#73726C]">X:</span>
                <select
                  value={activeX}
                  onChange={(e) => setSelectedX(e.target.value)}
                  className="bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded-lg px-2.5 py-1 text-xs outline-none"
                >
                  {numericCols.map((c, idx) => (
                    <option key={idx} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              <div className="flex items-center space-x-1">
                <span className="text-[#73726C]">Y:</span>
                <select
                  value={activeY}
                  onChange={(e) => setSelectedY(e.target.value)}
                  className="bg-[#FAF9F5] border border-[#E5E2D9] text-[#1F1E1D] rounded-lg px-2.5 py-1 text-xs outline-none"
                >
                  {numericCols.map((c, idx) => (
                    <option key={idx} value={c}>{c}</option>
                  ))}
                </select>
              </div>

              {/* Outlier view mode */}
              <div className="flex items-center space-x-1">
                <select
                  value={outlierViewMode}
                  onChange={(e: any) => setOutlierViewMode(e.target.value)}
                  className="bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C] rounded-lg px-2 py-1 text-xs outline-none"
                >
                  <option value="all">All Points (Highlight Outliers)</option>
                  <option value="clean-active">Filter Active Outliers</option>
                  <option value="clean-all">Filter Any Outlier</option>
                </select>
              </div>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-5 items-start">
            
            {/* Scatter Canvas */}
            <div className="lg:col-span-2 bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl p-4 h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart margin={{ top: 10, right: 10, bottom: 10, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5E2D9" className="" />
                  <XAxis 
                    type="number" 
                    dataKey="x" 
                    name={activeX} 
                    fontSize={9} 
                    stroke="#73726C" 
                    tickLine={false} 
                  />
                  <YAxis 
                    type="number" 
                    dataKey="y" 
                    name={activeY} 
                    fontSize={9} 
                    stroke="#73726C" 
                    tickLine={false} 
                  />
                  <Tooltip 
                    cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{ 
                      background: "#FFFFFF", 
                      border: "1px solid #E5E2D9", 
                      borderRadius: "10px",
                      color: "#1F1E1D",
                      fontSize: "11px"
                    }} 
                  />
                  <Scatter 
                    name="Observations" 
                    data={scatterData} 
                    onClick={handlePointClick}
                    onMouseEnter={handlePointHover}
                    onMouseLeave={handlePointLeave}
                  >
                    {scatterData.map((pt, idx) => (
                      <Cell 
                        key={`pt-${idx}`} 
                        fill={pt.isOutlier && outlierViewMode === "all" ? "#38BDF8" : "#0284C7"} 
                        opacity={0.85}
                      />
                    ))}
                  </Scatter>
                  {trendData.length > 0 && (
                    <Line 
                      name="Fitted Trend" 
                      data={trendData} 
                      dataKey="y" 
                      stroke="#1F1E1D" 
                      dot={false} 
                      strokeWidth={2} 
                      activeDot={false}
                      className=""
                      onMouseEnter={handleTrendLineHover}
                      onMouseLeave={handlePointLeave}
                    />
                  )}
                </ComposedChart>
              </ResponsiveContainer>
            </div>

            {/* AI Segment Inspector Card (Claude Prompt / Insight style) */}
            <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-xl p-4 space-y-3 text-xs">
              <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2">
                <div className="flex items-center space-x-1.5">
                  <Activity className="h-3.5 w-3.5 text-[#0284C7]" />
                  <span className="font-serif font-medium text-[#1F1E1D]">
                    Coordinate Inspector
                  </span>
                </div>
                {activeExplorerPoint && (
                  <button
                    onClick={handleTogglePin}
                    className="text-[10px] text-[#73726C] hover:text-[#1F1E1D] flex items-center gap-1 cursor-pointer"
                  >
                    <Pin className={`h-2.5 w-2.5 ${pinnedPoint ? "fill-current text-[#0284C7]" : ""}`} />
                    <span>{pinnedPoint ? "Pinned" : "Pin"}</span>
                  </button>
                )}
              </div>

              {activeExplorerPoint ? (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2 rounded-lg bg-white border border-[#E5E2D9]">
                      <span className="text-[10px] text-[#73726C] block">{activeX}</span>
                      <span className="font-medium text-[#1F1E1D]">{activeExplorerPoint.x.toFixed(3)}</span>
                    </div>
                    <div className="p-2 rounded-lg bg-white border border-[#E5E2D9]">
                      <span className="text-[10px] text-[#73726C] block">{activeY}</span>
                      <span className="font-medium text-[#1F1E1D]">{activeExplorerPoint.y.toFixed(3)}</span>
                    </div>
                  </div>

                  {regression && (
                    <div className="text-[11px] text-[#73726C] font-mono space-y-0.5">
                      <p>Slope: {regression.slope.toFixed(3)} · Pearson r: {regression.correlation.toFixed(3)}</p>
                    </div>
                  )}

                  {/* AI Explanation Box */}
                  <div className="p-3 bg-white border border-[#E5E2D9] rounded-xl text-xs space-y-1 shadow-xs">
                    {isLoadingExpl ? (
                      <div className="flex items-center space-x-2 py-2 text-[#73726C]">
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-[#0284C7]" />
                        <span className="text-[11px]">Explaining segment...</span>
                      </div>
                    ) : explanation ? (
                      <div className="space-y-1">
                        <div className="flex items-center space-x-1 text-[#0284C7]">
                          <Sparkles className="h-3 w-3" />
                          <span className="text-[10px] font-medium uppercase tracking-wider">Analysis</span>
                        </div>
                        <p className="text-xs text-[#1F1E1D] leading-relaxed">
                          {explanation}
                        </p>
                      </div>
                    ) : (
                      <p className="text-[11px] text-[#73726C] italic">
                        Hover or click any scatter point to generate analytical context.
                      </p>
                    )}
                  </div>
                </div>
              ) : (
                <div className="py-6 text-center text-xs text-[#73726C] space-y-1">
                  <HelpCircle className="h-4 w-4 mx-auto opacity-50" />
                  <p>Hover or click any point in the scatter graph</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 3. Pearson Correlation Heatmap */}
      {numericCols.length >= 2 && (
        <div className="bg-white border border-[#E5E2D9] rounded-xl p-5 space-y-4 shadow-xs">
          <div className="border-b border-[#E5E2D9] pb-2.5">
            <h3 className="font-serif text-sm font-medium text-[#1F1E1D]">
              Pearson Correlation Matrix Heatmap
            </h3>
            <p className="text-xs text-[#73726C]">
              Normalized linear correlation coefficients (-1.0 to +1.0) between numerical columns.
            </p>
          </div>

          <div className="overflow-x-auto pt-1">
            <table className="border-collapse w-full text-center font-mono text-xs">
              <thead>
                <tr>
                  <th className="p-2 font-medium border border-[#E5E2D9] text-left text-[#73726C] bg-[#FAF9F5] min-w-[120px]">
                    Feature
                  </th>
                  {numericCols.map((cName, idx) => (
                    <th key={idx} className="p-2 font-medium border border-[#E5E2D9] text-[#73726C] bg-[#FAF9F5] max-w-[110px] truncate" title={cName}>
                      {cName}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {numericCols.map((cRowName, rowIdx) => (
                  <tr key={rowIdx}>
                    <td className="p-2 font-medium border border-[#E5E2D9] text-left text-[#1F1E1D] bg-[#FAF9F5] truncate max-w-[120px]" title={cRowName}>
                      {cRowName}
                    </td>
                    {numericCols.map((cColName, colIdx) => {
                      const val = summary.correlation[cRowName]?.[cColName] ?? 0;
                      return (
                        <td
                          key={colIdx}
                          className={`p-2 border border-[#E5E2D9] text-xs transition-colors duration-150 ${getHeatmapBg(val)}`}
                          title={`${cRowName} ↔ ${cColName}: ${val.toFixed(4)}`}
                        >
                          {val.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
