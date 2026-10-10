import React, { useState, useMemo, useEffect } from "react";
import { 
  ResponsiveContainer, BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ReferenceLine, CartesianGrid 
} from "recharts";
import { 
  Scale, Info, HelpCircle, BarChart3, ArrowUpRight, TrendingUp, Sparkles, CheckCircle2 
} from "lucide-react";

interface FeatureImportanceItem {
  feature: string;
  importance: number;
  weight?: number;
  gain?: number;
}

interface FeatureImportancePanelProps {
  featureImportances: FeatureImportanceItem[];
  modelId: string;
  taskType: "Classification" | "Regression";
}

export default function FeatureImportancePanel({ 
  featureImportances, 
  modelId, 
  taskType 
}: FeatureImportancePanelProps) {
  const isTree = modelId === "decision_tree";

  // Allowed modes: "importance" | "weight" (for linear/logistic) | "gain" (for tree)
  const [metricType, setMetricType] = useState<"importance" | "weight" | "gain">("importance");
  const [limit, setLimit] = useState<number>(10);

  // Fallback / sync metric type when model changes
  useEffect(() => {
    if (isTree) {
      setMetricType("importance");
    } else {
      setMetricType("importance");
    }
  }, [modelId, isTree]);

  // Format and sort data based on selected metric
  const chartData = useMemo(() => {
    if (!featureImportances || featureImportances.length === 0) return [];

    let processed = [...featureImportances];

    // Map feature display names
    processed = processed.map(item => ({
      ...item,
      // Humanize weight label prefix
      feature: item.feature.replace(/^Weight_/, "Weight: ").replace(/^Feat_/, "Feature: ")
    }));

    if (metricType === "weight") {
      // Sort by absolute weight descending for readability, showing both signs
      processed.sort((a, b) => Math.abs(b.weight || 0) - Math.abs(a.weight || 0));
    } else if (metricType === "gain") {
      processed.sort((a, b) => (b.gain || 0) - (a.gain || 0));
    } else {
      processed.sort((a, b) => b.importance - a.importance);
    }

    return processed.slice(0, limit);
  }, [featureImportances, metricType, limit]);

  // Analytical insights generated from the data
  const insights = useMemo(() => {
    if (!featureImportances || featureImportances.length === 0) return null;

    const sortedByImp = [...featureImportances].sort((a, b) => b.importance - a.importance);
    const topFeature = sortedByImp[0];

    // Compute coefficients info if linear/logistic
    let positiveFeatures: string[] = [];
    let negativeFeatures: string[] = [];
    if (!isTree) {
      featureImportances.forEach(item => {
        if (item.weight && item.weight > 0) {
          positiveFeatures.push(item.feature);
        } else if (item.weight && item.weight < 0) {
          negativeFeatures.push(item.feature);
        }
      });
    }

    return {
      topFeature: {
        name: topFeature.feature.replace(/^Weight_/, "").replace(/^Feat_/, "Feature "),
        pct: (topFeature.importance * 100).toFixed(1)
      },
      positiveCount: positiveFeatures.length,
      negativeCount: negativeFeatures.length,
      topThree: sortedByImp.slice(0, 3).map(item => ({
        name: item.feature.replace(/^Weight_/, "").replace(/^Feat_/, "Feature "),
        pct: (item.importance * 100).toFixed(0),
        direction: item.weight !== undefined 
          ? (item.weight > 0 ? "positive" : item.weight < 0 ? "negative" : "neutral")
          : null,
        value: item.weight !== undefined 
          ? item.weight 
          : (item.gain !== undefined ? item.gain : null)
      }))
    };
  }, [featureImportances, isTree]);

  if (!featureImportances || featureImportances.length === 0) {
    return null;
  }

  return (
    <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded-lg p-5 space-y-6">
      
      {/* Header and Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E5E2D9] pb-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-sky-400" />
            <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
              Feature Importance & Model Interpretability
            </h3>
          </div>
          <p className="text-[11px] text-[#9E9D96] font-mono">
            Understand which independent variables drive your model's predictions.
          </p>
        </div>

        {/* Toggles */}
        <div className="flex flex-wrap items-center gap-2.5 font-mono text-[10px]">
          {/* Metric Selector */}
          <div className="flex items-center gap-1.5 bg-[#FAF9F5] border border-[#E5E2D9] p-1 rounded">
            <button
              onClick={() => setMetricType("importance")}
              className={`px-2.5 py-1 rounded font-bold uppercase tracking-wider transition-all cursor-pointer ${
                metricType === "importance"
                  ? "bg-sky-500/10 text-sky-400"
                  : "text-[#9E9D96] hover:text-[#73726C]"
              }`}
            >
              Importance
            </button>

            {!isTree && (
              <button
                onClick={() => setMetricType("weight")}
                className={`px-2.5 py-1 rounded font-bold uppercase tracking-wider transition-all cursor-pointer ${
                  metricType === "weight"
                    ? "bg-sky-500/10 text-sky-400"
                    : "text-[#9E9D96] hover:text-[#73726C]"
              }`}
              >
                Coefficients
              </button>
            )}

            {isTree && (
              <button
                onClick={() => setMetricType("gain")}
                className={`px-2.5 py-1 rounded font-bold uppercase tracking-wider transition-all cursor-pointer ${
                  metricType === "gain"
                    ? "bg-sky-500/10 text-sky-400"
                    : "text-[#9E9D96] hover:text-[#73726C]"
              }`}
              >
                Split Gain
              </button>
            )}
          </div>

          {/* Limit Selector */}
          <div className="flex items-center gap-1.5 bg-[#FAF9F5] border border-[#E5E2D9] px-2 py-1.5 rounded">
            <span className="text-[#9E9D96] font-bold uppercase">Show Top:</span>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="bg-transparent text-[#73726C] font-bold outline-none cursor-pointer text-[10px] uppercase font-mono"
            >
              <option value={5} className="bg-[#FAF9F5] text-[#73726C]">5 Features</option>
              <option value={10} className="bg-[#FAF9F5] text-[#73726C]">10 Features</option>
              <option value={20} className="bg-[#FAF9F5] text-[#73726C]">20 Features</option>
              <option value={100} className="bg-[#FAF9F5] text-[#73726C]">All Features</option>
            </select>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="grid lg:grid-cols-3 gap-6">
        
        {/* Chart Viewport */}
        <div className="lg:col-span-2 space-y-3">
          <div className="bg-white border border-[#E5E2D9] rounded-lg p-4 h-80 relative flex flex-col justify-between">
            <div className="absolute top-2 right-3 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-sky-500 animate-pulse" />
              <span className="text-[8px] font-mono font-bold text-[#9E9D96] uppercase tracking-widest">
                Active interpretability plot
              </span>
            </div>

            <div className="w-full h-full pt-4">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 10, right: 30, left: 15, bottom: 5 }}
                >
                  <CartesianGrid stroke="#1c1c1c" strokeDasharray="3 3" horizontal={true} vertical={true} />
                  
                  <XAxis 
                    type="number" 
                    fontSize={8} 
                    stroke="#4b5563"
                    tickFormatter={(val) => Number(val).toFixed(2)}
                    domain={metricType === "weight" ? ["auto", "auto"] : [0, "auto"]}
                  />
                  
                  <YAxis 
                    type="category" 
                    dataKey="feature" 
                    fontSize={8} 
                    stroke="#9ca3af" 
                    tickLine={false} 
                    width={110}
                    style={{ fontFamily: "monospace", fontWeight: "bold" }}
                  />
                  
                  <Tooltip
                    cursor={{ fill: "rgba(255,255,255,0.02)" }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const data = payload[0].payload as FeatureImportanceItem;
                        return (
                          <div className="bg-[#FAF9F5] border border-[#E5E2D9] p-2.5 rounded shadow-xl font-mono text-[10px] space-y-1.5 max-w-xs">
                            <p className="font-bold text-[#1F1E1D] uppercase border-b border-[#E5E2D9] pb-1">
                              {data.feature}
                            </p>
                            <p className="text-[#73726C]">
                              Relative Importance: <strong className="text-sky-400">{(data.importance * 100).toFixed(1)}%</strong>
                            </p>
                            
                            {metricType === "weight" && data.weight !== undefined && (
                              <p className="text-[#73726C]">
                                Coefficient Weight: <strong className={data.weight >= 0 ? "text-sky-400" : "text-rose-400"}>
                                  {data.weight > 0 ? "+" : ""}{data.weight.toFixed(4)}
                                </strong>
                              </p>
                            )}

                            {metricType === "gain" && data.gain !== undefined && (
                              <p className="text-[#73726C]">
                                Accumulated Gain: <strong className="text-blue-400">{data.gain.toFixed(5)}</strong>
                              </p>
                            )}

                            {/* Contextual description inside tooltip */}
                            <p className="text-[8px] text-[#9E9D96] leading-normal pt-1 italic">
                              {metricType === "weight" && data.weight !== undefined ? (
                                data.weight > 0 
                                  ? "Positive weight: higher values drive prediction values UP."
                                  : "Negative weight: higher values pull prediction values DOWN."
                              ) : "Higher relative score means the feature is more vital for fitting decision splits."}
                            </p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />

                  {metricType === "weight" && (
                    <ReferenceLine x={0} stroke="#444" strokeWidth={1} strokeDasharray="3 3" />
                  )}

                  <Bar 
                    dataKey={metricType === "weight" ? "weight" : (metricType === "gain" ? "gain" : "importance")}
                    radius={[0, 3, 3, 0]}
                  >
                    {chartData.map((entry, index) => {
                      // Positive vs. negative bar colors for weights
                      if (metricType === "weight" && entry.weight !== undefined) {
                        return (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.weight >= 0 ? "#10b981" : "#ef4444"} 
                            fillOpacity={0.8}
                          />
                        );
                      }
                      // Gradient colors for importance and gains
                      return (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={index === 0 ? "#10b981" : "#3b82f6"} 
                          fillOpacity={0.85 - (index * 0.03)}
                        />
                      );
                    })}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Statistical Interpretations and Explainer Pane */}
        <div className="bg-white border border-[#E5E2D9] rounded-lg p-4 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex items-center gap-1.5 border-b border-[#E5E2D9] pb-2">
              <Sparkles className="h-3.5 w-3.5 text-sky-400" />
              <span className="text-[10px] font-mono font-bold text-[#73726C] uppercase tracking-wider">
                Platform Insights
              </span>
            </div>

            {/* Explainer Box */}
            <div className="text-[10px] font-mono text-[#73726C] leading-relaxed space-y-2.5">
              <div className="flex items-start gap-1.5 p-2 bg-white rounded border border-[#E5E2D9]">
                <Info className="h-3.5 w-3.5 text-[#9E9D96] flex-shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-[#73726C] uppercase text-[9px] mb-0.5">
                    What are you looking at?
                  </h4>
                  <p className="text-[9px] leading-relaxed">
                    {metricType === "weight" ? (
                      "Standardized Coefficients show the direction and magnitude of feature relationships. Values farther from zero have a stronger impact."
                    ) : metricType === "gain" ? (
                      "Split Gain calculates the total reduction in impurity (Gini or Mean Squared Error) contributed by each feature across all tree splits."
                    ) : (
                      "Normalized Relative Importance scales each feature's contribution compared to the top predictor, which is pinned at 100%."
                    )}
                  </p>
                </div>
              </div>

              {/* Dynamic Summary Stats */}
              {insights && (
                <div className="space-y-3 pt-1">
                  <div>
                    <span className="text-[#9E9D96] font-bold uppercase text-[9px]">Top Predictor Feature:</span>
                    <p className="text-[#1F1E1D] font-bold text-xs mt-0.5 flex items-center gap-1">
                      <ArrowUpRight className="h-3.5 w-3.5 text-sky-400 flex-shrink-0" />
                      {insights.topFeature.name} 
                      <span className="text-[10px] font-mono text-sky-400 font-black">
                        ({insights.topFeature.pct}% weight)
                      </span>
                    </p>
                  </div>

                  {!isTree && (
                    <div className="grid grid-cols-2 gap-2 border-t border-[#E5E2D9]/60 pt-2">
                      <div>
                        <span className="text-[#9E9D96] font-bold uppercase text-[8px]">Positive Drivers:</span>
                        <p className="text-sky-400 font-black text-[11px] mt-0.5">
                          {insights.positiveCount} columns
                        </p>
                      </div>
                      <div>
                        <span className="text-[#9E9D96] font-bold uppercase text-[8px]">Negative Drivers:</span>
                        <p className="text-rose-400 font-black text-[11px] mt-0.5">
                          {insights.negativeCount} columns
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Top 3 List */}
                  <div className="space-y-1.5 border-t border-[#E5E2D9]/60 pt-2">
                    <span className="text-[#9E9D96] font-bold uppercase text-[9px]">Ranked Impact Hierarchy:</span>
                    <ul className="space-y-1 text-[9px]">
                      {insights.topThree.map((item, i) => (
                        <li key={i} className="flex items-center justify-between text-[#73726C]">
                          <span className="truncate max-w-[150px]">
                            {i + 1}. <strong className="text-[#1F1E1D]">{item.name}</strong>
                          </span>
                          <span className="font-bold font-mono text-sky-400">
                            {item.direction === "positive" && "Positive impact"}
                            {item.direction === "negative" && "Negative impact"}
                            {item.direction === null && `Gain: ${Number(item.value).toFixed(4)}`}
                          </span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="p-2 bg-sky-500/5 text-sky-400 rounded border border-sky-500/10 text-[9px] font-mono leading-relaxed flex items-start gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-sky-400 flex-shrink-0 mt-0.5" />
            <p>
              These results are computed strictly inside this secure local sandbox. Standardization prevents scale biases across units.
            </p>
          </div>
        </div>

      </div>
    </div>
  );
}
