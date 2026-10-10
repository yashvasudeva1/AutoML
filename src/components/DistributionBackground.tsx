import React, { useState, useMemo } from "react";
import { Sliders, Settings, Info, Activity } from "lucide-react";

interface DistributionBackgroundProps {
  interactive?: boolean;
}

export default function DistributionBackground({ interactive = true }: DistributionBackgroundProps) {
  // Selected distribution type to display in the main panel
  const [activeDist, setActiveDist] = useState<"normal" | "exponential" | "bimodal" | "uniform" | "binomial">("normal");

  // Normal distribution parameters
  const [normalMean, setNormalMean] = useState(200);
  const [normalStd, setNormalStd] = useState(40);

  // Exponential/Poisson distribution parameters
  const [lambda, setLambda] = useState(0.015);

  // Bimodal distribution parameters
  const [peak1Mean, setPeak1Mean] = useState(130);
  const [peak2Mean, setPeak2Mean] = useState(270);
  const [bimodalStd, setBimodalStd] = useState(30);

  // Uniform distribution parameters
  const [uniformStart, setUniformStart] = useState(100);
  const [uniformEnd, setUniformEnd] = useState(300);

  // Binomial trials
  const [binomialN, setBinomialN] = useState(15);
  const [binomialP, setBinomialP] = useState(0.5);

  // Generate path coordinates for Normal (Gaussian) Distribution
  const normalPath = useMemo(() => {
    const points: string[] = [];
    const width = 400;
    const height = 180;
    const mu = normalMean;
    const sigma = normalStd;
    
    for (let x = 0; x <= width; x += 2) {
      // Gaussian formula: height * exp(-0.5 * ((x - mu)/sigma)^2)
      const exponent = -0.5 * Math.pow((x - mu) / sigma, 2);
      const y = height - 25 - (130 * Math.exp(exponent));
      points.push(`${x},${Math.max(10, Math.min(height - 10, y))}`);
    }
    return `M ${points.join(" L ")}`;
  }, [normalMean, normalStd]);

  // Generate path coordinates for Exponential/Poisson decay
  const exponentialPath = useMemo(() => {
    const points: string[] = [];
    const width = 400;
    const height = 180;
    
    for (let x = 0; x <= width; x += 2) {
      // Exponential decay: height - offset - amplitude * exp(-lambda * x)
      const y = height - 25 - (140 * Math.exp(-lambda * x));
      points.push(`${x},${Math.max(10, Math.min(height - 10, y))}`);
    }
    return `M ${points.join(" L ")}`;
  }, [lambda]);

  // Generate path coordinates for Bimodal Distribution
  const bimodalPath = useMemo(() => {
    const points: string[] = [];
    const width = 400;
    const height = 180;
    
    for (let x = 0; x <= width; x += 2) {
      // Combination of two normal peaks
      const exponent1 = -0.5 * Math.pow((x - peak1Mean) / bimodalStd, 2);
      const exponent2 = -0.5 * Math.pow((x - peak2Mean) / bimodalStd, 2);
      
      const peak1 = 90 * Math.exp(exponent1);
      const peak2 = 90 * Math.exp(exponent2);
      const y = height - 25 - (peak1 + peak2);
      
      points.push(`${x},${Math.max(10, Math.min(height - 10, y))}`);
    }
    return `M ${points.join(" L ")}`;
  }, [peak1Mean, peak2Mean, bimodalStd]);

  // Uniform Distribution Box Coordinates
  const uniformCoords = useMemo(() => {
    const height = 180;
    const yTop = height - 120;
    const yBottom = height - 25;
    
    return {
      x1: uniformStart,
      x2: uniformEnd,
      yTop,
      yBottom
    };
  }, [uniformStart, uniformEnd]);

  // Binomial PMF Bars
  const binomialBars = useMemo(() => {
    const bars: Array<{ x: number; height: number; value: number }> = [];
    const width = 400;
    const svgHeight = 180;
    const n = binomialN;
    const p = binomialP;

    // Helper for combinations (n choose k)
    const choose = (n: number, k: number): number => {
      if (k < 0 || k > n) return 0;
      let res = 1;
      for (let i = 1; i <= k; i++) {
        res = res * (n - i + 1) / i;
      }
      return res;
    };

    const startX = 50;
    const endX = 350;
    const stepX = (endX - startX) / n;

    for (let k = 0; k <= n; k++) {
      // Binomial probability: P(X=k) = (n choose k) * p^k * (1-p)^(n-k)
      const prob = choose(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k);
      const barHeight = Math.max(1, prob * 350); // Scaled for display
      const x = startX + k * stepX;
      bars.push({
        x,
        height: barHeight,
        value: prob
      });
    }

    return bars;
  }, [binomialN, binomialP]);

  return (
    <div className="w-full flex flex-col lg:flex-row gap-6 relative z-10">
      {/* Visual Sandbox Viewport */}
      <div className="flex-1 bg-white border border-[#E5E2D9] rounded p-4 relative flex flex-col justify-between min-h-[300px]">
        
        {/* Background blueprint grids */}
        <div className="absolute inset-0 grid grid-cols-10 grid-rows-6 pointer-events-none opacity-[0.03]">
          {Array.from({ length: 60 }).map((_, i) => (
            <div key={i} className="border-t border-l border-neutral-500" />
          ))}
        </div>

        {/* Dynamic header overlay */}
        <div className="flex items-center justify-between z-10 border-b border-[#E5E2D9] pb-2">
          <div className="space-y-0.5">
            <span className="text-[8px] font-mono font-bold uppercase tracking-wider text-sky-500">Live Mathematical Sandbox</span>
            <h4 className="text-xs font-mono font-bold text-[#73726C] uppercase tracking-wide">
              {activeDist === "normal" && "Normal (Gaussian) Distribution"}
              {activeDist === "exponential" && "Exponential Decay Distribution"}
              {activeDist === "bimodal" && "Bimodal (Double Peak) Distribution"}
              {activeDist === "uniform" && "Continuous Uniform Distribution"}
              {activeDist === "binomial" && "Discrete Binomial Distribution"}
            </h4>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-sky-500 animate-ping" />
            <span className="text-[8px] font-mono text-[#9E9D96] uppercase tracking-widest font-black">Active Render</span>
          </div>
        </div>

        {/* SVG Drawing Canvas Container */}
        <div className="my-4 h-48 flex items-center justify-center relative">
          <svg className="w-full h-full overflow-visible" viewBox="0 0 400 180" preserveAspectRatio="none">
            {/* Draw active distribution profile */}
            {activeDist === "normal" && (
              <>
                {/* Mean Line (mu) */}
                <line 
                  x1={normalMean} 
                  y1={10} 
                  x2={normalMean} 
                  y2={155} 
                  stroke="#10b981" 
                  strokeWidth={1} 
                  strokeDasharray="4 4" 
                  opacity={0.4} 
                />
                {/* Standard dev margins (+- sigma) */}
                <line 
                  x1={normalMean - normalStd} 
                  y1={40} 
                  x2={normalMean - normalStd} 
                  y2={155} 
                  stroke="#3b82f6" 
                  strokeWidth={1} 
                  strokeDasharray="2 3" 
                  opacity={0.3} 
                />
                <line 
                  x1={normalMean + normalStd} 
                  y1={40} 
                  x2={normalMean + normalStd} 
                  y2={155} 
                  stroke="#3b82f6" 
                  strokeWidth={1} 
                  strokeDasharray="2 3" 
                  opacity={0.3} 
                />

                {/* Shaded Area under peak */}
                <path 
                  d={`${normalPath} L 400,155 L 0,155 Z`} 
                  fill="url(#normalGradient)" 
                  opacity={0.06} 
                />

                {/* Gaussian Curve Outline */}
                <path 
                  d={normalPath} 
                  fill="none" 
                  stroke="#10b981" 
                  strokeWidth={2} 
                  strokeLinecap="round" 
                />

                {/* Markers */}
                <circle cx={normalMean} cy={180 - 25 - 130} r={4} fill="#10b981" stroke="#34d399" strokeWidth={1} />
              </>
            )}

            {activeDist === "exponential" && (
              <>
                <path 
                  d={`${exponentialPath} L 400,155 L 0,155 Z`} 
                  fill="url(#expGradient)" 
                  opacity={0.06} 
                />
                <path 
                  d={exponentialPath} 
                  fill="none" 
                  stroke="#34d399" 
                  strokeWidth={2} 
                  strokeLinecap="round" 
                />
              </>
            )}

            {activeDist === "bimodal" && (
              <>
                <line 
                  x1={peak1Mean} 
                  y1={10} 
                  x2={peak1Mean} 
                  y2={155} 
                  stroke="#3b82f6" 
                  strokeWidth={1} 
                  strokeDasharray="3 3" 
                  opacity={0.3} 
                />
                <line 
                  x1={peak2Mean} 
                  y1={10} 
                  x2={peak2Mean} 
                  y2={155} 
                  stroke="#3b82f6" 
                  strokeWidth={1} 
                  strokeDasharray="3 3" 
                  opacity={0.3} 
                />

                <path 
                  d={`${bimodalPath} L 400,155 L 0,155 Z`} 
                  fill="url(#bimodalGradient)" 
                  opacity={0.05} 
                />
                <path 
                  d={bimodalPath} 
                  fill="none" 
                  stroke="#60a5fa" 
                  strokeWidth={2} 
                  strokeLinecap="round" 
                />
              </>
            )}

            {activeDist === "uniform" && (
              <>
                {/* Uniform distribution block */}
                <rect 
                  x={uniformCoords.x1} 
                  y={uniformCoords.yTop} 
                  width={uniformCoords.x2 - uniformCoords.x1} 
                  height={uniformCoords.yBottom - uniformCoords.yTop} 
                  fill="url(#uniformGradient)" 
                  stroke="#34d399" 
                  strokeWidth={2} 
                  rx={2}
                />
                {/* Dashed outer wings */}
                <line x1={0} y1={uniformCoords.yBottom} x2={uniformCoords.x1} y2={uniformCoords.yBottom} stroke="#525252" strokeWidth={1.5} strokeDasharray="3 3" />
                <line x1={uniformCoords.x2} y1={uniformCoords.yBottom} x2={400} y2={uniformCoords.yBottom} stroke="#525252" strokeWidth={1.5} strokeDasharray="3 3" />
              </>
            )}

            {activeDist === "binomial" && (
              <>
                {binomialBars.map((b, i) => (
                  <g key={i}>
                    {/* Probability Bar */}
                    <rect 
                      x={b.x - 4} 
                      y={155 - b.height} 
                      width={8} 
                      height={b.height} 
                      fill={b.height > 20 ? "#10b981" : "#4b5563"} 
                      opacity={b.height > 20 ? 0.8 : 0.4}
                      rx={1}
                    />
                    {/* Tip Circle */}
                    <circle 
                      cx={b.x} 
                      cy={155 - b.height} 
                      r={2} 
                      fill={b.height > 20 ? "#34d399" : "#6b7280"} 
                    />
                  </g>
                ))}
              </>
            )}

            {/* Base Coordinate Axis line */}
            <line x1={0} y1={155} x2={400} y2={155} stroke="#262626" strokeWidth={1.5} />

            {/* Definitions / Gradients */}
            <defs>
              <linearGradient id="normalGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="expGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#34d399" />
                <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="bimodalGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3b82f6" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="uniformGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.15" />
                <stop offset="100%" stopColor="#10b981" stopOpacity="0" />
              </linearGradient>
            </defs>
          </svg>
        </div>

        {/* Stats Summary Panel */}
        <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-2.5 flex items-center justify-between text-[10px] font-mono">
          <div className="text-[#9E9D96]">
            {activeDist === "normal" && (
              <span>Mean: <strong className="text-sky-400">{normalMean}</strong> | Std Dev: <strong className="text-[#73726C]">{normalStd}</strong></span>
            )}
            {activeDist === "exponential" && (
              <span>Decay rate (&lambda;): <strong className="text-sky-400">{lambda}</strong> | Half-Life: <strong className="text-[#73726C]">{(0.693 / lambda).toFixed(1)}s</strong></span>
            )}
            {activeDist === "bimodal" && (
              <span>Peak 1: <strong className="text-sky-400">{peak1Mean}</strong> | Peak 2: <strong className="text-sky-400">{peak2Mean}</strong></span>
            )}
            {activeDist === "uniform" && (
              <span>Interval: <strong className="text-sky-400">[{uniformStart}, {uniformEnd}]</strong> | Width: <strong className="text-[#73726C]">{uniformEnd - uniformStart}</strong></span>
            )}
            {activeDist === "binomial" && (
              <span>Trials (n): <strong className="text-sky-400">{binomialN}</strong> | Probability (p): <strong className="text-[#73726C]">{binomialP}</strong></span>
            )}
          </div>

          <span className="text-[8px] text-[#73726C] uppercase tracking-widest font-bold">Dynamic Probability Plot</span>
        </div>
      </div>

      {/* Control sliders deck */}
      {interactive && (
        <div className="w-full lg:w-72 bg-white border border-[#E5E2D9] rounded p-4 flex flex-col justify-between space-y-4">
          <div className="space-y-4">
            <div className="flex items-center gap-2 border-b border-[#E5E2D9] pb-2">
              <Sliders className="h-3.5 w-3.5 text-sky-400" />
              <span className="text-[10px] font-mono font-bold text-[#73726C] uppercase tracking-wider">Distribution Selector</span>
            </div>

            {/* Selector Grid Buttons */}
            <div className="grid grid-cols-2 gap-1.5">
              {(["normal", "exponential", "bimodal", "uniform", "binomial"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setActiveDist(t)}
                  className={`px-2 py-1.5 rounded text-[9px] font-mono font-bold uppercase transition-all duration-150 text-left truncate cursor-pointer border ${
                    activeDist === t
                      ? "bg-sky-500/10 text-sky-400 border-sky-500/20"
                      : "bg-transparent text-[#9E9D96] border-transparent hover:text-[#73726C]"
                  }`}
                >
                  • {t}
                </button>
              ))}
            </div>

            {/* Render sliders depending on choice */}
            <div className="space-y-3 pt-2">
              {activeDist === "normal" && (
                <>
                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Mean (&mu;):</span>
                      <span className="text-sky-400 font-bold">{normalMean}</span>
                    </div>
                    <input 
                      type="range" 
                      min="100" 
                      max="300" 
                      value={normalMean}
                      onChange={(e) => setNormalMean(Number(e.target.value))}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Standard Dev (&sigma;):</span>
                      <span className="text-sky-400 font-bold">{normalStd}</span>
                    </div>
                    <input 
                      type="range" 
                      min="15" 
                      max="75" 
                      value={normalStd}
                      onChange={(e) => setNormalStd(Number(e.target.value))}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>
                </>
              )}

              {activeDist === "exponential" && (
                <div className="space-y-1 font-mono">
                  <div className="flex justify-between text-[9px] text-[#73726C]">
                    <span>Decay (&lambda;):</span>
                    <span className="text-sky-400 font-bold">{lambda}</span>
                  </div>
                  <input 
                    type="range" 
                    min="5" 
                    max="45" 
                    step="1"
                    value={lambda * 1000}
                    onChange={(e) => setLambda(Number(e.target.value) / 1000)}
                    className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                  />
                </div>
              )}

              {activeDist === "bimodal" && (
                <>
                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Peak 1 Mean:</span>
                      <span className="text-sky-400 font-bold">{peak1Mean}</span>
                    </div>
                    <input 
                      type="range" 
                      min="80" 
                      max="180" 
                      value={peak1Mean}
                      onChange={(e) => setPeak1Mean(Number(e.target.value))}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Peak 2 Mean:</span>
                      <span className="text-sky-400 font-bold">{peak2Mean}</span>
                    </div>
                    <input 
                      type="range" 
                      min="220" 
                      max="320" 
                      value={peak2Mean}
                      onChange={(e) => setPeak2Mean(Number(e.target.value))}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>
                </>
              )}

              {activeDist === "uniform" && (
                <>
                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Interval Start (a):</span>
                      <span className="text-sky-400 font-bold">{uniformStart}</span>
                    </div>
                    <input 
                      type="range" 
                      min="30" 
                      max="180" 
                      value={uniformStart}
                      onChange={(e) => setUniformStart(Number(e.target.value))}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Interval End (b):</span>
                      <span className="text-sky-400 font-bold">{uniformEnd}</span>
                    </div>
                    <input 
                      type="range" 
                      min="220" 
                      max="370" 
                      value={uniformEnd}
                      onChange={(e) => setUniformEnd(Number(e.target.value))}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>
                </>
              )}

              {activeDist === "binomial" && (
                <>
                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Trials (n):</span>
                      <span className="text-sky-400 font-bold">{binomialN}</span>
                    </div>
                    <input 
                      type="range" 
                      min="5" 
                      max="30" 
                      value={binomialN}
                      onChange={(e) => setBinomialN(Number(e.target.value))}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1 font-mono">
                    <div className="flex justify-between text-[9px] text-[#73726C]">
                      <span>Prob of Success (p):</span>
                      <span className="text-sky-400 font-bold">{binomialP.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="10" 
                      max="90" 
                      step="5"
                      value={binomialP * 100}
                      onChange={(e) => setBinomialP(Number(e.target.value) / 100)}
                      className="w-full accent-sky-500 bg-white h-1 rounded cursor-pointer"
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          <div className="text-[10px] text-[#9E9D96] leading-relaxed border-t border-[#E5E2D9] pt-3 flex items-start gap-1.5 font-mono">
            <Info className="h-3.5 w-3.5 text-[#9E9D96] flex-shrink-0 mt-0.5" />
            <p>
              Probability density profiles computed dynamically for continuous functions. Adjust sliders to observe change in real-time variance.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
