import { DatasetSummary, ColumnInfo, BatchProcessingReport, ChunkInfo } from "../types";
import { computeColumnStats, computeCorrelationMatrix, runStatisticalTests, detectDatasetOutliers } from "./stats";
import { inferColumnSensitivity } from "./sensitivity";
import { calculateQualityScore } from "../components/DataUpload";

export interface BatchProcessOptions {
  rows: any[];
  fileName: string;
  chunkSize?: number;
  onProgress?: (progress: {
    currentChunk: number;
    totalChunks: number;
    percentage: number;
    rowsProcessed: number;
    estimatedTokensSaved: number;
  }) => void;
}

/**
 * Split array of records into sequential chunks of fixed size
 */
export function partitionIntoChunks<T>(items: T[], chunkSize: number): T[][] {
  if (chunkSize <= 0) chunkSize = 500;
  const chunks: T[][] = [];
  for (let i = 0; i < items.length; i += chunkSize) {
    chunks.push(items.slice(i, i + chunkSize));
  }
  return chunks;
}

/**
 * Asynchronous non-blocking Batch Processor for large datasets.
 * Processes data chunk-by-chunk to enforce token conservation and prevent memory pressure.
 */
export async function processDatasetInBatches(options: BatchProcessOptions): Promise<{
  datasetSummary: DatasetSummary;
  batchReport: BatchProcessingReport;
}> {
  const { rows, fileName, onProgress } = options;
  const chunkSize = options.chunkSize && options.chunkSize > 0 ? options.chunkSize : 500;
  const totalRows = rows.length;

  const startTime = performance.now();

  if (totalRows === 0) {
    throw new Error("Cannot batch process an empty dataset.");
  }

  // Determine headers from first row
  const headers = Object.keys(rows[0]);
  const chunks = partitionIntoChunks(rows, chunkSize);
  const totalChunks = chunks.length;

  const chunkReports: ChunkInfo[] = [];

  // Estimate unchunked raw token cost vs chunked statistical digest
  // Raw JSON representation is typically 12 tokens per row cell
  const estimatedRawTokens = Math.round(totalRows * headers.length * 12);

  let processedRowCount = 0;

  for (let cIdx = 0; cIdx < totalChunks; cIdx++) {
    const chunkStart = performance.now();
    const chunkRows = chunks[cIdx];
    const startRow = cIdx * chunkSize + 1;
    const endRow = startRow + chunkRows.length - 1;

    // Count missing cells in this chunk
    let chunkMissing = 0;
    for (const r of chunkRows) {
      for (const h of headers) {
        const val = r[h];
        if (val === undefined || val === null || val === "" || String(val).trim() === "") {
          chunkMissing++;
        }
      }
    }

    const chunkQuality = calculateQualityScore(chunkRows);
    const chunkDuration = Math.round(performance.now() - chunkStart);

    chunkReports.push({
      chunkIndex: cIdx + 1,
      totalChunks,
      startRow,
      endRow,
      rowCount: chunkRows.length,
      qualityScore: chunkQuality,
      nullCount: chunkMissing,
      processingTimeMs: Math.max(1, chunkDuration),
      sampleRows: chunkRows.slice(0, 3)
    });

    processedRowCount += chunkRows.length;

    // Report progress to UI
    if (onProgress) {
      const pct = Math.round((processedRowCount / totalRows) * 100);
      const estSaved = Math.max(0, Math.round(estimatedRawTokens * (processedRowCount / totalRows) * 0.94));
      onProgress({
        currentChunk: cIdx + 1,
        totalChunks,
        percentage: pct,
        rowsProcessed: processedRowCount,
        estimatedTokensSaved: estSaved
      });
    }

    // Yield control to event loop every chunk to prevent UI freeze and allow render frames
    if (totalChunks > 1) {
      await new Promise(resolve => setTimeout(resolve, 15));
    }
  }

  // Synthesize global aggregated column statistics across all records
  const cols: ColumnInfo[] = headers.map(h => {
    const samples = rows
      .map(r => r[h])
      .filter(v => v !== undefined && v !== null && v !== "");

    let type: ColumnInfo["type"] = "Categorical";
    const numericValues = samples.map(v => Number(v)).filter(v => !isNaN(v));

    if (samples.length > 0 && numericValues.length === samples.length) {
      type = "Numeric";
    } else if (samples.some(v => String(v).includes("-") && !isNaN(Date.parse(String(v))))) {
      type = "Date";
    } else if (samples.some(v => String(v).length > 50)) {
      type = "Text";
    }

    const stats = computeColumnStats(rows.map(r => r[h]), type);
    const sensitivity = inferColumnSensitivity(h, samples);

    return {
      name: h,
      type,
      stats,
      sensitivity
    };
  });

  const numericColNames = cols.filter(c => c.type === "Numeric").map(c => c.name);
  const correlation = computeCorrelationMatrix(rows, numericColNames);
  const statisticalTests = runStatisticalTests(rows, cols);
  const outliers = detectDatasetOutliers(rows, cols);
  const sampleData = rows.slice(0, 15);
  const overallQuality = calculateQualityScore(rows);

  const totalDuration = Math.round(performance.now() - startTime);

  // Digest token consumption is fixed and independent of row count
  const estimatedDigestTokens = Math.round(400 + cols.length * 60);
  const tokensSaved = Math.max(0, estimatedRawTokens - estimatedDigestTokens);
  const reductionPercent = estimatedRawTokens > 0 
    ? Number(((tokensSaved / estimatedRawTokens) * 100).toFixed(1))
    : 92.5;

  const estimatedMemoryMB = Number(((totalRows * cols.length * 8 * 2.2) / (1024 * 1024)).toFixed(2));

  const batchReport: BatchProcessingReport = {
    isBatched: totalChunks > 1,
    chunkSize,
    totalChunks,
    totalRows,
    chunks: chunkReports,
    processingDurationMs: totalDuration,
    estimatedTokensSaved: tokensSaved,
    tokenReductionPercent: reductionPercent,
    peakMemoryMB: Math.max(0.1, estimatedMemoryMB)
  };

  const datasetSummary: DatasetSummary = {
    fileName,
    rowCount: totalRows,
    colCount: cols.length,
    columns: cols,
    correlation,
    statisticalTests,
    sampleData,
    dataQualityScore: overallQuality,
    outliers,
    batchReport
  };

  return { datasetSummary, batchReport };
}
