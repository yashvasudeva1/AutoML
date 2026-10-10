import React, { useState, useMemo, useEffect } from "react";
import { DatasetSummary, ColumnInfo, MLTrainingResult, TrainedModelExportInfo } from "../types";
import { 
  preprocessData, trainTestSplit, standardScale, 
  trainLinearRegression, trainLogisticRegression, trainDecisionTree, trainRandomForest,
  trainGradientBoosting, trainKNearestNeighbors, trainGaussianNaiveBayes, trainElasticNetRegression,
  runKFoldCrossValidation, generateLocalPythonPipeline
} from "../utils/ml";
import { generatePdfReport } from "../utils/pdf";
import FeatureImportancePanel from "./FeatureImportancePanel";
import AutoMLPanel from "./AutoMLPanel";
import ModelExplanationModal from "./ModelExplanationModal";
import { 
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, 
  Legend, BarChart, Bar, Cell 
} from "recharts";
import { 
  Cpu, Play, CheckCircle, RotateCw, AlertTriangle, FileCode, Copy, Download, 
  Loader2, Sparkles, ChevronDown, ChevronUp, Layers, Sliders, ShieldCheck, 
  Activity, BarChart3, HelpCircle, ArrowRight, Trophy, Zap, Check
} from "lucide-react";

export interface ComparedModelRow {
  id: string;
  name: string;
  category: string;
  durationMs: number;
  results: MLTrainingResult;
  primaryMetricName: string;
  primaryMetricVal: number;
  secondaryMetricName: string;
  secondaryMetricVal: number;
  trainMetricVal: number;
  overfitGap: number;
  rank: number;
  isBest: boolean;
  isFastest: boolean;
  description: string;
}

interface MlTabProps {
  summary: DatasetSummary;
  rawData?: any[] | null;
  onModelTrained?: (info: TrainedModelExportInfo) => void;
}

export default function MlTab({ summary, rawData, onModelTrained }: MlTabProps) {
  // Target & Feature selections
  const [targetCol, setTargetCol] = useState<string>(
    summary.columns[summary.columns.length - 1]?.name || ""
  );
  const [selectedFeatures, setSelectedFeatures] = useState<{ [key: string]: boolean }>({});

  // Initialize features (all other columns checked by default)
  useEffect(() => {
    const init: { [key: string]: boolean } = {};
    for (const col of summary.columns) {
      if (col.name !== targetCol) {
        init[col.name] = true;
      }
    }
    setSelectedFeatures(init);
  }, [targetCol, summary.columns]);

  const targetColInfo = useMemo(() => {
    return summary.columns.find(c => c.name === targetCol) || null;
  }, [targetCol, summary.columns]);

  // Auto-detect optimal ML task type
  const taskType = useMemo<"Classification" | "Regression">(() => {
    if (!targetColInfo) return "Classification";
    if (targetColInfo.type === "Numeric" && targetColInfo.stats.uniqueCount > 10) {
      return "Regression";
    }
    return "Classification";
  }, [targetColInfo]);

  // Comprehensive candidate models based on taskType
  const modelsList = useMemo(() => {
    if (taskType === "Classification") {
      return [
        { id: "gradient_boosting", name: "Gradient Boosted Trees (GBDT)", badge: "High Accuracy" },
        { id: "random_forest", name: "Random Forest Classifier (Ensemble)", badge: "Robust" },
        { id: "decision_tree", name: "Decision Tree Classifier (CART)", badge: "Interpretable" },
        { id: "knn", name: "K-Nearest Neighbors (KNN)", badge: "Non-Parametric" },
        { id: "naive_bayes", name: "Gaussian Naive Bayes (GNB)", badge: "Fast" },
        { id: "regularized_logistic", name: "Ridge Logistic Regression (L2)", badge: "Stable" },
        { id: "logistic", name: "Logistic Regression (Standard OLS)", badge: "Linear" }
      ];
    } else {
      return [
        { id: "gradient_boosting", name: "Gradient Boosted Regressor (GBDT)", badge: "High Accuracy" },
        { id: "random_forest", name: "Random Forest Regressor (Ensemble)", badge: "Robust" },
        { id: "decision_tree", name: "Decision Tree Regressor (CART)", badge: "Interpretable" },
        { id: "knn", name: "K-Nearest Neighbors Regressor (KNN)", badge: "Non-Parametric" },
        { id: "elastic_net", name: "ElasticNet Regression (L1 + L2)", badge: "Feature Selection" },
        { id: "ridge", name: "Ridge Regression (L2 Regularized)", badge: "Stable" },
        { id: "linear", name: "Linear Regression (Standard OLS)", badge: "Linear" }
      ];
    }
  }, [taskType]);

  const [selectedModelId, setSelectedModelId] = useState<string>("gradient_boosting");

  useEffect(() => {
    if (modelsList.length > 0) {
      setSelectedModelId(modelsList[0].id);
    }
  }, [modelsList]);

  // Hyperparameters
  const [learningRate, setLearningRate] = useState<number>(0.1);
  const [epochs, setEpochs] = useState<number>(200);
  const [l2Penalty, setL2Penalty] = useState<number>(0.05);
  const [maxDepth, setMaxDepth] = useState<number>(4);
  const [minSamplesSplit, setMinSamplesSplit] = useState<number>(2);
  const [nEstimators, setNEstimators] = useState<number>(12);
  const [kNeighbors, setKNeighbors] = useState<number>(5);
  const [knnWeights, setKnnWeights] = useState<"distance" | "uniform">("distance");
  const [l1Ratio, setL1Ratio] = useState<number>(0.5);
  const [enableCv, setEnableCv] = useState<boolean>(true);

  // Active training state
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentEpoch, setCurrentEpoch] = useState<number>(0);
  const [trainingLosses, setTrainingLosses] = useState<Array<{ epoch: number; trainLoss: number; valLoss: number }>>([]);
  const [trainingError, setTrainingError] = useState<string | null>(null);
  
  // Results
  const [results, setResults] = useState<MLTrainingResult | null>(null);
  const [lastTrainedInfo, setLastTrainedInfo] = useState<TrainedModelExportInfo | null>(null);

  // Parallel Model Comparison State
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [comparisonResults, setComparisonResults] = useState<ComparedModelRow[] | null>(null);
  const [comparisonSuccessMsg, setComparisonSuccessMsg] = useState<string | null>(null);
  const [explainingModel, setExplainingModel] = useState<ComparedModelRow | null>(null);

  // Interactive Live Prediction state
  const [liveFeatureInputs, setLiveFeatureInputs] = useState<{ [key: string]: string }>({});
  const [livePredictionResult, setLivePredictionResult] = useState<any>(null);

  // Python Pipeline Code generation states
  const [isGeneratingPipeline, setIsGeneratingPipeline] = useState<boolean>(false);
  const [pipelineCode, setPipelineCode] = useState<string | null>(null);
  const [copySuccess, setCopySuccess] = useState<boolean>(false);

  // Collapsible AutoML suite state
  const [showAutoML, setShowAutoML] = useState<boolean>(true);

  const activeFeaturesList = useMemo(() => {
    return Object.keys(selectedFeatures).filter(k => selectedFeatures[k] && k !== targetCol);
  }, [selectedFeatures, targetCol]);

  // Initialize live prediction inputs when activeFeaturesList changes
  useEffect(() => {
    const initialInputs: { [key: string]: string } = {};
    activeFeaturesList.forEach(feat => {
      const colInfo = summary.columns.find(c => c.name === feat);
      if (colInfo?.type === "Numeric") {
        initialInputs[feat] = colInfo.stats.median !== undefined ? String(colInfo.stats.median) : "0";
      } else {
        const topVal = colInfo?.stats.frequencies ? Object.keys(colInfo.stats.frequencies)[0] : "0";
        initialInputs[feat] = topVal || "0";
      }
    });
    setLiveFeatureInputs(initialInputs);
    setLivePredictionResult(null);
  }, [activeFeaturesList, summary.columns]);

  // Handler to apply recommended model and parameters
  const handleSelectModelFromAutoML = (modelId: string, suggestedParams?: Record<string, any>) => {
    setSelectedModelId(modelId);
    if (suggestedParams) {
      handleApplyTunedParams(suggestedParams);
    }
  };

  // Handler to apply tuned hyperparameters from HPO
  const handleApplyTunedParams = (params: Record<string, any>) => {
    if (params.learningRate !== undefined) setLearningRate(params.learningRate);
    if (params.epochs !== undefined) setEpochs(params.epochs);
    if (params.l2Penalty !== undefined) setL2Penalty(params.l2Penalty);
    if (params.maxDepth !== undefined) setMaxDepth(params.maxDepth);
    if (params.minSamplesSplit !== undefined) setMinSamplesSplit(params.minSamplesSplit);
    if (params.nEstimators !== undefined) setNEstimators(params.nEstimators);
    if (params.kNeighbors !== undefined) setKNeighbors(params.kNeighbors);
    if (params.weights !== undefined) setKnnWeights(params.weights);
    if (params.l1Ratio !== undefined) setL1Ratio(params.l1Ratio);
  };

  // Run training routine client-side
  const handleTrainModel = async () => {
    if (activeFeaturesList.length === 0) {
      setTrainingError("Please select at least one feature column to train on.");
      return;
    }

    setIsTraining(true);
    setTrainingError(null);
    setResults(null);
    setPipelineCode(null);
    setTrainingLosses([]);
    setCurrentEpoch(0);
    setLivePredictionResult(null);

    try {
      // 1. Preprocess data
      const dataToUse = rawData && rawData.length > 0
        ? (rawData.length > 3000 ? rawData.slice(0, 3000) : rawData)
        : summary.sampleData;
      const prep = preprocessData(dataToUse, activeFeaturesList, targetCol, taskType);
      
      // 2. Train-Test Split (80/20)
      const { XTrain, yTrain, XTest, yTest } = trainTestSplit(prep.X, prep.y, 0.2, 42);

      // 3. Feature Scaling (scaled for gradient descent & distance models)
      const { XTrainScaled, XTestScaled } = standardScale(XTrain, XTest);

      let trainingRes: MLTrainingResult;

      if (selectedModelId === "gradient_boosting") {
        trainingRes = await trainGradientBoosting(
          XTrain,
          yTrain,
          XTest,
          yTest,
          prep.classLabels,
          taskType,
          { nEstimators, maxDepth, learningRate }
        );
        for (let i = 0; i < trainingRes.lossHistory.length; i++) {
          setCurrentEpoch(trainingRes.lossHistory[i].epoch);
          setTrainingLosses(trainingRes.lossHistory.slice(0, i + 1));
          await new Promise(r => setTimeout(r, 25));
        }
      } else if (selectedModelId === "knn") {
        trainingRes = await trainKNearestNeighbors(
          XTrainScaled,
          yTrain,
          XTestScaled,
          yTest,
          prep.classLabels,
          taskType,
          { kNeighbors, weights: knnWeights }
        );
        setCurrentEpoch(1);
        setTrainingLosses(trainingRes.lossHistory);
      } else if (selectedModelId === "naive_bayes" && taskType === "Classification") {
        trainingRes = await trainGaussianNaiveBayes(
          XTrain,
          yTrain,
          XTest,
          yTest,
          prep.classLabels
        );
        setCurrentEpoch(1);
        setTrainingLosses(trainingRes.lossHistory);
      } else if (selectedModelId === "elastic_net" && taskType === "Regression") {
        trainingRes = await trainElasticNetRegression(
          XTrainScaled,
          yTrain,
          XTestScaled,
          yTest,
          epochs,
          learningRate,
          l1Ratio,
          l2Penalty
        );
        for (let i = 0; i < trainingRes.lossHistory.length; i += 5) {
          setCurrentEpoch(trainingRes.lossHistory[i].epoch);
          setTrainingLosses(trainingRes.lossHistory.slice(0, i + 1));
          await new Promise(r => setTimeout(r, 10));
        }
      } else if (selectedModelId === "linear") {
        trainingRes = await trainLinearRegression(
          XTrainScaled,
          yTrain,
          XTestScaled,
          yTest,
          epochs,
          learningRate,
          (epoch, tLoss, vLoss) => {
            setCurrentEpoch(epoch);
            setTrainingLosses(prev => [...prev, { epoch, trainLoss: Number(tLoss.toFixed(4)), valLoss: Number(vLoss.toFixed(4)) }]);
          },
          0
        );
      } else if (selectedModelId === "ridge") {
        trainingRes = await trainLinearRegression(
          XTrainScaled,
          yTrain,
          XTestScaled,
          yTest,
          epochs,
          learningRate,
          (epoch, tLoss, vLoss) => {
            setCurrentEpoch(epoch);
            setTrainingLosses(prev => [...prev, { epoch, trainLoss: Number(tLoss.toFixed(4)), valLoss: Number(vLoss.toFixed(4)) }]);
          },
          l2Penalty
        );
      } else if (selectedModelId === "logistic") {
        trainingRes = await trainLogisticRegression(
          XTrainScaled,
          yTrain,
          XTestScaled,
          yTest,
          prep.classLabels,
          epochs,
          learningRate,
          (epoch, tLoss, vLoss) => {
            setCurrentEpoch(epoch);
            setTrainingLosses(prev => [...prev, { epoch, trainLoss: Number(tLoss.toFixed(4)), valLoss: Number(vLoss.toFixed(4)) }]);
          },
          0
        );
      } else if (selectedModelId === "regularized_logistic") {
        trainingRes = await trainLogisticRegression(
          XTrainScaled,
          yTrain,
          XTestScaled,
          yTest,
          prep.classLabels,
          epochs,
          learningRate,
          (epoch, tLoss, vLoss) => {
            setCurrentEpoch(epoch);
            setTrainingLosses(prev => [...prev, { epoch, trainLoss: Number(tLoss.toFixed(4)), valLoss: Number(vLoss.toFixed(4)) }]);
          },
          l2Penalty
        );
      } else if (selectedModelId === "random_forest") {
        trainingRes = await trainRandomForest(
          XTrain,
          yTrain,
          XTest,
          yTest,
          prep.classLabels,
          taskType,
          { nEstimators, maxDepth, minSamplesSplit }
        );
        for (let i = 0; i < trainingRes.lossHistory.length; i++) {
          setCurrentEpoch(trainingRes.lossHistory[i].epoch);
          setTrainingLosses(trainingRes.lossHistory.slice(0, i + 1));
          await new Promise(r => setTimeout(r, 25));
        }
      } else {
        // Decision Tree (CART)
        trainingRes = await trainDecisionTree(
          XTrain,
          yTrain,
          XTest,
          yTest,
          prep.classLabels,
          maxDepth,
          taskType,
          minSamplesSplit
        );
        for (let i = 0; i < trainingRes.lossHistory.length; i++) {
          setCurrentEpoch(trainingRes.lossHistory[i].epoch);
          setTrainingLosses(trainingRes.lossHistory.slice(0, i + 1));
          await new Promise(r => setTimeout(r, 25));
        }
      }

      // 4. Run optional 5-Fold Cross-Validation for generalizability
      if (enableCv) {
        try {
          const cvRes = await runKFoldCrossValidation(
            prep.X,
            prep.y,
            prep.classLabels,
            taskType,
            selectedModelId,
            {
              learningRate,
              epochs,
              l2Penalty,
              maxDepth,
              minSamplesSplit,
              nEstimators,
              kNeighbors,
              l1Ratio
            },
            5
          );
          trainingRes.crossValidation = cvRes;
        } catch (cvErr) {
          console.warn("CV execution skipped:", cvErr);
        }
      }

      // Map anonymous Feature Importance variables back to actual names
      if (trainingRes.featureImportances) {
        trainingRes.featureImportances = trainingRes.featureImportances.map((f, idx) => ({
          ...f,
          feature: prep.featureNames[idx] || f.feature,
        })).sort((a, b) => b.importance - a.importance);
      }

      const selectedModelName = modelsList.find(m => m.id === selectedModelId)?.name || selectedModelId;
      const trainedInfo: TrainedModelExportInfo = {
        modelId: selectedModelId,
        modelName: selectedModelName,
        taskType,
        targetCol,
        features: activeFeaturesList,
        hyperparameters: {
          learningRate,
          epochs,
          l2Penalty,
          maxDepth,
          minSamplesSplit,
          nEstimators,
          kNeighbors,
          knnWeights,
          l1Ratio
        },
        results: trainingRes,
        trainedAt: new Date().toISOString()
      };

      setResults(trainingRes);
      setLastTrainedInfo(trainedInfo);
      if (onModelTrained) {
        onModelTrained(trainedInfo);
      }
    } catch (err: any) {
      console.error(err);
      setTrainingError(err.message || "Model training crashed. Verify your feature metrics.");
    } finally {
      setIsTraining(false);
    }
  };

  // Parallel Training Run for Three Common Algorithms: Random Forest, Gradient Boosting, Linear Regression
  const handleCompareModels = async () => {
    if (activeFeaturesList.length === 0) {
      setTrainingError("Please select at least one feature column to train on.");
      return;
    }

    setIsComparing(true);
    setTrainingError(null);
    setComparisonSuccessMsg(null);

    try {
      // 1. Data preprocessing
      const dataToUse = rawData && rawData.length > 0
        ? (rawData.length > 3000 ? rawData.slice(0, 3000) : rawData)
        : summary.sampleData;
      const prep = preprocessData(dataToUse, activeFeaturesList, targetCol, taskType);
      
      // 2. Train-Test Split (80/20 partition)
      const { XTrain, yTrain, XTest, yTest } = trainTestSplit(prep.X, prep.y, 0.2, 42);
      const { XTrainScaled, XTestScaled } = standardScale(XTrain, XTest);

      // Define 3 parallel candidate algorithm runners
      const runRandomForest = async () => {
        const t0 = performance.now();
        const res = await trainRandomForest(
          XTrain,
          yTrain,
          XTest,
          yTest,
          prep.classLabels,
          taskType,
          { nEstimators: 12, maxDepth: 5, minSamplesSplit: 2 }
        );
        const duration = Math.round(performance.now() - t0);
        return {
          id: "random_forest",
          name: taskType === "Classification" ? "Random Forest Classifier" : "Random Forest Regressor",
          category: "Ensemble (Bagging)",
          description: "Bagging ensemble of decision trees. Low variance, resistant to non-linear distortion.",
          duration,
          res
        };
      };

      const runGradientBoosting = async () => {
        const t0 = performance.now();
        const res = await trainGradientBoosting(
          XTrain,
          yTrain,
          XTest,
          yTest,
          prep.classLabels,
          taskType,
          { nEstimators: 12, maxDepth: 4, learningRate: 0.1 }
        );
        const duration = Math.round(performance.now() - t0);
        return {
          id: "gradient_boosting",
          name: taskType === "Classification" ? "Gradient Boosted Trees (GBDT)" : "Gradient Boosted Regressor (GBDT)",
          category: "Ensemble (Boosting)",
          description: "Iterative gradient loss minimizer. High precision on complex tabular boundaries.",
          duration,
          res
        };
      };

      const runLinear = async () => {
        const t0 = performance.now();
        let res: MLTrainingResult;
        const isClass = taskType === "Classification";
        if (isClass) {
          res = await trainLogisticRegression(
            XTrainScaled,
            yTrain,
            XTestScaled,
            yTest,
            prep.classLabels,
            160,
            0.1,
            undefined,
            0.01,
            true // silent = true
          );
        } else {
          res = await trainLinearRegression(
            XTrainScaled,
            yTrain,
            XTestScaled,
            yTest,
            160,
            0.1,
            undefined,
            0.01,
            true // silent = true
          );
        }
        const duration = Math.round(performance.now() - t0);
        return {
          id: isClass ? "logistic" : "linear",
          name: isClass ? "Logistic Regression (Linear OLS)" : "Linear Regression (Standard OLS)",
          category: "Parametric Linear (SGD/OLS)",
          description: isClass 
            ? "Convex log-loss optimizer with direct probabilistic interpretations."
            : "Closed-form/SGD OLS baseline; clear linear coefficients and fast inference.",
          duration,
          res
        };
      };

      // Concurrent parallel training execution via Promise.all
      const [rf, gb, lr] = await Promise.all([
        runRandomForest(),
        runGradientBoosting(),
        runLinear()
      ]);

      const primaryKey = taskType === "Classification" ? "Accuracy" : "R2 Score";
      const secondaryKey = taskType === "Classification" ? "F1-Score" : "RMSE";

      const rows: ComparedModelRow[] = [rf, gb, lr].map((item) => {
        // Remap feature importances if present
        if (item.res.featureImportances) {
          item.res.featureImportances = item.res.featureImportances.map((f, idx) => ({
            ...f,
            feature: prep.featureNames[idx] || f.feature,
          })).sort((a, b) => b.importance - a.importance);
        }

        const testMetricPrimary = item.res.testMetrics[primaryKey] ?? 
          (taskType === "Regression" ? item.res.testMetrics["R² Score"] ?? 0 : 0);
        const testMetricSecondary = item.res.testMetrics[secondaryKey] ?? 0;
        const trainMetricPrimary = item.res.trainMetrics[primaryKey] ?? 
          (taskType === "Regression" ? item.res.trainMetrics["R² Score"] ?? 0 : 0);

        const overfitGap = Number((trainMetricPrimary - testMetricPrimary).toFixed(4));

        return {
          id: item.id,
          name: item.name,
          category: item.category,
          durationMs: item.duration,
          results: item.res,
          primaryMetricName: primaryKey,
          primaryMetricVal: testMetricPrimary,
          secondaryMetricName: secondaryKey,
          secondaryMetricVal: testMetricSecondary,
          trainMetricVal: trainMetricPrimary,
          overfitGap,
          rank: 1,
          isBest: false,
          isFastest: false,
          description: item.description
        };
      });

      // Sort by primary score descending (higher Accuracy or R2 is better)
      rows.sort((a, b) => b.primaryMetricVal - a.primaryMetricVal);
      
      const minDuration = Math.min(...rows.map(r => r.durationMs));
      rows.forEach((r, idx) => {
        r.rank = idx + 1;
        r.isBest = idx === 0;
        r.isFastest = r.durationMs === minDuration;
      });

      setComparisonResults(rows);

      // Auto-load best performer into the main results view so user immediately gets deep diagnostics!
      const best = rows[0];
      setSelectedModelId(best.id);
      setResults(best.results);

      const trainedInfo: TrainedModelExportInfo = {
        modelId: best.id,
        modelName: best.name,
        taskType,
        targetCol,
        features: activeFeaturesList,
        hyperparameters: {
          parallelComparisonRun: true,
          bestModelRank: 1,
          primaryScore: best.primaryMetricVal
        },
        results: best.results,
        trainedAt: new Date().toISOString()
      };
      setLastTrainedInfo(trainedInfo);
      if (onModelTrained) {
        onModelTrained(trainedInfo);
      }

      setComparisonSuccessMsg(`Parallel comparison complete! Evaluated Random Forest, Gradient Boosting, and Linear Regression. '${best.name}' ranked #1 with ${primaryKey} = ${(best.primaryMetricVal * (taskType === "Classification" ? 100 : 1)).toFixed(1)}${taskType === "Classification" ? "%" : ""}.`);
    } catch (err: any) {
      console.error("Parallel model comparison failed:", err);
      setTrainingError(err.message || "Failed running parallel model comparison.");
    } finally {
      setIsComparing(false);
    }
  };

  // Handler to select and load any model from the comparison table as the active pipeline
  const handleSelectComparedModel = (model: ComparedModelRow) => {
    setSelectedModelId(model.id);
    setResults(model.results);
    const trainedInfo: TrainedModelExportInfo = {
      modelId: model.id,
      modelName: model.name,
      taskType,
      targetCol,
      features: activeFeaturesList,
      hyperparameters: {
        parallelComparisonSelected: true,
        rank: model.rank,
        primaryScore: model.primaryMetricVal
      },
      results: model.results,
      trainedAt: new Date().toISOString()
    };
    setLastTrainedInfo(trainedInfo);
    if (onModelTrained) {
      onModelTrained(trainedInfo);
    }
    setComparisonSuccessMsg(`Loaded '${model.name}' as the active model. Diagnostics updated below.`);
  };

  // Helper to trigger browser download of arbitrary text/blob file
  const downloadFile = (filename: string, content: string, contentType: string) => {
    try {
      const blob = new Blob([content], { type: contentType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to download file:", err);
    }
  };

  // Download single trained model as JSON artifact
  const handleDownloadModelJson = (modelItem?: ComparedModelRow | TrainedModelExportInfo) => {
    const isCompared = modelItem && "durationMs" in modelItem;
    const targetModel = modelItem || lastTrainedInfo;
    const modelId = isCompared ? (modelItem as ComparedModelRow).id : (targetModel as TrainedModelExportInfo)?.modelId || selectedModelId;
    const modelName = isCompared ? (modelItem as ComparedModelRow).name : (targetModel as TrainedModelExportInfo)?.modelName || modelsList.find(m => m.id === modelId)?.name || modelId;
    const modelResults = isCompared ? (modelItem as ComparedModelRow).results : (targetModel as TrainedModelExportInfo)?.results || results;

    if (!modelResults) return;

    const payload = {
      format: "AutoML_Studio_Model_Package",
      version: "2.0.0",
      metadata: {
        modelId,
        modelName,
        taskType,
        targetColumn: targetCol,
        features: activeFeaturesList,
        featuresCount: activeFeaturesList.length,
        datasetFile: summary.fileName,
        totalDatasetRows: summary.rowCount,
        trainedAt: new Date().toISOString(),
        trainTestSplit: "80% Train, 20% Test",
      },
      evaluation: {
        testMetrics: modelResults.testMetrics,
        trainMetrics: modelResults.trainMetrics,
        crossValidation: modelResults.crossValidation || null,
        featureImportances: modelResults.featureImportances || []
      },
      modelArtifact: modelResults.modelArtifact,
      samplePredictions: modelResults.predictions?.slice(0, 10) || []
    };

    const cleanFilename = `${modelId}_trained_model_${summary.fileName.replace(/[^a-zA-Z0-9_-]/g, "_")}.json`;
    downloadFile(cleanFilename, JSON.stringify(payload, null, 2), "application/json");
  };

  // Download runnable Scikit-Learn Python script
  const handleDownloadModelPython = (modelIdToExport?: string) => {
    const mId = modelIdToExport || selectedModelId;
    const mName = modelsList.find(m => m.id === mId)?.name || mId;
    const code = generateLocalPythonPipeline({
      targetColumn: targetCol,
      features: activeFeaturesList,
      modelId: mId,
      modelName: mName,
      taskType,
      hyperParams: {
        learningRate,
        epochs,
        l2Penalty,
        maxDepth,
        nEstimators,
        kNeighbors
      }
    });

    const cleanFilename = `${mId}_pipeline_${summary.fileName.replace(/[^a-zA-Z0-9_-]/g, "_")}.py`;
    downloadFile(cleanFilename, code, "text/x-python");
  };

  // Download all compared trained models in a consolidated JSON bundle
  const handleDownloadAllComparedModelsBundle = () => {
    if (!comparisonResults || comparisonResults.length === 0) return;

    const bundle = {
      format: "AutoML_Studio_Multi_Model_Comparison_Bundle",
      version: "2.0.0",
      generatedAt: new Date().toISOString(),
      datasetFile: summary.fileName,
      taskType,
      targetColumn: targetCol,
      features: activeFeaturesList,
      comparisonTournament: comparisonResults.map(r => ({
        rank: r.rank,
        isBest: r.isBest,
        isFastest: r.isFastest,
        modelId: r.id,
        modelName: r.name,
        category: r.category,
        durationMs: r.durationMs,
        testMetrics: r.results.testMetrics,
        trainMetrics: r.results.trainMetrics,
        featureImportances: r.results.featureImportances || [],
        modelArtifact: r.results.modelArtifact
      }))
    };

    const cleanFilename = `automl_comparison_bundle_${summary.fileName.replace(/[^a-zA-Z0-9_-]/g, "_")}.json`;
    downloadFile(cleanFilename, JSON.stringify(bundle, null, 2), "application/json");
  };

  // Execute Live Interactive Sample Prediction
  const handleRunLivePrediction = () => {
    if (!results || !results.modelArtifact) return;

    try {
      const artifact = results.modelArtifact;
      // Build raw sample vector matching feature order
      const sampleVector: number[] = activeFeaturesList.map(f => {
        const val = parseFloat(liveFeatureInputs[f] ?? "0");
        return isNaN(val) ? 0 : val;
      });

      let predictionVal: any = null;

      if (artifact.type === "gradient_boosting" && artifact.trees) {
        let currentPred = artifact.initialPrediction ?? 0;
        for (const tree of artifact.trees) {
          let node = tree;
          while (!node.isLeaf) {
            if (sampleVector[node.featureIndex] <= node.threshold) {
              node = node.left;
            } else {
              node = node.right;
            }
          }
          currentPred += (artifact.learningRate ?? 0.1) * (node.value ?? 0);
        }
        if (taskType === "Classification") {
          const prob = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, currentPred))));
          predictionVal = prob >= 0.5 ? artifact.classLabels?.[1] ?? "1" : artifact.classLabels?.[0] ?? "0";
        } else {
          predictionVal = Number(currentPred.toFixed(4));
        }
      } else if (artifact.type === "decision_tree" && artifact.root) {
        let node = artifact.root;
        while (!node.isLeaf) {
          if (sampleVector[node.featureIndex] <= node.threshold) {
            node = node.left;
          } else {
            node = node.right;
          }
        }
        predictionVal = taskType === "Classification" 
          ? artifact.classLabels?.[Math.round(node.value)] ?? node.value
          : Number(Number(node.value).toFixed(4));
      } else if (artifact.type === "linear" && artifact.weights) {
        let dot = artifact.bias ?? 0;
        for (let i = 0; i < sampleVector.length; i++) {
          dot += sampleVector[i] * (artifact.weights[i] ?? 0);
        }
        predictionVal = Number(dot.toFixed(4));
      } else if (artifact.type === "logistic" && artifact.weights) {
        let dot = artifact.bias ?? 0;
        for (let i = 0; i < sampleVector.length; i++) {
          dot += sampleVector[i] * (artifact.weights[i] ?? 0);
        }
        const prob = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, dot))));
        predictionVal = prob >= 0.5 ? artifact.classLabels?.[1] ?? "1" : artifact.classLabels?.[0] ?? "0";
      } else if (artifact.type === "knn") {
        // Nearest neighbor lookup against sample
        const distances = (artifact.XTrain as number[][]).map((row, idx) => {
          let distSq = 0;
          for (let j = 0; j < sampleVector.length; j++) {
            distSq += Math.pow(sampleVector[j] - (row[j] || 0), 2);
          }
          return { dist: Math.sqrt(distSq), y: artifact.yTrain[idx] };
        }).sort((a, b) => a.dist - b.dist).slice(0, artifact.kNeighbors || 5);

        if (taskType === "Classification") {
          const counts: { [k: number]: number } = {};
          distances.forEach(d => { counts[d.y] = (counts[d.y] || 0) + 1; });
          let bestY = distances[0].y;
          let maxC = 0;
          Object.entries(counts).forEach(([yKey, cnt]) => {
            if (cnt > maxC) { maxC = cnt; bestY = Number(yKey); }
          });
          predictionVal = artifact.classLabels?.[bestY] ?? bestY;
        } else {
          const avg = distances.reduce((acc, d) => acc + d.y, 0) / distances.length;
          predictionVal = Number(avg.toFixed(4));
        }
      } else {
        // Fallback to sample prediction
        predictionVal = results.predictions[0]?.predicted ?? "Evaluated";
      }

      setLivePredictionResult(predictionVal);
    } catch (err: any) {
      console.error(err);
      setLivePredictionResult("Error computing prediction");
    }
  };

  // Generate Python Pipeline Code (Server Gemini + Guaranteed Local Fallback)
  const handleGeneratePipeline = async () => {
    setIsGeneratingPipeline(true);
    setPipelineCode(null);
    try {
      const selectedModelName = modelsList.find(m => m.id === selectedModelId)?.name || selectedModelId;
      
      try {
        const response = await fetch("/api/eda/pipeline", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pipelineParams: {
              targetColumn: targetCol,
              features: activeFeaturesList,
              modelName: selectedModelName,
              hyperParams: {
                learning_rate: learningRate,
                epochs,
                max_depth: maxDepth,
                l2_penalty: l2Penalty,
                n_estimators: nEstimators,
                min_samples_split: minSamplesSplit,
                k_neighbors: kNeighbors,
                l1_ratio: l1Ratio
              },
              taskType
            }
          })
        });
        if (response.ok) {
          const data = await response.json();
          if (data.code) {
            setPipelineCode(data.code);
            setIsGeneratingPipeline(false);
            return;
          }
        }
      } catch (apiErr) {
        console.warn("Express pipeline endpoint offline, generating local scikit-learn pipeline:", apiErr);
      }

      // Offline deterministic fallback
      const localCode = generateLocalPythonPipeline({
        targetColumn: targetCol,
        features: activeFeaturesList,
        modelId: selectedModelId,
        modelName: selectedModelName,
        taskType,
        hyperParams: {
          learningRate,
          epochs,
          l2Penalty,
          maxDepth,
          nEstimators,
          kNeighbors,
          l1Ratio
        }
      });
      setPipelineCode(localCode);
    } catch (err: any) {
      console.error(err);
      setTrainingError("Failed to generate pipeline script.");
    } finally {
      setIsGeneratingPipeline(false);
    }
  };

  const handleCopyCode = () => {
    if (pipelineCode) {
      navigator.clipboard.writeText(pipelineCode);
      setCopySuccess(true);
      setTimeout(() => setCopySuccess(false), 2000);
    }
  };

  const handleDownloadCode = () => {
    if (!pipelineCode) return;
    const element = document.createElement("a");
    const file = new Blob([pipelineCode], { type: "text/plain" });
    element.href = URL.createObjectURL(file);
    element.download = `automl_pipeline_${selectedModelId}.py`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleExportFullPdf = () => {
    generatePdfReport(
      summary,
      null,
      results,
      modelsList.find(m => m.id === selectedModelId)?.name || selectedModelId,
      lastTrainedInfo
    );
  };

  return (
    <div className="space-y-6" id="ml-tab-content">
      {/* Target & Problem Definition Section */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
        <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
          <div className="flex items-center space-x-2">
            <Cpu className="h-4 w-4 text-sky-400" />
            <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">Predictive Machine Learning Workbench</h3>
          </div>
          <span className="px-2.5 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider bg-[#FAF9F5] border border-[#E5E2D9] text-[#73726C]">
            Task: <span className="text-sky-400 font-bold">{taskType}</span>
          </span>
        </div>

        <div className="grid sm:grid-cols-2 gap-4 items-center">
          <div className="space-y-1.5">
            <label className="text-[10px] font-mono font-bold text-[#9E9D96] uppercase tracking-wider">Target Variable (Y)</label>
            <select
              value={targetCol}
              onChange={(e) => setTargetCol(e.target.value)}
              className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded px-3 py-2 text-xs font-mono text-[#1F1E1D] focus:outline-none focus:border-sky-500/50"
            >
              {summary.columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.type} — {c.stats.uniqueCount} distinct)
                </option>
              ))}
            </select>
          </div>

          <div className="bg-white p-3 rounded border border-[#E5E2D9] space-y-1">
            <span className="text-[9px] font-mono font-bold text-[#9E9D96] uppercase tracking-wider">Task Formulation</span>
            <p className="text-[11px] font-mono text-[#73726C]">
              {taskType === "Classification" ? (
                <>Predicting discrete categorical classes for target <span className="font-bold text-sky-400">'{targetCol}'</span>.</>
              ) : (
                <>Estimating continuous real-valued quantities for target <span className="font-bold text-sky-400">'{targetCol}'</span>.</>
              )}
            </p>
          </div>
        </div>
      </div>

      {/* Collapsible AutoML Suite */}
      <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded overflow-hidden">
        <button
          onClick={() => setShowAutoML(prev => !prev)}
          className="w-full px-5 py-3 flex items-center justify-between bg-white hover:bg-white border-b border-[#E5E2D9] transition-colors cursor-pointer text-left"
        >
          <div className="flex items-center space-x-2.5">
            <Sparkles className="h-4 w-4 text-sky-400" />
            <span className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
              AutoML Recommendation Engine & Tournament Benchmark
            </span>
          </div>
          <div className="flex items-center space-x-2.5">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                handleCompareModels();
              }}
              disabled={isComparing || isTraining}
              className="px-2.5 py-1 rounded bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-200 text-[10px] font-mono font-bold flex items-center space-x-1 cursor-pointer transition-colors shadow-2xs disabled:opacity-50"
              title="Parallel training run for Random Forest, Gradient Boosting, and Linear Regression"
            >
              <BarChart3 className="h-3 w-3" />
              <span>{isComparing ? "Comparing..." : "Compare Models"}</span>
            </button>

            <span className="font-mono text-[10px] text-[#9E9D96]">
              {showAutoML ? "Hide Suite" : "Show Suite"}
            </span>
            {showAutoML ? (
              <ChevronUp className="h-3.5 w-3.5 text-[#73726C]" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 text-[#73726C]" />
            )}
          </div>
        </button>

        {showAutoML && (
          <div className="p-5">
            <AutoMLPanel
              summary={summary}
              rawData={rawData}
              targetCol={targetCol}
              taskType={taskType}
              activeFeatures={activeFeaturesList}
              currentModelId={selectedModelId}
              onSelectModel={handleSelectModelFromAutoML}
              onApplyTunedParams={handleApplyTunedParams}
            />
          </div>
        )}
      </div>

      {/* Feature Selection and Model Configuration */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left Column: Feature Selection Matrix */}
        <div className="lg:col-span-2 bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
            <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
              Input Predictor Features (X)
            </h4>
            <span className="text-[10px] font-mono text-sky-400 font-bold">
              {activeFeaturesList.length} of {summary.columns.length - 1} Selected
            </span>
          </div>

          <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-2 max-h-60 overflow-y-auto pr-1">
            {summary.columns.map((col) => {
              if (col.name === targetCol) return null;
              const isChecked = !!selectedFeatures[col.name];

              return (
                <label
                  key={col.name}
                  className={`flex items-center space-x-2 p-2 rounded border transition-all cursor-pointer select-none text-[11px] font-mono ${
                    isChecked
                      ? "bg-sky-500/10 border-sky-500/30 text-sky-300"
                      : "bg-white border-[#E5E2D9] text-[#9E9D96] hover:border-[#D5D1C5]"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={(e) =>
                      setSelectedFeatures(prev => ({ ...prev, [col.name]: e.target.checked }))
                    }
                    className="rounded bg-white border-[#D5D1C5] text-sky-500 focus:ring-0 h-3.5 w-3.5 cursor-pointer"
                  />
                  <span className="truncate">{col.name}</span>
                </label>
              );
            })}
          </div>
        </div>

        {/* Right Column: Model Selection & Hyperparameter Knobs */}
        <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
            <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
              Algorithm & Knobs
            </h4>
            <span className="text-[10px] font-mono text-[#9E9D96] uppercase font-bold">
              {modelsList.length} Options
            </span>
          </div>

          <div className="space-y-3">
            <div className="space-y-1">
              <label className="text-[10px] font-mono font-bold text-[#9E9D96] uppercase tracking-wider">Model Architecture</label>
              <select
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
                className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded px-2.5 py-1.5 text-xs font-mono text-[#1F1E1D] focus:outline-none focus:border-sky-500/50"
              >
                {modelsList.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Dynamic Model-Specific Hyperparameters */}
            <div className="pt-2 border-t border-[#E5E2D9] space-y-3">
              {/* Gradient Boosting Knobs */}
              {selectedModelId === "gradient_boosting" && (
                <>
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Estimator Trees</span>
                      <span className="text-sky-400">{nEstimators}</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="30"
                      step="1"
                      value={nEstimators}
                      onChange={(e) => setNEstimators(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Shrinkage / Learning Rate (α)</span>
                      <span className="text-sky-400">{learningRate}</span>
                    </div>
                    <input
                      type="range"
                      min="0.02"
                      max="0.3"
                      step="0.01"
                      value={learningRate}
                      onChange={(e) => setLearningRate(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Max Tree Depth</span>
                      <span className="text-sky-400">{maxDepth}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="6"
                      step="1"
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>
                </>
              )}

              {/* Random Forest Knobs */}
              {selectedModelId === "random_forest" && (
                <>
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Forest Trees</span>
                      <span className="text-sky-400">{nEstimators}</span>
                    </div>
                    <input
                      type="range"
                      min="4"
                      max="24"
                      step="2"
                      value={nEstimators}
                      onChange={(e) => setNEstimators(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Max Tree Depth</span>
                      <span className="text-sky-400">{maxDepth}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="8"
                      step="1"
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>
                </>
              )}

              {/* Decision Tree Knobs */}
              {selectedModelId === "decision_tree" && (
                <>
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Max Tree Depth</span>
                      <span className="text-sky-400">{maxDepth}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="8"
                      step="1"
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Min Samples Split</span>
                      <span className="text-sky-400">{minSamplesSplit}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="8"
                      step="1"
                      value={minSamplesSplit}
                      onChange={(e) => setMinSamplesSplit(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>
                </>
              )}

              {/* KNN Knobs */}
              {selectedModelId === "knn" && (
                <>
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>K Neighbors</span>
                      <span className="text-blue-400">{kNeighbors}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      step="1"
                      value={kNeighbors}
                      onChange={(e) => setKNeighbors(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-blue-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Distance Weighting</span>
                      <span className="text-blue-400 capitalize">{knnWeights}</span>
                    </div>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setKnnWeights("distance")}
                        className={`flex-1 py-1 text-[10px] font-mono rounded border ${
                          knnWeights === "distance"
                            ? "bg-blue-500/20 text-blue-300 border-blue-500/40 font-bold"
                            : "bg-[#FAF9F5] text-[#9E9D96] border-[#E5E2D9]"
                        }`}
                      >
                        Distance
                      </button>
                      <button
                        type="button"
                        onClick={() => setKnnWeights("uniform")}
                        className={`flex-1 py-1 text-[10px] font-mono rounded border ${
                          knnWeights === "uniform"
                            ? "bg-blue-500/20 text-blue-300 border-blue-500/40 font-bold"
                            : "bg-[#FAF9F5] text-[#9E9D96] border-[#E5E2D9]"
                        }`}
                      >
                        Uniform
                      </button>
                    </div>
                  </div>
                </>
              )}

              {/* ElasticNet Knobs */}
              {selectedModelId === "elastic_net" && (
                <>
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>L1 Ratio (Lasso Weight)</span>
                      <span className="text-purple-400">{l1Ratio}</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.1"
                      value={l1Ratio}
                      onChange={(e) => setL1Ratio(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-purple-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>L2 Ridge Penalty</span>
                      <span className="text-purple-400">{l2Penalty}</span>
                    </div>
                    <input
                      type="range"
                      min="0.005"
                      max="0.2"
                      step="0.005"
                      value={l2Penalty}
                      onChange={(e) => setL2Penalty(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-purple-500"
                    />
                  </div>
                </>
              )}

              {/* Linear / Ridge / Logistic Knobs */}
              {(selectedModelId === "linear" || selectedModelId === "logistic" || selectedModelId === "ridge" || selectedModelId === "regularized_logistic") && (
                <>
                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Learning Rate (α)</span>
                      <span className="text-sky-400">{learningRate}</span>
                    </div>
                    <input
                      type="range"
                      min="0.01"
                      max="0.3"
                      step="0.01"
                      value={learningRate}
                      onChange={(e) => setLearningRate(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                      <span>Epochs</span>
                      <span className="text-sky-400">{epochs}</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="400"
                      step="25"
                      value={epochs}
                      onChange={(e) => setEpochs(Number(e.target.value))}
                      className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-sky-500"
                    />
                  </div>

                  {(selectedModelId === "ridge" || selectedModelId === "regularized_logistic") && (
                    <div className="space-y-1">
                      <div className="flex justify-between font-mono font-bold text-[#73726C] uppercase text-[10px]">
                        <span>L2 Penalty (λ)</span>
                        <span className="text-purple-400">{l2Penalty}</span>
                      </div>
                      <input
                        type="range"
                        min="0.005"
                        max="0.2"
                        step="0.005"
                        value={l2Penalty}
                        onChange={(e) => setL2Penalty(Number(e.target.value))}
                        className="w-full h-1 bg-[#F0EEE6] rounded appearance-none cursor-pointer accent-purple-500"
                      />
                    </div>
                  )}
                </>
              )}

              {/* 5-Fold Cross Validation Toggle */}
              <div className="pt-2 border-t border-[#E5E2D9] flex items-center justify-between">
                <span className="text-[10px] font-mono font-bold text-[#73726C] uppercase">
                  5-Fold Cross-Validation
                </span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={enableCv}
                    onChange={(e) => setEnableCv(e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="w-8 h-4 bg-[#F0EEE6] peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-neutral-300 after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-sky-600"></div>
                </label>
              </div>

              {/* Dual Action Buttons: Train Selected & Compare Models */}
              <div className="grid grid-cols-2 gap-2 mt-3">
                <button
                  type="button"
                  onClick={handleTrainModel}
                  disabled={isTraining || isComparing}
                  className="w-full bg-sky-600 hover:bg-sky-500 border border-sky-500/20 text-white font-mono font-bold text-xs py-2.5 rounded transition-all duration-200 flex items-center justify-center space-x-1.5 cursor-pointer disabled:opacity-50 shadow-sm"
                  title="Train single selected model architecture"
                >
                  {isTraining ? (
                    <>
                      <RotateCw className="h-3.5 w-3.5 animate-spin" />
                      <span className="truncate">Fitting ({currentEpoch})...</span>
                    </>
                  ) : (
                    <>
                      <Play className="h-3.5 w-3.5 fill-current" />
                      <span>Train Selected</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={handleCompareModels}
                  disabled={isTraining || isComparing}
                  className="w-full bg-purple-600 hover:bg-purple-500 border border-purple-500/20 text-white font-mono font-bold text-xs py-2.5 rounded transition-all duration-200 flex items-center justify-center space-x-1.5 cursor-pointer disabled:opacity-50 shadow-sm"
                  title="Parallel training run for Random Forest, Gradient Boosting, and Linear Regression"
                >
                  {isComparing ? (
                    <>
                      <RotateCw className="h-3.5 w-3.5 animate-spin" />
                      <span className="truncate">Comparing...</span>
                    </>
                  ) : (
                    <>
                      <BarChart3 className="h-3.5 w-3.5" />
                      <span>Compare Models</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {trainingError && (
        <div className="p-3 bg-rose-500/10 text-rose-400 border border-rose-500/20 rounded flex items-center space-x-2 text-[11px] font-mono font-bold">
          <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0" />
          <span>{trainingError}</span>
        </div>
      )}

      {comparisonSuccessMsg && (
        <div className="p-3 bg-emerald-500/10 text-emerald-800 border border-emerald-500/30 rounded flex items-center space-x-2 text-[11px] font-mono font-bold">
          <CheckCircle className="h-4 w-4 text-emerald-600 flex-shrink-0" />
          <span>{comparisonSuccessMsg}</span>
        </div>
      )}

      {/* ========================================================================= */}
      {/* PARALLEL MODEL COMPARISON RESULTS TABLE */}
      {/* ========================================================================= */}
      {comparisonResults && (
        <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E5E2D9] pb-3">
            <div className="flex items-center space-x-2.5">
              <div className="w-8 h-8 rounded bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-700">
                <BarChart3 className="h-4 w-4" />
              </div>
              <div>
                <div className="flex items-center space-x-2">
                  <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                    Model Comparison Benchmark Matrix
                  </h3>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-purple-50 text-purple-800 border border-purple-200">
                    3 Algorithms Trained in Parallel
                  </span>
                </div>
                <p className="text-[11px] font-mono text-[#73726C]">
                  Direct head-to-head comparison under identical 80/20 train-test partition across {activeFeaturesList.length} features
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-2">
              <button
                type="button"
                onClick={handleDownloadAllComparedModelsBundle}
                className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 text-white text-xs font-mono font-bold rounded transition-colors cursor-pointer flex items-center space-x-1.5 shadow-2xs"
                title="Download all 3 trained models in a consolidated JSON bundle"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Download All (.json)</span>
              </button>

              <button
                type="button"
                onClick={handleCompareModels}
                disabled={isComparing}
                className="px-3 py-1.5 bg-white border border-[#E5E2D9] hover:bg-[#F0EEE6] text-[#1F1E1D] text-xs font-mono font-bold rounded transition-colors cursor-pointer flex items-center space-x-1.5"
              >
                <RotateCw className={`h-3 w-3 ${isComparing ? "animate-spin" : ""}`} />
                <span>Re-run Comparison</span>
              </button>
            </div>
          </div>

          {/* Quick Comparison KPI Pills */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="p-3 bg-white rounded border border-[#E5E2D9] space-y-1">
              <div className="flex items-center justify-between text-[#73726C] text-[10px] font-mono uppercase font-bold">
                <span>Top Performer (#1)</span>
                <Trophy className="h-3.5 w-3.5 text-amber-500" />
              </div>
              <p className="font-mono font-bold text-[#1F1E1D] text-sm">
                {comparisonResults[0].name}
              </p>
              <p className="text-[10px] font-mono text-emerald-700 font-semibold">
                Accuracy/R-squared: {taskType === "Classification" ? `${(comparisonResults[0].primaryMetricVal * 100).toFixed(1)}%` : comparisonResults[0].primaryMetricVal.toFixed(4)}
              </p>
            </div>

            <div className="p-3 bg-white rounded border border-[#E5E2D9] space-y-1">
              <div className="flex items-center justify-between text-[#73726C] text-[10px] font-mono uppercase font-bold">
                <span>Fastest Training</span>
                <Zap className="h-3.5 w-3.5 text-sky-500" />
              </div>
              <p className="font-mono font-bold text-[#1F1E1D] text-sm">
                {comparisonResults.reduce((prev, curr) => curr.durationMs < prev.durationMs ? curr : prev).name}
              </p>
              <p className="text-[10px] font-mono text-sky-600 font-semibold">
                Latency: {Math.min(...comparisonResults.map(r => r.durationMs))} ms
              </p>
            </div>

            <div className="p-3 bg-white rounded border border-[#E5E2D9] space-y-1">
              <div className="flex items-center justify-between text-[#73726C] text-[10px] font-mono uppercase font-bold">
                <span>Task & Target</span>
                <Sliders className="h-3.5 w-3.5 text-purple-500" />
              </div>
              <p className="font-mono font-bold text-[#1F1E1D] text-sm">
                {taskType} on '{targetCol}'
              </p>
              <p className="text-[10px] font-mono text-[#73726C]">
                {activeFeaturesList.length} Input Predictor Vectors
              </p>
            </div>
          </div>

          {/* Interactive Row Click Hint */}
          <div className="flex items-center justify-between text-[11px] text-[#73726C] px-3 py-2 bg-purple-50/50 rounded border border-purple-200/60 font-mono">
            <span className="flex items-center space-x-1.5">
              <Sparkles className="h-3.5 w-3.5 text-purple-600 flex-shrink-0" />
              <span><strong>Model Explanation:</strong> Click any row or the <strong>'Explain'</strong> button to open Gemini's plain-English breakdown of feature importance and decision logic.</span>
            </span>
            <span className="hidden sm:inline text-[10px] text-purple-700 font-bold bg-white px-2 py-0.5 rounded border border-purple-200">
              Gemini 3.8 Flash XAI
            </span>
          </div>

          {/* Comparative Results Table */}
          <div className="bg-white rounded border border-[#E5E2D9] overflow-hidden shadow-2xs">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[10px] font-mono uppercase text-[#73726C] font-bold">
                    <th className="py-2.5 px-3">Rank & Model</th>
                    <th className="py-2.5 px-3 font-bold text-[#1F1E1D]">Accuracy/R-squared</th>
                    <th className="py-2.5 px-3">{taskType === "Classification" ? "F1-Score" : "RMSE"}</th>
                    <th className="py-2.5 px-3">Train Score / Fit Gap</th>
                    <th className="py-2.5 px-3">Training Time</th>
                    <th className="py-2.5 px-3">Architectural Profile</th>
                    <th className="py-2.5 px-3 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#E5E2D9] text-xs font-mono">
                  {comparisonResults.map((row) => {
                    const isSelected = selectedModelId === row.id;
                    const primaryPct = taskType === "Classification" 
                      ? Math.round(row.primaryMetricVal * 100)
                      : Math.max(0, Math.min(100, Math.round(row.primaryMetricVal * 100)));

                    return (
                      <tr 
                        key={row.id} 
                        onClick={() => setExplainingModel(row)}
                        className={`transition-colors cursor-pointer ${
                          isSelected ? "bg-sky-50/70 hover:bg-sky-50" : "hover:bg-purple-50/40"
                        }`}
                        title="Click to view Gemini-powered Model Explanation & Decision Logic"
                      >
                        {/* Rank & Model */}
                        <td className="py-3 px-3">
                          <div className="flex items-center space-x-2">
                            <span className={`inline-flex items-center justify-center w-5 h-5 rounded text-[10px] font-bold ${
                              row.rank === 1 
                                ? "bg-amber-100 text-amber-800 border border-amber-300"
                                : row.rank === 2
                                ? "bg-slate-200 text-slate-800 border border-slate-300"
                                : "bg-orange-100 text-orange-800 border border-orange-200"
                            }`}>
                              #{row.rank}
                            </span>
                            <div>
                              <div className="font-bold text-[#1F1E1D] flex items-center space-x-1.5">
                                <span>{row.name}</span>
                                {row.isBest && (
                                  <span className="px-1.5 py-0.2 bg-emerald-100 text-emerald-800 text-[9px] font-bold rounded">
                                    Best
                                  </span>
                                )}
                              </div>
                              <span className="text-[10px] text-[#73726C]">{row.category}</span>
                            </div>
                          </div>
                        </td>

                        {/* Accuracy/R-squared Column */}
                        <td className="py-3 px-3">
                          <div className="space-y-1">
                            <div className="font-bold text-[#1F1E1D] flex items-center space-x-1.5">
                              <span className={row.isBest ? "text-emerald-700 font-bold" : ""}>
                                {taskType === "Classification" 
                                  ? `${(row.primaryMetricVal * 100).toFixed(1)}%` 
                                  : row.primaryMetricVal.toFixed(4)
                                }
                              </span>
                              <span className="text-[9px] font-mono text-[#73726C] px-1 py-0.2 bg-[#FAF9F5] border border-[#E5E2D9] rounded">
                                {taskType === "Classification" ? "Accuracy" : "R²"}
                              </span>
                            </div>
                            <div className="w-24 h-1.5 bg-[#E5E2D9] rounded-full overflow-hidden">
                              <div 
                                className={`h-full ${row.isBest ? "bg-emerald-500" : "bg-sky-500"}`} 
                                style={{ width: `${primaryPct}%` }}
                              />
                            </div>
                          </div>
                        </td>

                        {/* Secondary Metric */}
                        <td className="py-3 px-3 text-[#1F1E1D]">
                          <span>{row.secondaryMetricVal.toFixed(4)}</span>
                        </td>

                        {/* Train Score & Fit Gap */}
                        <td className="py-3 px-3">
                          <div className="space-y-0.5">
                            <div className="text-[11px] text-[#1F1E1D]">
                              Train: {taskType === "Classification" ? `${(row.trainMetricVal * 100).toFixed(1)}%` : row.trainMetricVal.toFixed(4)}
                            </div>
                            <div className={`text-[10px] ${
                              Math.abs(row.overfitGap) > 0.15 
                                ? "text-amber-600 font-bold" 
                                : "text-[#73726C]"
                            }`}>
                              Δ Gap: {row.overfitGap >= 0 ? `+${row.overfitGap}` : row.overfitGap}
                              {Math.abs(row.overfitGap) > 0.15 && " (High Overfit)"}
                            </div>
                          </div>
                        </td>

                        {/* Latency */}
                        <td className="py-3 px-3">
                          <span className="inline-flex items-center space-x-1 text-[#1F1E1D]">
                            <span>{row.durationMs} ms</span>
                            {row.isFastest && (
                              <Zap className="h-3 w-3 text-sky-500" title="Fastest training convergence" />
                            )}
                          </span>
                        </td>

                        {/* Architectural Profile */}
                        <td className="py-3 px-3 text-[11px] text-[#73726C] max-w-[200px]">
                          <span className="line-clamp-2">{row.description}</span>
                        </td>

                        {/* Action */}
                        <td className="py-3 px-3 text-right">
                          <div className="flex items-center justify-end space-x-1.5" onClick={(e) => e.stopPropagation()}>
                            <button
                              type="button"
                              onClick={() => setExplainingModel(row)}
                              className="px-2 py-1 bg-purple-50 hover:bg-purple-100 text-purple-700 border border-purple-200 hover:border-purple-300 rounded text-[10px] font-bold transition-all cursor-pointer shadow-2xs flex items-center space-x-1"
                              title={`Open Gemini Model Explanation & Decision Logic for ${row.name}`}
                            >
                              <Sparkles className="h-3 w-3 text-purple-600" />
                              <span className="hidden sm:inline">Explain</span>
                            </button>

                            <button
                              type="button"
                              onClick={() => handleDownloadModelJson(row)}
                              className="px-2 py-1 bg-white hover:bg-purple-50 text-purple-700 border border-purple-200 hover:border-purple-300 rounded text-[10px] font-bold transition-all cursor-pointer shadow-2xs flex items-center space-x-1"
                              title={`Download ${row.name} weights and model architecture as JSON`}
                            >
                              <Download className="h-3 w-3" />
                              <span className="hidden sm:inline">JSON</span>
                            </button>

                            {isSelected ? (
                              <span className="inline-flex items-center space-x-1 px-2.5 py-1 rounded bg-sky-100 text-sky-800 text-[10px] font-bold border border-sky-300">
                                <Check className="h-3 w-3" />
                                <span>Active</span>
                              </span>
                            ) : (
                              <button
                                type="button"
                                onClick={() => handleSelectComparedModel(row)}
                                className="px-2.5 py-1 bg-white hover:bg-sky-50 text-sky-600 border border-sky-300 hover:border-sky-400 rounded text-[10px] font-bold transition-all cursor-pointer shadow-2xs"
                              >
                                Select & Load
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Visual Comparison Bar Chart */}
          <div className="p-4 bg-white rounded border border-[#E5E2D9] space-y-2">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="font-bold text-[#1F1E1D] uppercase text-[10px] tracking-wider">
                Visual Scorecard Comparison: Accuracy/R-squared ({taskType === "Classification" ? "Accuracy %" : "R² Score"})
              </span>
              <span className="text-[10px] text-[#73726C]">
                Click 'Select & Load' on any algorithm to inspect its confusion matrix and feature importances
              </span>
            </div>

            <div className="h-36 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart 
                  data={comparisonResults.map(r => ({
                    name: r.name.split(" ")[0] + " " + (r.name.split(" ")[1] || ""),
                    score: Number((r.primaryMetricVal * (taskType === "Classification" ? 100 : 1)).toFixed(2)),
                    duration: r.durationMs,
                    id: r.id
                  }))}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                >
                  <XAxis 
                    type="number" 
                    fontSize={9} 
                    stroke="#73726C" 
                    domain={[0, taskType === "Classification" ? 100 : 'auto']} 
                  />
                  <YAxis type="category" dataKey="name" fontSize={10} stroke="#1F1E1D" width={140} />
                  <Tooltip 
                    formatter={(val: any) => [`${val}${taskType === "Classification" ? "%" : ""}`, `${taskType === "Classification" ? "Accuracy" : "R² Score"}`]}
                    contentStyle={{ backgroundColor: "#FFFFFF", borderColor: "#E5E2D9", borderRadius: "6px", fontSize: "11px" }}
                  />
                  <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                    {comparisonResults.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.rank === 1 ? "#059669" : entry.id === "random_forest" ? "#0284C7" : "#8B5CF6"} 
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* Real-time convergence plot */}
      {isTraining && trainingLosses.length > 0 && (
        <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-3">
          <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
            Real-time Training Convergence Loss
          </h4>
          <div className="h-40 bg-white rounded p-2 border border-[#E5E2D9]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingLosses}>
                <XAxis dataKey="epoch" fontSize={8} stroke="#4b5563" />
                <YAxis fontSize={8} stroke="#4b5563" />
                <Tooltip contentStyle={{ background: "#171717", border: "1px solid #262626", borderRadius: "4px" }} />
                <Line type="monotone" dataKey="trainLoss" name="Training Loss" stroke="#10b981" strokeWidth={1.5} dot={false} />
                <Line type="monotone" dataKey="valLoss" name="Validation Loss" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* COMPREHENSIVE TRAINING RESULTS & DIAGNOSTICS */}
      {/* ========================================================================= */}
      {results && !isTraining && (
        <div className="space-y-6">
          {/* Top Results Banner with PDF Action */}
          <div className="p-4 bg-sky-500/10 border border-sky-500/30 rounded flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center space-x-2.5">
              <CheckCircle className="h-5 w-5 text-sky-400 flex-shrink-0" />
              <div>
                <h4 className="font-mono font-bold text-sky-300 text-xs uppercase tracking-wider">
                  Model Trained Successfully — {modelsList.find(m => m.id === selectedModelId)?.name}
                </h4>
                <p className="text-[11px] font-mono text-sky-400/80">
                  Evaluated on 80/20 train-test split across {activeFeaturesList.length} features.
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={() => handleDownloadModelJson()}
                className="flex items-center space-x-1.5 bg-purple-600 hover:bg-purple-500 text-white font-mono font-bold text-[10px] uppercase tracking-wider px-3 py-1.5 rounded transition-all cursor-pointer border border-purple-500/20 shadow-sm"
                title="Download complete trained model weights and architecture JSON"
              >
                <Download className="h-3 w-3" />
                <span>Download Model (.json)</span>
              </button>

              <button
                type="button"
                onClick={() => handleDownloadModelPython()}
                className="flex items-center space-x-1.5 bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold text-[10px] uppercase tracking-wider px-3 py-1.5 rounded transition-all cursor-pointer border border-emerald-500/20 shadow-sm"
                title="Download executable scikit-learn Python pipeline script"
              >
                <FileCode className="h-3 w-3" />
                <span>Download Pipeline (.py)</span>
              </button>

              <button
                type="button"
                onClick={handleExportFullPdf}
                className="flex items-center space-x-1.5 bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-[10px] uppercase tracking-wider px-3 py-1.5 rounded transition-all cursor-pointer border border-sky-500/20 shadow-sm"
              >
                <Download className="h-3 w-3" />
                <span>Export PDF</span>
              </button>
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {/* Metric Scorecard */}
            <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
                <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                  Model Performance Scorecard
                </h3>
                <span className="text-[10px] font-mono font-bold text-sky-400 uppercase">
                  Optimal Convergence
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2 bg-white p-3 rounded border border-[#E5E2D9]">
                  <span className="text-[9px] font-mono font-bold text-[#9E9D96] uppercase tracking-wider">Train Set Metrics</span>
                  <div className="space-y-1.5 font-mono">
                    {Object.entries(results.trainMetrics).map(([name, val], idx) => (
                      <div key={idx} className="flex justify-between text-[11px] border-b border-[#E5E2D9] pb-1 last:border-0">
                        <span className="text-[#73726C]">{name}</span>
                        <span className="font-bold text-[#1F1E1D]">{val}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="space-y-2 bg-white p-3 rounded border border-[#E5E2D9]">
                  <span className="text-[9px] font-mono font-bold text-sky-400 uppercase tracking-wider">Validation Set Metrics</span>
                  <div className="space-y-1.5 font-mono">
                    {Object.entries(results.testMetrics).map(([name, val], idx) => (
                      <div key={idx} className="flex justify-between text-[11px] border-b border-[#E5E2D9] pb-1 last:border-0">
                        <span className="text-[#73726C]">{name}</span>
                        <span className="font-bold text-sky-400">{val}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 5-Fold Cross-Validation Display */}
              {results.crossValidation && (
                <div className="p-3 bg-white rounded border border-[#E5E2D9] space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-mono font-bold text-[#73726C] uppercase tracking-wider">
                      5-Fold Cross-Validation Stability
                    </span>
                    <span className="text-[11px] font-mono font-bold text-sky-400">
                      Mean: {(results.crossValidation.meanScore * 100).toFixed(1)}% (±{(results.crossValidation.stdScore * 100).toFixed(2)}%)
                    </span>
                  </div>
                  <div className="grid grid-cols-5 gap-1.5 font-mono text-[10px] text-center">
                    {results.crossValidation.foldScores.map((score, idx) => (
                      <div key={idx} className="p-1.5 bg-[#FAF9F5] border border-[#E5E2D9] rounded">
                        <div className="text-[#9E9D96] text-[8px] uppercase">Fold {idx + 1}</div>
                        <div className="font-bold text-[#1F1E1D]">{(score * 100).toFixed(1)}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Prediction vs Ground Truth Preview */}
            <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5 mb-3">
                  <h3 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                    Prediction vs. Ground Truth Sample
                  </h3>
                  <span className="text-[10px] font-mono text-[#9E9D96]">First 8 Validation Rows</span>
                </div>

                <div className="overflow-x-auto border border-[#E5E2D9] rounded">
                  <table className="w-full text-left border-collapse text-[11px]">
                    <thead>
                      <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[#9E9D96] font-bold uppercase tracking-wider font-mono text-[9px]">
                        <th className="py-2 px-2.5">Row</th>
                        <th className="py-2 px-2.5">Actual Value</th>
                        <th className="py-2 px-2.5">Predicted</th>
                        <th className="py-2 px-2.5">Verdict</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-800 text-[#73726C] font-mono">
                      {results.predictions.slice(0, 8).map((p, idx) => {
                        const isCorrect = taskType === "Classification" 
                          ? String(p.actual) === String(p.predicted) 
                          : Math.abs(Number(p.actual) - Number(p.predicted)) <= (Math.abs(Number(p.actual)) * 0.15 || 1);

                        return (
                          <tr key={idx} className="hover:bg-white">
                            <td className="py-1.5 px-2.5 text-[#9E9D96]">#{idx + 1}</td>
                            <td className="py-1.5 px-2.5 font-bold text-[#1F1E1D]">{String(p.actual)}</td>
                            <td className="py-1.5 px-2.5 font-bold text-sky-400">{String(p.predicted)}</td>
                            <td className="py-1.5 px-2.5">
                              <span className={`px-1.5 py-0.5 rounded text-[8.5px] font-bold uppercase tracking-wider border ${
                                isCorrect 
                                  ? "bg-sky-500/10 text-sky-400 border-sky-500/20" 
                                  : "bg-rose-500/10 text-rose-400 border-rose-500/20"
                              }`}>
                                {isCorrect ? "Match" : "Deviation"}
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Code Export Button */}
              <div className="pt-3 border-t border-[#E5E2D9]">
                <button
                  onClick={handleGeneratePipeline}
                  disabled={isGeneratingPipeline}
                  className="w-full bg-[#FAF9F5] hover:bg-white border border-[#E5E2D9] text-[#1F1E1D] font-mono font-bold text-xs py-2 rounded transition-all flex items-center justify-center space-x-2 cursor-pointer"
                >
                  {isGeneratingPipeline ? (
                    <>
                      <Loader2 className="h-3 w-3 animate-spin text-sky-400" />
                      <span>Generating Scikit-Learn Pipeline...</span>
                    </>
                  ) : (
                    <>
                      <FileCode className="h-3 w-3 text-sky-400" />
                      <span>Export Python Production Script</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* ========================================================================= */}
          {/* DIAGNOSTIC PANELS: CONFUSION MATRIX & RESIDUAL ANALYSIS */}
          {/* ========================================================================= */}
          {taskType === "Classification" && results.confusionMatrix && (
            <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
                <div className="flex items-center space-x-2">
                  <Layers className="h-4 w-4 text-sky-400" />
                  <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                    Classification Confusion Matrix & Class Diagnostics
                  </h4>
                </div>
                <span className="text-[10px] font-mono text-[#9E9D96]">
                  {results.confusionMatrix.labels.length} Discrete Classes
                </span>
              </div>

              <div className="grid md:grid-cols-2 gap-6 items-start">
                {/* Visual Confusion Matrix Table */}
                <div className="space-y-2 font-mono">
                  <span className="text-[9px] font-bold text-[#9E9D96] uppercase tracking-wider">
                    Actual vs. Predicted Heatmap
                  </span>
                  <div className="overflow-x-auto border border-[#E5E2D9] rounded">
                    <table className="w-full text-center text-xs">
                      <thead>
                        <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[#9E9D96] text-[9px] uppercase">
                          <th className="p-2 text-left">Actual \ Pred</th>
                          {results.confusionMatrix.labels.map(lbl => (
                            <th key={lbl} className="p-2 font-bold text-[#73726C]">
                              {lbl.slice(0, 8)}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.confusionMatrix.matrix.map((row, rIdx) => (
                          <tr key={rIdx} className="border-b border-[#E5E2D9] last:border-0">
                            <td className="p-2 text-left font-bold text-[#73726C] bg-white text-[10px]">
                              {results.confusionMatrix?.labels[rIdx]?.slice(0, 10)}
                            </td>
                            {row.map((cellVal, cIdx) => {
                              const isDiagonal = rIdx === cIdx;
                              return (
                                <td
                                  key={cIdx}
                                  className={`p-2.5 font-bold ${
                                    isDiagonal 
                                      ? "bg-sky-500/20 text-sky-400 border border-sky-500/30" 
                                      : cellVal > 0 
                                      ? "bg-rose-500/10 text-rose-400" 
                                      : "text-[#73726C]"
                                  }`}
                                >
                                  {cellVal}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Per-Class Precision / Recall / F1 Table */}
                <div className="space-y-2 font-mono">
                  <span className="text-[9px] font-bold text-[#9E9D96] uppercase tracking-wider">
                    Per-Class Performance Scorecard
                  </span>
                  <div className="overflow-x-auto border border-[#E5E2D9] rounded">
                    <table className="w-full text-left text-[11px]">
                      <thead>
                        <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[#9E9D96] text-[9px] uppercase">
                          <th className="p-2">Class</th>
                          <th className="p-2">Precision</th>
                          <th className="p-2">Recall</th>
                          <th className="p-2">F1</th>
                          <th className="p-2">Support</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-neutral-800 text-[#73726C]">
                        {Object.entries(results.confusionMatrix.perClass).map(([lbl, met]: [string, any], idx) => (
                          <tr key={idx} className="hover:bg-white">
                            <td className="p-2 font-bold text-[#1F1E1D]">{lbl}</td>
                            <td className="p-2 font-bold text-sky-400">{(met.precision * 100).toFixed(1)}%</td>
                            <td className="p-2 font-bold text-blue-400">{(met.recall * 100).toFixed(1)}%</td>
                            <td className="p-2 font-bold text-purple-400">{met.f1.toFixed(3)}</td>
                            <td className="p-2 text-[#9E9D96]">{met.support}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {taskType === "Regression" && results.residualAnalysis && (
            <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-sky-400" />
                  <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                    Regression Residual Analysis & Error Homoscedasticity
                  </h4>
                </div>
                <div className="flex items-center space-x-3 text-[10px] font-mono">
                  <span className="text-[#73726C]">Mean: <strong className="text-sky-400">{results.residualAnalysis.meanResidual}</strong></span>
                  <span className="text-[#73726C]">Std Error: <strong className="text-[#1F1E1D]">{results.residualAnalysis.stdResidual}</strong></span>
                  <span className="text-[#73726C]">Max Error: <strong className="text-rose-400">{results.residualAnalysis.maxResidual}</strong></span>
                </div>
              </div>

              {/* Residuals Table & Distribution summary */}
              <div className="overflow-x-auto border border-[#E5E2D9] rounded font-mono text-[11px]">
                <table className="w-full text-left">
                  <thead>
                    <tr className="bg-[#FAF9F5] border-b border-[#E5E2D9] text-[#9E9D96] text-[9px] uppercase">
                      <th className="p-2">Index</th>
                      <th className="p-2">Actual (y)</th>
                      <th className="p-2">Predicted (ŷ)</th>
                      <th className="p-2">Residual Error (y - ŷ)</th>
                      <th className="p-2">Relative Error %</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-800 text-[#73726C]">
                    {results.residualAnalysis.residuals.slice(0, 6).map((item, idx) => {
                      const relPct = Math.abs(item.residual / (item.actual || 1)) * 100;
                      return (
                        <tr key={idx} className="hover:bg-white">
                          <td className="p-2 text-[#9E9D96]">Record #{idx + 1}</td>
                          <td className="p-2 font-bold text-[#1F1E1D]">{item.actual.toFixed(2)}</td>
                          <td className="p-2 font-bold text-sky-400">{item.predicted.toFixed(2)}</td>
                          <td className={`p-2 font-bold ${item.residual >= 0 ? "text-sky-400" : "text-rose-400"}`}>
                            {item.residual.toFixed(3)}
                          </td>
                          <td className="p-2 text-[#73726C]">{relPct.toFixed(1)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ========================================================================= */}
          {/* TRAINED MODEL EXPORT & DEPLOYMENT HUB */}
          {/* ========================================================================= */}
          <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
              <div className="flex items-center space-x-2">
                <Download className="h-4 w-4 text-purple-600" />
                <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                  Download Trained Model & Deployment Artifacts
                </h4>
              </div>
              <span className="text-[10px] font-mono text-purple-700 bg-purple-50 px-2 py-0.5 rounded border border-purple-200 font-bold">
                Production Ready
              </span>
            </div>

            <p className="text-xs font-mono text-[#73726C]">
              Export the fitted <strong>{modelsList.find(m => m.id === selectedModelId)?.name || selectedModelId}</strong> pipeline in multiple distribution formats for web, edge runtime, or Python cloud microservices.
            </p>

            <div className="grid sm:grid-cols-3 gap-3">
              {/* Option 1: Serialized Model JSON */}
              <div className="bg-white p-4 rounded border border-[#E5E2D9] flex flex-col justify-between space-y-3 shadow-2xs">
                <div className="space-y-1.5">
                  <div className="w-8 h-8 rounded bg-purple-50 border border-purple-200 text-purple-700 flex items-center justify-center">
                    <Cpu className="h-4 w-4" />
                  </div>
                  <h5 className="font-mono font-bold text-xs text-[#1F1E1D]">
                    Serialized Model Artifact (.json)
                  </h5>
                  <p className="text-[11px] font-mono text-[#73726C] leading-relaxed">
                    Complete weights, tree nodes, decision splits, and normalization parameters for direct browser or Node.js inference.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => handleDownloadModelJson()}
                  className="w-full bg-purple-600 hover:bg-purple-500 text-white font-mono font-bold text-xs py-2 rounded transition-all cursor-pointer flex items-center justify-center space-x-1.5 shadow-2xs"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Download .json</span>
                </button>
              </div>

              {/* Option 2: Python Scikit-Learn Pipeline */}
              <div className="bg-white p-4 rounded border border-[#E5E2D9] flex flex-col justify-between space-y-3 shadow-2xs">
                <div className="space-y-1.5">
                  <div className="w-8 h-8 rounded bg-emerald-50 border border-emerald-200 text-emerald-700 flex items-center justify-center">
                    <FileCode className="h-4 w-4" />
                  </div>
                  <h5 className="font-mono font-bold text-xs text-[#1F1E1D]">
                    Production Python Pipeline (.py)
                  </h5>
                  <p className="text-[11px] font-mono text-[#73726C] leading-relaxed">
                    Clean Python script implementing standard preprocessing, cross-validation, and joblib pickle serialization.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => handleDownloadModelPython()}
                  className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-mono font-bold text-xs py-2 rounded transition-all cursor-pointer flex items-center justify-center space-x-1.5 shadow-2xs"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Download .py</span>
                </button>
              </div>

              {/* Option 3: Full Analytical PDF Audit Report */}
              <div className="bg-white p-4 rounded border border-[#E5E2D9] flex flex-col justify-between space-y-3 shadow-2xs">
                <div className="space-y-1.5">
                  <div className="w-8 h-8 rounded bg-sky-50 border border-sky-200 text-sky-700 flex items-center justify-center">
                    <BarChart3 className="h-4 w-4" />
                  </div>
                  <h5 className="font-mono font-bold text-xs text-[#1F1E1D]">
                    Full Analytical Report (.pdf)
                  </h5>
                  <p className="text-[11px] font-mono text-[#73726C] leading-relaxed">
                    Comprehensive multi-page audit report with model scorecards, confusion matrix, residuals, and publication charts.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleExportFullPdf}
                  className="w-full bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-xs py-2 rounded transition-all cursor-pointer flex items-center justify-center space-x-1.5 shadow-2xs"
                >
                  <Download className="h-3.5 w-3.5" />
                  <span>Download .pdf</span>
                </button>
              </div>
            </div>
          </div>

          {/* ========================================================================= */}
          {/* INTERACTIVE LIVE PREDICTION SANDBOX */}
          {/* ========================================================================= */}
          <div className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
              <div className="flex items-center space-x-2">
                <Sliders className="h-4 w-4 text-sky-400" />
                <h4 className="font-mono font-bold text-[#1F1E1D] text-xs uppercase tracking-wider">
                  Interactive Live Prediction Sandbox
                </h4>
              </div>
              <span className="text-[10px] font-mono text-[#9E9D96]">
                Test trained model on custom inputs
              </span>
            </div>

            <div className="grid sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {activeFeaturesList.slice(0, 8).map(feat => {
                const colInfo = summary.columns.find(c => c.name === feat);
                return (
                  <div key={feat} className="space-y-1 font-mono text-[11px]">
                    <label className="text-[#73726C] font-medium truncate block" title={feat}>
                      {feat}
                    </label>
                    <input
                      type="text"
                      value={liveFeatureInputs[feat] ?? ""}
                      onChange={(e) => setLiveFeatureInputs(prev => ({ ...prev, [feat]: e.target.value }))}
                      placeholder={colInfo?.type === "Numeric" ? "e.g. 24.5" : "e.g. Category"}
                      className="w-full bg-[#FAF9F5] border border-[#E5E2D9] rounded px-2.5 py-1.5 text-[#1F1E1D] focus:outline-none focus:border-sky-500/50"
                    />
                  </div>
                );
              })}
            </div>

            <div className="pt-2 flex flex-wrap items-center gap-4">
              <button
                onClick={handleRunLivePrediction}
                className="bg-sky-600 hover:bg-sky-500 text-white font-mono font-bold text-xs px-4 py-2 rounded transition-all cursor-pointer flex items-center space-x-2 shadow-sm"
              >
                <span>Compute Live Prediction</span>
                <ArrowRight className="h-3.5 w-3.5" />
              </button>

              {livePredictionResult !== null && (
                <div className="px-4 py-1.5 bg-sky-500/10 border border-sky-500/30 rounded flex items-center space-x-2 font-mono text-xs">
                  <span className="text-[#73726C]">Target '{targetCol}':</span>
                  <span className="font-bold text-sky-400 text-sm">
                    {String(livePredictionResult)}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Feature Importance Segment */}
          {results.featureImportances && results.featureImportances.length > 0 && (
            <div className="animate-fade-in">
              <FeatureImportancePanel 
                featureImportances={results.featureImportances} 
                modelId={selectedModelId} 
                taskType={taskType} 
              />
            </div>
          )}
        </div>
      )}

      {/* Python Code View */}
      {pipelineCode && (
        <div id="python-pipeline-code-container" className="bg-[#FAF9F5] border border-[#E5E2D9] rounded p-5 space-y-4">
          <div className="flex items-center justify-between border-b border-[#E5E2D9] pb-2.5">
            <span className="text-xs text-[#73726C] font-mono font-bold">
              production_automl_pipeline.py
            </span>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={handleCopyCode}
                className="px-2.5 py-1 bg-[#F0EEE6] hover:bg-[#D5D1C5] text-[#73726C] hover:text-white rounded text-[11px] font-mono font-bold transition-all flex items-center space-x-1.5 cursor-pointer"
              >
                <Copy className="h-3 w-3" />
                <span>{copySuccess ? "Copied!" : "Copy Code"}</span>
              </button>
              
              <button
                onClick={handleDownloadCode}
                className="px-2.5 py-1 bg-sky-600 hover:bg-sky-500 text-white rounded text-[11px] font-mono font-bold transition-all flex items-center space-x-1.5 cursor-pointer"
              >
                <Download className="h-3 w-3 text-white" />
                <span>Download .py</span>
              </button>
            </div>
          </div>

          <pre className="text-xs text-[#73726C] font-mono overflow-x-auto max-h-96 leading-relaxed bg-white p-3 rounded border border-[#E5E2D9]">
            <code>{pipelineCode}</code>
          </pre>
        </div>
      )}

      {/* Gemini Model Explanation & Decision Logic Modal */}
      <ModelExplanationModal
        isOpen={!!explainingModel}
        onClose={() => setExplainingModel(null)}
        model={explainingModel}
        taskType={taskType}
        targetCol={targetCol}
        fileName={summary.fileName}
        features={activeFeaturesList}
        onSelectModel={handleSelectComparedModel}
        onDownloadJson={handleDownloadModelJson}
      />
    </div>
  );
}
