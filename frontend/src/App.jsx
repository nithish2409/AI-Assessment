import { lazy, Suspense, useRef, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import Layout from "./components/Layout";
import LoadingState from "./components/LoadingState";

const DashboardOverview = lazy(() => import("./pages/DashboardOverview"));
const ImageComparison = lazy(() => import("./pages/ImageComparison"));
const HybridProgression = lazy(() => import("./pages/HybridProgression"));
const Inspection = lazy(() => import("./pages/Inspection"));
const NewsClassification = lazy(() => import("./pages/NewsClassification"));
const FeatureAnalysis = lazy(() => import("./pages/FeatureAnalysis"));
const TextSimilarity = lazy(() => import("./pages/TextSimilarity"));
const SystemStatus = lazy(() => import("./pages/SystemStatus"));

export default function App() {
  const [article, setArticleState] = useState("");
  const [analysis, setAnalysis] = useState(null);

  const requestVersion = useRef(0);

  const setArticle = (value) => {
    requestVersion.current += 1;
    setArticleState(value);
    setAnalysis(null);
  };

  const beginRequest = () => {
    requestVersion.current += 1;
    return requestVersion.current;
  };

  const isCurrentRequest = (version) =>
    requestVersion.current === version;

  const nlpSharedProps = {
    article,
    setArticle,
    analysis,
    setAnalysis,
    beginRequest,
    isCurrentRequest,
  };

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          {/* Main Dashboard Overview */}
          <Route
            path="/"
            element={
              <Suspense fallback={<LoadingState title="Loading Overview..." />}>
                <DashboardOverview />
              </Suspense>
            }
          />

          {/* Task 2: Before/After Image Comparison */}
          <Route
            path="/image-comparison"
            element={
              <Suspense fallback={<LoadingState title="Loading Image Comparison..." />}>
                <ImageComparison />
              </Suspense>
            }
          />

          {/* Additional Feature: Hybrid Crack Progression Analysis */}
          <Route
            path="/hybrid-analysis"
            element={
              <Suspense fallback={<LoadingState title="Loading Hybrid Progression..." />}>
                <HybridProgression />
              </Suspense>
            }
          />

          {/* Task 1: AI Inspection & Visual Severity */}
          <Route
            path="/inspection"
            element={
              <Suspense fallback={<LoadingState title="Loading Inspection View..." />}>
                <Inspection />
              </Suspense>
            }
          />

          {/* Task 3: NLP News Classification & Exploration */}
          <Route
            path="/news-classification"
            element={
              <Suspense fallback={<LoadingState title="Loading News Classifier..." />}>
                <NewsClassification {...nlpSharedProps} />
              </Suspense>
            }
          />
          <Route
            path="/features"
            element={
              <Suspense fallback={<LoadingState title="Loading Feature Analysis..." />}>
                <FeatureAnalysis {...nlpSharedProps} />
              </Suspense>
            }
          />
          <Route
            path="/similarity"
            element={
              <Suspense fallback={<LoadingState title="Loading Text Similarity..." />}>
                <TextSimilarity />
              </Suspense>
            }
          />

          {/* System Diagnostics */}
          <Route
            path="/system-status"
            element={
              <Suspense fallback={<LoadingState title="Loading System Status..." />}>
                <SystemStatus />
              </Suspense>
            }
          />

          {/* Convenient Route Aliases */}
          <Route path="/comparison" element={<Navigate to="/image-comparison" replace />} />
          <Route path="/hybrid" element={<Navigate to="/hybrid-analysis" replace />} />
          <Route path="/news" element={<Navigate to="/news-classification" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
