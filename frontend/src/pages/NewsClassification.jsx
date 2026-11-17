import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
    Newspaper,
    Sparkles,
    ArrowRight,
    RotateCcw,
    ChartNoAxesColumn,
    GitCompareArrows,
    Layers,
    Info,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ResultCard from "../components/ResultCard";
import FeatureChart from "../components/FeatureChart";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import {
    analyzeNews,
    classifyNews,
    getModelMetrics,
    parseApiError,
} from "../services/api";

const CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"];

const SAMPLE_ARTICLES = [
    {
        title: "Space Exploration",
        category: "Sci/Tech",
        text: "NASA and European Space Agency scientists confirmed new spectral observations from the James Webb Space Telescope revealing atmospheric signatures on distant rocky exoplanets.",
    },
    {
        title: "Financial Markets",
        category: "Business",
        text: "Wall Street indices surged to record highs following a quarterly earnings report from major technology and manufacturing conglomerates showing resilient revenue growth.",
    },
    {
        title: "Championship Victory",
        category: "Sports",
        text: "The national soccer squad clinched the championship title in extra time with a decisive volley from the star striker, concluding a thrilling tournament final.",
    },
    {
        title: "International Summit",
        category: "World",
        text: "Delegates from fifty nations convened at the international climate summit to finalize bilateral agreements addressing carbon reduction targets and sustainable infrastructure development.",
    },
];

export default function NewsClassification({
    article = "",
    setArticle,
    analysis = null,
    setAnalysis,
    beginRequest,
    isCurrentRequest,
}) {
    const [localArticle, setLocalArticle] = useState(article);
    const [localAnalysis, setLocalAnalysis] = useState(analysis);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const navigate = useNavigate();

    // Use shared state if provided, or local fallback
    const currentArticle = setArticle ? article : localArticle;
    const setCurrentArticle = (val) => {
        if (setArticle) setArticle(val);
        else setLocalArticle(val);
    };

    const currentAnalysis = setAnalysis ? analysis : localAnalysis;
    const setCurrentAnalysis = (val) => {
        if (setAnalysis) setAnalysis(val);
        else setLocalAnalysis(val);
    };

    useEffect(() => {
        getModelMetrics()
            .then(setMetrics)
            .catch((err) => {
                console.error("NLP Metrics unavailable:", err);
            });
    }, []);

    const handleClassify = async () => {
        const trimmed = currentArticle.trim();
        if (!trimmed) {
            setError("Please enter a news article or headline before classifying.");
            return;
        }

        const version = beginRequest ? beginRequest() : 1;
        setLoading(true);
        setError(null);
        setCurrentAnalysis(null);

        try {
            const data = await analyzeNews(trimmed, 10);
            if (!isCurrentRequest || isCurrentRequest(version)) {
                setCurrentAnalysis(data);
            }
        } catch (err) {
            if (!isCurrentRequest || isCurrentRequest(version)) {
                setError(parseApiError(err));
            }
        } finally {
            if (!isCurrentRequest || isCurrentRequest(version)) {
                setLoading(false);
            }
        }
    };

    const handleClear = () => {
        setCurrentArticle("");
        setCurrentAnalysis(null);
        setError(null);
    };

    const classification = currentAnalysis?.classification;
    const probabilities = classification?.probabilities
        ? Object.entries(classification.probabilities)
              .map(([cat, prob]) => ({
                  category: cat,
                  probability: prob * 100,
              }))
              .sort((a, b) => b.probability - a.probability)
        : [];

    return (
        <div>
            <PageHeader
                title="News Classification"
                description="Classify news articles and headlines into AG News categories (World, Sports, Business, Sci/Tech) using TF-IDF feature extraction and Logistic Regression."
                badge="Task 3"
                icon={Newspaper}
                actions={
                    <div className="flex items-center gap-2">
                        <button
                            type="button"
                            onClick={() => navigate("/features")}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-50"
                        >
                            <ChartNoAxesColumn size={14} /> Feature Analysis
                        </button>
                        <button
                            type="button"
                            onClick={() => navigate("/similarity")}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-50"
                        >
                            <GitCompareArrows size={14} /> Text Similarity
                        </button>
                    </div>
                }
            />

            {/* Offline Model Evaluation Metrics */}
            <div className="mb-6 grid gap-4 sm:grid-cols-3">
                <ResultCard
                    title="Model Test Accuracy"
                    value={
                        metrics
                            ? `${(metrics.accuracy * 100).toFixed(2)}%`
                            : "N/A"
                    }
                    subtitle="Evaluated on AG News test set"
                    badge="Logistic Regression"
                />

                <ResultCard
                    title="Macro F1-Score"
                    value={
                        metrics
                            ? `${(metrics.macro_f1 * 100).toFixed(2)}%`
                            : "N/A"
                    }
                    subtitle="Balanced four-class metric"
                    badge="Macro Avg"
                />

                <ResultCard
                    title="Evaluated Test Samples"
                    value={
                        metrics
                            ? metrics.testing_samples.toLocaleString()
                            : "N/A"
                    }
                    subtitle="Held-out evaluation articles"
                />
            </div>

            {/* Input & Classification Panel */}
            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs">
                <div className="flex items-center justify-between gap-2 mb-3">
                    <label
                        htmlFor="news-article-textarea"
                        className="text-xs font-semibold uppercase text-slate-600"
                    >
                        News Article or Headline
                    </label>

                    {/* Sample Article Quick-Fill Pills */}
                    <div className="flex items-center gap-1.5 overflow-x-auto">
                        <span className="text-[11px] text-slate-400 mr-1 hidden sm:inline">Samples:</span>
                        {SAMPLE_ARTICLES.map((sample, idx) => (
                            <button
                                key={idx}
                                type="button"
                                onClick={() => {
                                    setCurrentArticle(sample.text);
                                    setCurrentAnalysis(null);
                                    setError(null);
                                }}
                                disabled={loading}
                                className="rounded-lg bg-slate-100 px-2.5 py-1 text-[11px] font-medium text-slate-700 hover:bg-slate-200 disabled:opacity-50"
                            >
                                {sample.title}
                            </button>
                        ))}
                    </div>
                </div>

                <textarea
                    id="news-article-textarea"
                    value={currentArticle}
                    onChange={(e) => {
                        setCurrentArticle(e.target.value);
                        setCurrentAnalysis(null);
                    }}
                    placeholder="Type or paste a news headline or full article text here..."
                    rows={5}
                    disabled={loading}
                    className="w-full resize-y rounded-xl border border-slate-300 p-4 text-xs sm:text-sm text-slate-800 outline-none focus:border-blue-500 font-sans leading-relaxed"
                />

                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
                    <div className="flex items-center gap-3 text-xs text-slate-500">
                        <span>
                            {currentArticle.trim().split(/\s+/).filter(Boolean).length} words
                        </span>
                        <span>&bull;</span>
                        <span>{currentArticle.length} characters</span>
                    </div>

                    <div className="flex items-center gap-2">
                        {currentArticle && (
                            <button
                                type="button"
                                onClick={handleClear}
                                disabled={loading}
                                className="inline-flex items-center gap-1 rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                            >
                                <RotateCcw size={13} /> Clear
                            </button>
                        )}
                        <button
                            type="button"
                            onClick={handleClassify}
                            disabled={!currentArticle.trim() || loading}
                            className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-2.5 text-xs font-semibold text-white shadow-xs hover:bg-blue-700 disabled:opacity-50"
                        >
                            <Sparkles size={16} />
                            {loading ? "Classifying News..." : "Classify Article"}
                        </button>
                    </div>
                </div>
            </section>

            {/* Error Notification */}
            {error && (
                <div className="mt-6">
                    <ErrorState
                        error={error}
                        title="News Classification Failed"
                        onRetry={handleClassify}
                    />
                </div>
            )}

            {/* Loading Indicator */}
            {loading && (
                <div className="mt-6">
                    <LoadingState
                        title="Classifying Article & Extracting Features..."
                        description="Computing TF-IDF vectorization and multiclass Logistic Regression inference."
                    />
                </div>
            )}

            {/* Classification Results */}
            {classification && !loading && (
                <div className="mt-8 space-y-6">
                    {/* Primary Prediction Cards */}
                    <div className="grid gap-4 sm:grid-cols-2">
                        <ResultCard
                            title="Predicted Category"
                            value={classification.category}
                            subtitle="Assigned AG News domain"
                            badge="Class Output"
                            badgeType="info"
                            valueColor="text-blue-600"
                        />

                        <ResultCard
                            title="Model Confidence"
                            value={`${(classification.confidence * 100).toFixed(2)}%`}
                            subtitle="Estimated posterior class probability"
                            badge={
                                classification.confidence > 0.75
                                    ? "High Confidence"
                                    : "Moderate Confidence"
                            }
                            badgeType={
                                classification.confidence > 0.75
                                    ? "success"
                                    : "warning"
                            }
                        />
                    </div>

                    {/* Class Probability Distribution */}
                    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-xs">
                        <h3 className="text-xs font-semibold uppercase text-slate-500 mb-4">
                            Category Probability Breakdown
                        </h3>

                        <div className="space-y-4">
                            {probabilities.map((item) => {
                                const isTop = item.category === classification.category;
                                return (
                                    <div key={item.category}>
                                        <div className="mb-1.5 flex justify-between text-xs font-medium">
                                            <span className={isTop ? "font-bold text-slate-900" : "text-slate-600"}>
                                                {item.category}
                                            </span>
                                            <span className={isTop ? "font-bold text-blue-600 font-mono" : "text-slate-500 font-mono"}>
                                                {item.probability.toFixed(2)}%
                                            </span>
                                        </div>
                                        <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                                            <div
                                                className={`h-full rounded-full transition-all duration-300 ${
                                                    isTop ? "bg-blue-600" : "bg-slate-300"
                                                }`}
                                                style={{ width: `${Math.max(item.probability, 1)}%` }}
                                            />
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Feature Visualizations (TF-IDF and BoW) */}
                    {currentAnalysis?.tfidf?.features && (
                        <div className="grid gap-6 lg:grid-cols-2">
                            <FeatureChart
                                title="Top TF-IDF Terms"
                                data={currentAnalysis.tfidf.features}
                            />

                            {currentAnalysis?.bow?.features && (
                                <FeatureChart
                                    title="Top Bag of Words Frequencies"
                                    data={currentAnalysis.bow.features}
                                    valueKey="count"
                                />
                            )}
                        </div>
                    )}

                    {/* Deep-dive Navigation Button */}
                    <div className="flex justify-end">
                        <button
                            type="button"
                            onClick={() => navigate("/features")}
                            className="inline-flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-2.5 text-xs font-semibold text-white shadow-xs hover:bg-slate-800"
                        >
                            Explore All NLP Features
                            <ArrowRight size={15} />
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
}
