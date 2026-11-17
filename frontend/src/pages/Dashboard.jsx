
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
    BrainCircuit,
    ArrowRight,
    Sparkles,
} from "lucide-react";

import PageHeader from "../components/PageHeader";
import ResultCard from "../components/ResultCard";
import FeatureChart from "../components/FeatureChart";


import {
    analyzeNews,
    getErrorMessage,
    getModelMetrics,
} from "../services/api";

export default function Dashboard({
    article,
    setArticle,
    analysis,
    setAnalysis,
    beginRequest,
    isCurrentRequest,
}) {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const navigate = useNavigate();

    const [metrics, setMetrics] = useState(null);

    useEffect(() => {
        getModelMetrics()
            .then(setMetrics)
            .catch((error) => {
                console.error("Metrics unavailable:", error);
            });
    }, []);




    const handleAnalyze = async () => {
        if (!article.trim()) {
            setError("Please enter a news article.");
            return;
        }

        const version = beginRequest();

        setLoading(true);
        setError("");
        setAnalysis(null);

        try {
            const result = await analyzeNews(article, 10);

            if (isCurrentRequest(version)) {
                setAnalysis(result);
            }
        } catch (err) {
            if (isCurrentRequest(version)) {
                setError(getErrorMessage(err));
            }
        } finally {
            if (isCurrentRequest(version)) {
                setLoading(false);
            }
        }
    };



    const classification = analysis?.classification;

    const probabilities = classification
        ? Object.entries(classification.probabilities)
            .map(([category, probability]) => ({
                category,
                probability: probability * 100,
            }))
            .sort((a, b) => b.probability - a.probability)
        : [];

    return (
        <div>
            <PageHeader
                title="News Classification Dashboard"
                description="Classify news articles and explore their NLP features using machine learning."
            />


            <div className="mb-6 grid gap-4 md:grid-cols-3">
                <ResultCard
                    title="Model Test Accuracy"
                    value={
                        metrics
                            ? `${(metrics.accuracy * 100).toFixed(2)}%`
                            : "Unavailable"
                    }
                    subtitle="AG News test dataset"
                />

                <ResultCard
                    title="Macro F1-Score"
                    value={
                        metrics
                            ? `${(metrics.macro_f1 * 100).toFixed(2)}%`
                            : "Unavailable"
                    }
                    subtitle="Four-class evaluation"
                />

                <ResultCard
                    title="Test Samples"
                    value={
                        metrics
                            ? metrics.testing_samples.toLocaleString()
                            : "Unavailable"
                    }
                    subtitle="Held-out evaluation dataset"
                />
            </div>


            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="mb-4 flex items-center gap-2">
                    <BrainCircuit className="text-blue-600" />
                    <h2 className="text-lg font-semibold">
                        Analyze News Article
                    </h2>
                </div>

                <textarea
                    value={article}
                    onChange={(e) => {
                        setArticle(e.target.value);
                        setAnalysis(null);
                    }}
                    placeholder="Paste a news headline or article here..."
                    className="min-h-48 w-full resize-y rounded-xl border border-slate-300 p-4 text-sm outline-none focus:border-blue-500"
                />

                <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
                    <p className="text-xs text-slate-500">
                        {article.trim().split(/\s+/).filter(Boolean).length} words
                    </p>

                    <button
                        onClick={handleAnalyze}
                        disabled={loading}
                        className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
                    >
                        <Sparkles size={17} />
                        {loading ? "Analyzing..." : "Analyze News"}
                    </button>
                </div>

                {error && (
                    <p role="alert" className="mt-4 text-sm text-red-600">
                        {error}
                    </p>
                )}
            </section>

            {classification && (
                <div className="mt-6 space-y-6">
                    <div className="grid gap-4 md:grid-cols-2">
                        <ResultCard
                            title="Predicted Category"
                            value={classification.category}
                            subtitle="Logistic Regression classifier"
                        />

                        <ResultCard
                            title="Model Confidence"
                            value={`${(classification.confidence * 100).toFixed(2)}%`}
                            subtitle="Estimated class probability"
                        />
                    </div>

                    <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="mb-5 font-semibold">
                            Category Probabilities
                        </h3>

                        <div className="space-y-5">
                            {probabilities.map((item) => (
                                <div key={item.category}>
                                    <div className="mb-2 flex justify-between text-sm">
                                        <span>{item.category}</span>
                                        <span className="font-medium">
                                            {item.probability.toFixed(2)}%
                                        </span>
                                    </div>

                                    <div className="h-3 overflow-hidden rounded-full bg-slate-100">
                                        <div
                                            className="h-full rounded-full bg-blue-600"
                                            style={{
                                                width: `${item.probability}%`,
                                            }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </section>

                    <div className="grid gap-6 lg:grid-cols-2">
                        <FeatureChart
                            title="Top TF-IDF Terms"
                            data={analysis.tfidf.features}
                        />

                        <FeatureChart
                            title="Bag of Words"
                            data={analysis.bow.features}
                            valueKey="count"
                        />
                    </div>

                    <button
                        onClick={() => navigate("/features")}
                        className="flex items-center gap-2 rounded-xl bg-slate-900 px-5 py-3 text-sm font-medium text-white hover:bg-slate-800"
                    >
                        Explore All NLP Features
                        <ArrowRight size={17} />
                    </button>
                </div>
            )}
        </div>
    );
}
