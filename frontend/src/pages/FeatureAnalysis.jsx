
import { useState } from "react";
import PageHeader from "../components/PageHeader";
import FeatureChart from "../components/FeatureChart";

import {
    analyzeNews,
    extractNgrams,
    selectFeatures,
    getErrorMessage,
} from "../services/api";

const categories = [
    "World",
    "Sports",
    "Business",
    "Technology",
];

export default function FeatureAnalysis({
    article,
    setArticle,
    analysis,
    setAnalysis,
    beginRequest,
    isCurrentRequest,
}) {
    const [n, setN] = useState(2);
    const [category, setCategory] = useState("Technology");
    const [ngrams, setNgrams] = useState(null);
    const [features, setFeatures] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");


    const runAnalysis = async () => {
        if (!article.trim()) {
            setError("Please enter an article first.");
            return;
        }

        const version = beginRequest();

        setLoading(true);
        setError("");
        setAnalysis(null);
        setNgrams(null);
        setFeatures(null);

        try {
            const result = await analyzeNews(article, 10);

            const selectedCategory = result.classification.category;

            const [ngramResult, featureResult] = await Promise.all([
                extractNgrams(article, n, 10),
                selectFeatures(selectedCategory, 10),
            ]);

            if (!isCurrentRequest(version)) return;

            setAnalysis(result);
            setCategory(selectedCategory);
            setNgrams(ngramResult);
            setFeatures(featureResult);

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


    const updateNgrams = async (value) => {
        const next = Number(value);
        setN(next);

        if (!analysis) return;

        try {
            setError("");
            const result = await extractNgrams(article, next, 10);
            setNgrams(result);
        } catch (err) {
            setError(getErrorMessage(err));
        }
    };

    const updateCategory = async (value) => {
        setCategory(value);

        if (!analysis) return;

        try {
            setError("");
            const result = await selectFeatures(value, 10);
            setFeatures(result);
        } catch (err) {
            setError(getErrorMessage(err));
        }
    };

    const keywordData = analysis?.keywords?.keywords || [];

    return (
        <div>
            <PageHeader
                title="NLP Feature Analysis"
                description="Explore the statistical and model-based features extracted from news articles."
            />

            <section className="mb-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <label className="mb-3 block font-semibold">
                    News Article
                </label>

                <textarea
                    value={article}
                    onChange={(e) => {
                        setArticle(e.target.value);
                        setAnalysis(null);
                        setNgrams(null);
                        setFeatures(null);
                    }}
                    placeholder="Enter an article for NLP analysis..."
                    className="min-h-40 w-full rounded-xl border border-slate-300 p-4 text-sm outline-none focus:border-blue-500"
                />

                <button
                    onClick={runAnalysis}
                    disabled={loading}
                    className="mt-4 rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
                >
                    {loading ? "Processing..." : "Extract NLP Features"}
                </button>

                {error && (
                    <p role="alert" className="mt-4 text-sm text-red-600">
                        {error}
                    </p>
                )}
            </section>

            {analysis && (
                <div className="space-y-6">
                    <div className="grid gap-6 lg:grid-cols-2">
                        <FeatureChart
                            title="1. TF-IDF Vectorization"
                            data={analysis.tfidf.features}
                        />

                        <FeatureChart
                            title="2. Bag of Words"
                            data={analysis.bow.features}
                            valueKey="count"
                        />
                    </div>

                    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                            <h3 className="font-semibold">
                                3. N-gram Analysis
                            </h3>

                            <select
                                value={n}
                                onChange={(e) => updateNgrams(e.target.value)}
                                className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
                            >
                                <option value={1}>Unigrams</option>
                                <option value={2}>Bigrams</option>
                                <option value={3}>Trigrams</option>
                            </select>
                        </div>

                        <FeatureChart
                            title={`${n}-gram Frequencies`}
                            data={
                                ngrams?.features ||
                                (n === 2 ? analysis.ngrams.features : [])
                            }
                            nameKey="phrase"
                            valueKey="count"
                        />
                    </section>

                    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                        <h3 className="mb-4 font-semibold">
                            4. Keyword Extraction
                        </h3>

                        {keywordData.length ? (
                            <div className="flex flex-wrap gap-2">
                                {keywordData.map((item) => (
                                    <span
                                        key={item.keyword}
                                        title={`TF-IDF: ${item.score}`}
                                        className="rounded-full bg-blue-50 px-4 py-2 text-sm font-medium text-blue-700"
                                    >
                                        {item.keyword}
                                    </span>
                                ))}
                            </div>
                        ) : (
                            <p className="text-sm text-slate-500">
                                No keywords found.
                            </p>
                        )}
                    </section>

                    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
                        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                            <h3 className="font-semibold">
                                5. Model-Based Feature Selection
                            </h3>

                            <select
                                value={category}
                                onChange={(e) => updateCategory(e.target.value)}
                                className="rounded-lg border border-slate-300 px-3 py-2 text-sm"
                            >
                                {categories.map((item) => (
                                    <option key={item} value={item}>
                                        {item}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <FeatureChart
                            title={`Top Features for ${category}`}
                            data={
                                features?.features ||
                                (category === analysis.classification.category
                                    ? analysis.feature_selection.features
                                    : [])
                            }
                            valueKey="importance"
                        />
                    </section>
                </div>
            )}
        </div>
    );
}
