
import { useState } from "react";
import {
    GitCompareArrows,
    RotateCcw,
} from "lucide-react";

import PageHeader from "../components/PageHeader";
import {
    compareTexts,
    getErrorMessage,
} from "../services/api";

export default function TextSimilarity() {
    const [text1, setText1] = useState("");
    const [text2, setText2] = useState("");
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const handleCompare = async () => {
        if (!text1.trim() || !text2.trim()) {
            setError("Please enter both texts.");
            return;
        }

        setLoading(true);
        setError("");
        setResult(null);

        try {
            const response = await compareTexts(text1, text2);
            setResult(response);
        } catch (err) {
            setError(getErrorMessage(err));
        } finally {
            setLoading(false);
        }
    };

    const reset = () => {
        setText1("");
        setText2("");
        setResult(null);
        setError("");
    };

    const percentage = result?.similarity_percentage ?? 0;

    return (
        <div>
            <PageHeader
                title="Text Similarity Analysis"
                description="Compare two news articles using TF-IDF vectorization and cosine similarity."
            />

            <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="grid gap-5 lg:grid-cols-2">
                    <div>
                        <label className="mb-3 block font-semibold">
                            Article 1
                        </label>
                        <textarea
                            value={text1}
                            onChange={(e) => {
                                setText1(e.target.value);
                                setResult(null);
                            }}
                            placeholder="Enter the first news article..."
                            className="min-h-56 w-full rounded-xl border border-slate-300 p-4 text-sm outline-none focus:border-blue-500"
                        />
                    </div>

                    <div>
                        <label className="mb-3 block font-semibold">
                            Article 2
                        </label>
                        <textarea
                            value={text2}
                            onChange={(e) => {
                                setText2(e.target.value);
                                setResult(null);
                            }}
                            placeholder="Enter the second news article..."
                            className="min-h-56 w-full rounded-xl border border-slate-300 p-4 text-sm outline-none focus:border-blue-500"
                        />
                    </div>
                </div>

                <div className="mt-5 flex flex-wrap gap-3">
                    <button
                        onClick={handleCompare}
                        disabled={loading}
                        className="flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-3 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-60"
                    >
                        <GitCompareArrows size={18} />
                        {loading ? "Comparing..." : "Compare Articles"}
                    </button>

                    <button
                        onClick={reset}
                        className="flex items-center gap-2 rounded-xl border border-slate-300 px-5 py-3 text-sm hover:bg-slate-50"
                    >
                        <RotateCcw size={17} />
                        Reset
                    </button>
                </div>

                {error && (
                    <p role="alert" className="mt-4 text-sm text-red-600">
                        {error}
                    </p>
                )}
            </section>

            {result && (
                <section className="mt-6 rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
                    <h2 className="text-lg font-semibold">
                        Similarity Result
                    </h2>

                    <div className="mx-auto mt-6 flex h-44 w-44 items-center justify-center rounded-full border-[12px] border-blue-500">
                        <div>
                            <p className="text-4xl font-bold text-slate-900">
                                {percentage.toFixed(1)}%
                            </p>
                            <p className="mt-1 text-xs text-slate-500">
                                Cosine similarity
                            </p>
                        </div>
                    </div>

                    <div className="mx-auto mt-7 max-w-lg">
                        <div className="h-4 overflow-hidden rounded-full bg-slate-100">
                            <div
                                className="h-full rounded-full bg-blue-600"
                                style={{ width: `${percentage}%` }}
                            />
                        </div>
                    </div>

                    <p className="mt-5 text-sm text-slate-600">
                        Similarity score: {result.similarity_score.toFixed(4)}
                    </p>

                    <p className="mx-auto mt-3 max-w-xl text-xs text-slate-500">
                        This score measures similarity in the trained TF-IDF
                        feature space. It is not a semantic similarity
                        probability or a plagiarism verdict.
                    </p>
                </section>
            )}
        </div>
    );
}
