import { AlertTriangle, AlertCircle, RefreshCw, Info } from "lucide-react";
import { parseApiError } from "../services/api";

export default function ErrorState({
    error,
    onRetry,
    title: fallbackTitle,
    className = "",
}) {
    if (!error) return null;

    const parsed =
        typeof error === "string"
            ? {
                  title: fallbackTitle || "Operation Failed",
                  message: error,
                  code: null,
                  guidance: null,
                  failureReasons: [],
                  attemptedMethods: [],
              }
            : parseApiError(error);

    const isWarning =
        parsed.code === "ALIGNMENT_FAILED" ||
        parsed.code === "INSUFFICIENT_OVERLAP" ||
        parsed.code === "INVALID_ALIGNMENT_MODE";

    const Icon = isWarning ? AlertTriangle : AlertCircle;
    const borderStyle = isWarning
        ? "border-amber-300 bg-amber-50/70 text-amber-900"
        : "border-rose-300 bg-rose-50/70 text-rose-900";
    const iconColor = isWarning ? "text-amber-600" : "text-rose-600";
    const badgeStyle = isWarning
        ? "bg-amber-100 text-amber-800 border-amber-200"
        : "bg-rose-100 text-rose-800 border-rose-200";

    const displayTitle =
        parsed.code === "ALIGNMENT_FAILED"
            ? "Image Alignment Failed"
            : parsed.title && parsed.title !== "Operation Failed" && parsed.title !== "Error"
            ? parsed.title
            : fallbackTitle || parsed.title || "Operation Failed";

    return (
        <div
            role="alert"
            className={`rounded-2xl border p-5 shadow-xs ${borderStyle} ${className}`}
        >
            <div className="flex items-start gap-3">
                <div className={`mt-0.5 rounded-lg p-1.5 ${iconColor}`}>
                    <Icon size={20} />
                </div>
                <div className="flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <h3 className="text-sm font-semibold">
                            {displayTitle}
                        </h3>
                        {parsed.code && (
                            <span
                                className={`rounded-md border px-2 py-0.5 text-[11px] font-mono font-semibold uppercase ${badgeStyle}`}
                            >
                                {parsed.code}
                            </span>
                        )}
                    </div>

                    <p className="mt-1.5 text-xs text-slate-700 leading-relaxed font-medium">
                        {parsed.message}
                    </p>

                    {parsed.attemptedMethods && parsed.attemptedMethods.length > 0 && (
                        <div className="mt-2.5 text-xs text-slate-700">
                            <span className="font-semibold text-slate-900">
                                Attempted alignment methods:
                            </span>{" "}
                            <span className="font-mono text-slate-800">
                                {parsed.attemptedMethods.join(", ")}
                            </span>
                        </div>
                    )}

                    {parsed.failureReasons && parsed.failureReasons.length > 0 && (
                        <div className="mt-2 text-xs text-slate-700">
                            <span className="font-semibold text-slate-900">
                                {parsed.failureReasons.length === 1 ? "Failure reason:" : "Failure reasons:"}
                            </span>
                            <ul className="mt-1 list-inside list-disc text-xs text-slate-700 space-y-0.5 pl-1">
                                {parsed.failureReasons.map((r, i) => (
                                    <li key={i}>{r}</li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {parsed.guidance && (
                        <div className="mt-3 flex items-start gap-2 rounded-xl bg-white/90 p-3 text-xs text-slate-700 border border-slate-200/90 shadow-2xs">
                            <Info size={15} className="mt-0.5 shrink-0 text-blue-600" />
                            <div>
                                <span className="font-semibold text-slate-900">
                                    Recommended action:
                                </span>{" "}
                                {parsed.guidance}
                            </div>
                        </div>
                    )}

                    {onRetry && (
                        <div className="mt-4">
                            <button
                                type="button"
                                onClick={onRetry}
                                className="inline-flex items-center gap-1.5 rounded-lg bg-slate-900 px-3.5 py-1.5 text-xs font-semibold text-white shadow-xs hover:bg-slate-800"
                            >
                                <RefreshCw size={13} /> Retry Operation
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
