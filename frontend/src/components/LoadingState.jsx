import { Loader2, XCircle } from "lucide-react";

export default function LoadingState({
    title = "Processing...",
    description = "Please wait while the inference engine completes computation.",
    onCancel,
    className = "",
}) {
    return (
        <div
            role="status"
            aria-live="polite"
            className={`flex flex-col items-center justify-center rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-xs ${className}`}
        >
            <Loader2 className="h-9 w-9 animate-spin text-blue-600" />
            <h3 className="mt-4 text-base font-semibold text-slate-900">{title}</h3>
            {description && (
                <p className="mt-1.5 max-w-md text-xs text-slate-500 leading-relaxed">
                    {description}
                </p>
            )}

            {onCancel && (
                <button
                    type="button"
                    onClick={onCancel}
                    className="mt-4 flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50"
                >
                    <XCircle size={14} />
                    Cancel request
                </button>
            )}
        </div>
    );
}
