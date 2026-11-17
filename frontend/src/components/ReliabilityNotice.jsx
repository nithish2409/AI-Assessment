import { ShieldAlert, AlertTriangle, Info, CheckCircle2 } from "lucide-react";

export default function ReliabilityNotice({
    type = "info", // "info" | "warning" | "caution" | "progression"
    title,
    children,
    progressionData = null,
    className = "",
}) {
    if (progressionData) {
        const {
            progression_label,
            uncertainty_level = "low",
            interpretation_summary,
            calibrated_noise_floor_pp,
            is_above_noise_floor,
            coverage_change_percentage_points,
        } = progressionData;

        const uncertaintyColors = {
            low: "bg-emerald-100 text-emerald-800 border-emerald-300",
            medium: "bg-amber-100 text-amber-800 border-amber-300",
            high: "bg-rose-100 text-rose-800 border-rose-300",
        };

        const isAbove = Boolean(is_above_noise_floor);
        const noiseFloorDisplay =
            calibrated_noise_floor_pp !== undefined && calibrated_noise_floor_pp !== null
                ? `${calibrated_noise_floor_pp} pp`
                : "0.89 pp (provisional)";

        return (
            <div
                className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-xs ${className}`}
            >
                <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
                    <div className="flex items-center gap-2">
                        <ShieldAlert size={18} className="text-slate-600" />
                        <h4 className="text-sm font-semibold text-slate-900">
                            Reliability & Progression Interpretation
                        </h4>
                    </div>

                    <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-500 font-medium">
                            Uncertainty Level:
                        </span>
                        <span
                            className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold capitalize ${
                                uncertaintyColors[uncertainty_level] || uncertaintyColors.low
                            }`}
                        >
                            {uncertainty_level} uncertainty
                        </span>
                    </div>
                </div>

                <div className="mt-4 grid gap-4 sm:grid-cols-2">
                    <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-200/80">
                        <span className="text-xs font-semibold uppercase text-slate-500">
                            Progression Classification
                        </span>
                        <p className="mt-1 text-sm font-bold text-slate-900">
                            {progression_label || "No substantial predicted change detected"}
                        </p>
                        <p className="mt-1 text-[11px] text-slate-500">
                            Categorical label assigned by the backend crack change engine.
                        </p>
                    </div>

                    <div className="rounded-xl bg-slate-50 p-3.5 border border-slate-200/80">
                        <span className="text-xs font-semibold uppercase text-slate-500">
                            Noise Floor Filter
                        </span>
                        <div className="mt-1 flex items-center gap-2">
                            <span
                                className={`inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold ${
                                    isAbove
                                        ? "bg-amber-100 text-amber-800"
                                        : "bg-slate-200 text-slate-700"
                                }`}
                            >
                                {isAbove
                                    ? "Above Noise Floor"
                                    : "Within Expected Noise"}
                            </span>
                            <span className="text-xs text-slate-600 font-mono">
                                ({noiseFloorDisplay})
                            </span>
                        </div>
                        <p className="mt-1 text-[11px] text-slate-500">
                            Observed change:{" "}
                            <span className="font-semibold text-slate-800">
                                {coverage_change_percentage_points !== undefined
                                    ? `${coverage_change_percentage_points >= 0 ? "+" : ""}${coverage_change_percentage_points.toFixed(2)} pp`
                                    : "N/A"}
                            </span>
                        </p>
                    </div>
                </div>

                {interpretation_summary && (
                    <div className="mt-4 rounded-xl bg-blue-50/60 p-3.5 text-xs text-slate-700 border border-blue-100">
                        <div className="flex items-start gap-2">
                            <Info size={16} className="mt-0.5 shrink-0 text-blue-600" />
                            <div>
                                <span className="font-semibold text-blue-900">
                                    Backend Interpretation Summary:
                                </span>
                                <p className="mt-0.5 text-slate-700 leading-relaxed">
                                    {interpretation_summary}
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                <div className="mt-4 text-[11px] text-slate-500 leading-relaxed space-y-1 border-t border-slate-100 pt-3">
                    <p>
                        <strong className="text-slate-700">Important Methodological Note:</strong> The 0.89 pp noise floor is a provisional empirical threshold derived from controlled image experiments. It helps differentiate plausible crack progression from segmentation and alignment variability.
                    </p>
                    <p className="text-amber-800 font-medium">
                        Disclaimers: The system does not guarantee zero false positives, does not provide physical crack widths in millimeters, and does not assess overall structural safety. Results represent potential predicted crack regions only.
                    </p>
                </div>
            </div>
        );
    }

    const configs = {
        info: {
            border: "border-blue-200 bg-blue-50/70 text-blue-900",
            icon: Info,
            iconColor: "text-blue-600",
        },
        warning: {
            border: "border-amber-200 bg-amber-50/70 text-amber-900",
            icon: AlertTriangle,
            iconColor: "text-amber-600",
        },
        caution: {
            border: "border-rose-200 bg-rose-50/70 text-rose-900",
            icon: ShieldAlert,
            iconColor: "text-rose-600",
        },
    };

    const cfg = configs[type] || configs.info;
    const Icon = cfg.icon;

    return (
        <aside
            role="note"
            className={`rounded-2xl border p-4 text-xs leading-relaxed shadow-xs ${cfg.border} ${className}`}
        >
            <div className="flex items-start gap-3">
                <div className={`mt-0.5 shrink-0 ${cfg.iconColor}`}>
                    <Icon size={18} />
                </div>
                <div className="flex-1">
                    {title && (
                        <h4 className="mb-1 font-semibold text-slate-900">
                            {title}
                        </h4>
                    )}
                    <div className="text-slate-700">{children}</div>
                </div>
            </div>
        </aside>
    );
}
