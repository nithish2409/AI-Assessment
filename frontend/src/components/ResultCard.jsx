export default function ResultCard({
    title,
    value,
    subtitle,
    badge,
    badgeType = "neutral",
    icon: Icon,
    valueColor = "text-slate-900",
    className = "",
}) {
    const badgeStyles = {
        neutral: "bg-slate-100 text-slate-700 border-slate-200",
        success: "bg-emerald-50 text-emerald-700 border-emerald-200",
        warning: "bg-amber-50 text-amber-700 border-amber-200",
        danger: "bg-rose-50 text-rose-700 border-rose-200",
        info: "bg-blue-50 text-blue-700 border-blue-200",
    };

    return (
        <div
            className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-xs transition hover:shadow-sm ${className}`}
        >
            <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    {title}
                </span>
                <div className="flex items-center gap-2">
                    {badge && (
                        <span
                            className={`rounded-full border px-2 py-0.5 text-xs font-semibold ${
                                badgeStyles[badgeType] || badgeStyles.neutral
                            }`}
                        >
                            {badge}
                        </span>
                    )}
                    {Icon && <Icon size={18} className="text-slate-400" />}
                </div>
            </div>

            <p className={`mt-3 text-2xl font-bold tracking-tight ${valueColor}`}>
                {value}
            </p>

            {subtitle && (
                <p className="mt-2 text-xs text-slate-500 leading-relaxed">
                    {subtitle}
                </p>
            )}
        </div>
    );
}
