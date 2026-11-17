export default function PageHeader({
    title,
    description,
    badge,
    icon: Icon,
    actions,
}) {
    return (
        <header className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
            <div>
                <div className="flex flex-wrap items-center gap-3">
                    {Icon && (
                        <div className="flex items-center justify-center rounded-xl bg-blue-50 p-2 text-blue-600 border border-blue-100">
                            <Icon size={22} />
                        </div>
                    )}
                    <h1 className="text-2xl font-bold tracking-tight text-slate-900 md:text-3xl">
                        {title}
                    </h1>
                    {badge && (
                        <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 border border-slate-200">
                            {badge}
                        </span>
                    )}
                </div>
                {description && (
                    <p className="mt-2 text-sm text-slate-600 max-w-3xl">
                        {description}
                    </p>
                )}
            </div>
            {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
        </header>
    );
}
