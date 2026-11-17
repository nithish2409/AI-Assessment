
import {
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid,
} from "recharts";

export default function FeatureChart({
    title,
    data = [],
    nameKey = "term",
    valueKey = "score",
}) {
    return (
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="mb-5 font-semibold text-slate-900">{title}</h3>

            {!data.length ? (
                <p className="text-sm text-slate-500">
                    No features found for this input.
                </p>
            ) : (
                <div className="h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                            data={data}
                            layout="vertical"
                            margin={{ left: 15, right: 20 }}
                        >
                            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                            <XAxis type="number" />
                            <YAxis
                                type="category"
                                dataKey={nameKey}
                                width={115}
                                tick={{ fontSize: 11 }}
                            />
                            <Tooltip />
                            <Bar
                                dataKey={valueKey}
                                fill="#2563eb"
                                radius={[0, 5, 5, 0]}
                            />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            )}
        </div>
    );
}
