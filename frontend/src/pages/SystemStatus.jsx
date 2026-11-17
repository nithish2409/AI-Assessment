import { useState, useEffect } from "react";
import {
    Activity,
    Server,
    Cpu,
    CheckCircle2,
    XCircle,
    AlertTriangle,
    RefreshCw,
    Clock,
    ShieldAlert,
    Layers,
    FileText,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ResultCard from "../components/ResultCard";
import {
    checkHealth,
    checkDamageHealth,
    checkNlpStatus,
    getModelMetrics,
} from "../services/api";

export default function SystemStatus() {
    const [loading, setLoading] = useState(false);
    const [lastChecked, setLastChecked] = useState(null);

    // Main backend
    const [mainStatus, setMainStatus] = useState(null);
    const [mainLatency, setMainLatency] = useState(null);

    // YOLO service (via backend /api/damage/health)
    const [damageStatus, setDamageStatus] = useState(null);
    const [damageLatency, setDamageLatency] = useState(null);

    // NLP service
    const [nlpStatus, setNlpStatus] = useState(null);
    const [nlpLatency, setNlpLatency] = useState(null);

    const runDiagnostics = async () => {
        setLoading(true);

        // 1. Main FastAPI health
        const t0 = performance.now();
        try {
            const data = await checkHealth();
            setMainLatency(Math.round(performance.now() - t0));
            setMainStatus({ ok: true, data });
        } catch (err) {
            setMainLatency(Math.round(performance.now() - t0));
            setMainStatus({ ok: false, error: err.message });
        }

        // 2. YOLO service via main backend /api/damage/health
        const t1 = performance.now();
        try {
            const data = await checkDamageHealth();
            setDamageLatency(Math.round(performance.now() - t1));
            setDamageStatus({ ok: data?.is_available === true, data });
        } catch (err) {
            setDamageLatency(Math.round(performance.now() - t1));
            setDamageStatus({ ok: false, error: err.message });
        }

        // 3. NLP status
        const t2 = performance.now();
        try {
            const data = await checkNlpStatus();
            setNlpLatency(Math.round(performance.now() - t2));
            setNlpStatus({ ok: data?.status === "ready", data });
        } catch (err) {
            setNlpLatency(Math.round(performance.now() - t2));
            setNlpStatus({ ok: false, error: err.message });
        }

        setLastChecked(new Date().toLocaleTimeString());
        setLoading(false);
    };

    useEffect(() => {
        runDiagnostics();
    }, []);

    const allOnline = mainStatus?.ok && damageStatus?.ok && nlpStatus?.ok;

    return (
        <div>
            <PageHeader
                title="System Status & Telemetry"
                description="Live availability, hardware acceleration, and model readiness status across all AI/ML subsystems."
                badge="Diagnostics"
                icon={Activity}
                actions={
                    <button
                        type="button"
                        onClick={runDiagnostics}
                        disabled={loading}
                        className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-50 disabled:opacity-50"
                    >
                        <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
                        Refresh Status
                    </button>
                }
            />

            {/* Overall Health Summary Banner */}
            <div
                className={`mb-6 rounded-2xl border p-5 shadow-xs ${
                    allOnline
                        ? "border-emerald-200 bg-emerald-50/70 text-emerald-950"
                        : mainStatus?.ok
                        ? "border-amber-200 bg-amber-50/70 text-amber-950"
                        : "border-rose-200 bg-rose-50/70 text-rose-950"
                }`}
            >
                <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                        {allOnline ? (
                            <CheckCircle2 size={24} className="text-emerald-600" />
                        ) : (
                            <AlertTriangle size={24} className="text-amber-600" />
                        )}
                        <div>
                            <h2 className="text-sm font-bold">
                                {allOnline
                                    ? "All Subsystems Fully Operational"
                                    : mainStatus?.ok
                                    ? "Main Backend Operational with Degraded Peripheral Services"
                                    : "Main Backend Offline or Unreachable"}
                            </h2>
                            <p className="mt-0.5 text-xs text-slate-600">
                                Last verified at: {lastChecked || "Checking..."}
                            </p>
                        </div>
                    </div>

                    <span
                        className={`rounded-full border px-3 py-1 text-xs font-semibold ${
                            allOnline
                                ? "bg-emerald-100 text-emerald-800 border-emerald-300"
                                : "bg-amber-100 text-amber-800 border-amber-300"
                        }`}
                    >
                        {allOnline ? "HEALTHY" : "ATTENTION REQUIRED"}
                    </span>
                </div>
            </div>

            {/* Detailed Service Cards */}
            <div className="grid gap-6 md:grid-cols-3">
                {/* 1. Main FastAPI Backend */}
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs flex flex-col justify-between">
                    <div>
                        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                            <div className="flex items-center gap-2">
                                <Server size={18} className="text-blue-600" />
                                <h3 className="text-sm font-bold text-slate-900">
                                    Main Backend
                                </h3>
                            </div>
                            {mainStatus?.ok ? (
                                <span className="flex items-center gap-1 text-xs font-semibold text-emerald-600">
                                    <CheckCircle2 size={15} /> Online
                                </span>
                            ) : (
                                <span className="flex items-center gap-1 text-xs font-semibold text-rose-600">
                                    <XCircle size={15} /> Offline
                                </span>
                            )}
                        </div>

                        <div className="mt-4 space-y-2.5 text-xs text-slate-600">
                            <div className="flex justify-between">
                                <span className="text-slate-500">Service:</span>
                                <span className="font-semibold text-slate-800">
                                    FastAPI (REST API)
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Port:</span>
                                <span className="font-mono text-slate-800">8000</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Health Probe:</span>
                                <span className="font-mono text-slate-700">GET /api/health</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Probe Latency:</span>
                                <span className="font-mono text-slate-800">
                                    {mainLatency !== null ? `${mainLatency} ms` : "..."}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="mt-4 border-t border-slate-100 pt-3 text-[11px] text-slate-400">
                        Hosts Task 2 Image Comparison and Task 3 NLP APIs.
                    </div>
                </div>

                {/* 2. YOLO Inference Microservice */}
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs flex flex-col justify-between">
                    <div>
                        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                            <div className="flex items-center gap-2">
                                <Cpu size={18} className="text-blue-600" />
                                <h3 className="text-sm font-bold text-slate-900">
                                    YOLO Microservice
                                </h3>
                            </div>
                            {damageStatus?.ok ? (
                                <span className="flex items-center gap-1 text-xs font-semibold text-emerald-600">
                                    <CheckCircle2 size={15} /> Ready
                                </span>
                            ) : (
                                <span className="flex items-center gap-1 text-xs font-semibold text-amber-600">
                                    <AlertTriangle size={15} /> Unavailable
                                </span>
                            )}
                        </div>

                        <div className="mt-4 space-y-2.5 text-xs text-slate-600">
                            <div className="flex justify-between">
                                <span className="text-slate-500">Model:</span>
                                <span className="font-semibold text-slate-800">
                                    YOLO26s-seg
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Access Path:</span>
                                <span className="text-slate-700">Via Main Backend Gateway</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Damage Health Probe:</span>
                                <span className="font-mono text-slate-700">GET /api/damage/health</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Gateway Latency:</span>
                                <span className="font-mono text-slate-800">
                                    {damageLatency !== null ? `${damageLatency} ms` : "..."}
                                </span>
                            </div>
                            {damageStatus?.data?.details?.device && (
                                <div className="mt-2 rounded-lg bg-slate-50 p-2 text-[11px] text-slate-600 border border-slate-200/80">
                                    <span className="font-semibold block text-slate-700">Compute Hardware:</span>
                                    <span className="font-mono truncate block">
                                        {damageStatus.data.details.device}
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="mt-4 border-t border-slate-100 pt-3 text-[11px] text-slate-400">
                        Never called directly from the browser to maintain architecture isolation.
                    </div>
                </div>

                {/* 3. NLP News Intelligence */}
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs flex flex-col justify-between">
                    <div>
                        <div className="flex items-center justify-between border-b border-slate-100 pb-3">
                            <div className="flex items-center gap-2">
                                <FileText size={18} className="text-blue-600" />
                                <h3 className="text-sm font-bold text-slate-900">
                                    NLP Classifier
                                </h3>
                            </div>
                            {nlpStatus?.ok ? (
                                <span className="flex items-center gap-1 text-xs font-semibold text-emerald-600">
                                    <CheckCircle2 size={15} /> Ready
                                </span>
                            ) : (
                                <span className="flex items-center gap-1 text-xs font-semibold text-amber-600">
                                    <AlertTriangle size={15} /> Pending
                                </span>
                            )}
                        </div>

                        <div className="mt-4 space-y-2.5 text-xs text-slate-600">
                            <div className="flex justify-between">
                                <span className="text-slate-500">Pipeline:</span>
                                <span className="font-semibold text-slate-800">
                                    TF-IDF &bull; Logistic Reg.
                                </span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Dataset Domain:</span>
                                <span className="text-slate-700">AG News (4 Categories)</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Status Probe:</span>
                                <span className="font-mono text-slate-700">GET /api/nlp/status</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-slate-500">Probe Latency:</span>
                                <span className="font-mono text-slate-800">
                                    {nlpLatency !== null ? `${nlpLatency} ms` : "..."}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="mt-4 border-t border-slate-100 pt-3 text-[11px] text-slate-400">
                        Powers category prediction, BoW, and cosine similarity.
                    </div>
                </div>
            </div>

            {/* Architecture Isolation Documentation */}
            <div className="mt-8 rounded-2xl border border-slate-200 bg-white p-6 shadow-xs">
                <div className="flex items-center gap-2 mb-3">
                    <ShieldAlert size={18} className="text-blue-600" />
                    <h3 className="text-sm font-semibold text-slate-900">
                        Architectural Isolation Protocol
                    </h3>
                </div>
                <p className="text-xs text-slate-600 leading-relaxed">
                    The frontend client interacts strictly with the main FastAPI gateway (port 8000). The dedicated YOLO GPU inference microservice (port 8001) is isolated on an internal network loopback and accessed solely via async proxy calls from the main backend. Internal service addresses and raw hardware credentials are never exposed directly to the browser.
                </p>
            </div>
        </div>
    );
}
