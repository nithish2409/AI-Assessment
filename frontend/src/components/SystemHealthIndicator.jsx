import { useEffect, useState } from "react";
import { Activity, Server, Cpu, CheckCircle2, AlertTriangle, XCircle, RefreshCw } from "lucide-react";
import { checkHealth, checkDamageHealth, checkNlpStatus } from "../services/api";

export default function SystemHealthIndicator({
    showDetails = false,
    onStatusChange = null,
    className = "",
}) {
    const [mainHealthy, setMainHealthy] = useState(null); // null | true | false
    const [yoloHealthy, setYoloHealthy] = useState(null);
    const [nlpHealthy, setNlpHealthy] = useState(null);
    const [yoloDetails, setYoloDetails] = useState(null);
    const [checking, setChecking] = useState(false);
    const [lastChecked, setLastChecked] = useState(null);

    const refreshHealth = async () => {
        setChecking(true);
        let mOk = false;
        let yOk = false;
        let nOk = false;
        let yDetails = null;

        try {
            const h = await checkHealth();
            mOk = h?.status === "healthy";
        } catch {
            mOk = false;
        }

        try {
            const d = await checkDamageHealth();
            yOk = d?.is_available === true;
            yDetails = d?.details || null;
        } catch {
            yOk = false;
        }

        try {
            const n = await checkNlpStatus();
            nOk = n?.status === "ready";
        } catch {
            nOk = false;
        }

        setMainHealthy(mOk);
        setYoloHealthy(yOk);
        setNlpHealthy(nOk);
        setYoloDetails(yDetails);
        setLastChecked(new Date().toLocaleTimeString());
        setChecking(false);

        if (onStatusChange) {
            onStatusChange({
                main: mOk,
                yolo: yOk,
                nlp: nOk,
                yoloDetails,
            });
        }
    };

    useEffect(() => {
        refreshHealth();
        const interval = setInterval(refreshHealth, 30000);
        return () => clearInterval(interval);
    }, []);

    const allHealthy = mainHealthy && yoloHealthy && nlpHealthy;
    const partialHealthy = mainHealthy && (!yoloHealthy || !nlpHealthy);

    if (!showDetails) {
        // Compact sidebar indicator
        return (
            <div className={`flex flex-col gap-2 ${className}`}>
                <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                        <Activity
                            size={14}
                            className={
                                allHealthy
                                    ? "text-emerald-400"
                                    : partialHealthy
                                    ? "text-amber-400"
                                    : "text-rose-400"
                            }
                        />
                        <span className="text-slate-300 font-medium">System Status</span>
                    </div>
                    <span
                        className={`inline-block h-2 w-2 rounded-full ${
                            allHealthy
                                ? "bg-emerald-400"
                                : partialHealthy
                                ? "bg-amber-400 animate-pulse"
                                : "bg-rose-500"
                        }`}
                    />
                </div>

                <div className="space-y-1 text-[11px] text-slate-400 pl-5">
                    <div className="flex items-center justify-between">
                        <span>Backend API (8000)</span>
                        <span className={mainHealthy ? "text-emerald-400" : "text-rose-400"}>
                            {mainHealthy === null ? "..." : mainHealthy ? "Online" : "Offline"}
                        </span>
                    </div>
                    <div className="flex items-center justify-between">
                        <span>YOLO Service (8001)</span>
                        <span className={yoloHealthy ? "text-emerald-400" : "text-amber-400"}>
                            {yoloHealthy === null ? "..." : yoloHealthy ? "Online" : "Offline"}
                        </span>
                    </div>
                </div>
            </div>
        );
    }

    // Full detailed diagnostic card (e.g. for System Status or Dashboard Overview)
    return (
        <div className={`rounded-2xl border border-slate-200 bg-white p-5 shadow-xs ${className}`}>
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
                <div className="flex items-center gap-2">
                    <Server size={18} className="text-blue-600" />
                    <h3 className="text-sm font-semibold text-slate-900">
                        System Health & Service Connectivity
                    </h3>
                </div>

                <div className="flex items-center gap-2 text-xs text-slate-500">
                    {lastChecked && <span>Checked: {lastChecked}</span>}
                    <button
                        type="button"
                        onClick={refreshHealth}
                        disabled={checking}
                        className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-100 disabled:opacity-50"
                    >
                        <RefreshCw size={12} className={checking ? "animate-spin" : ""} />
                        Refresh
                    </button>
                </div>
            </div>

            <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {/* Main FastAPI Backend */}
                <div className="rounded-xl border border-slate-200/80 bg-slate-50 p-3.5">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold uppercase text-slate-500">
                            Main Backend
                        </span>
                        {mainHealthy ? (
                            <CheckCircle2 size={16} className="text-emerald-600" />
                        ) : (
                            <XCircle size={16} className="text-rose-600" />
                        )}
                    </div>
                    <p className="mt-1 text-sm font-bold text-slate-900">
                        FastAPI Service
                    </p>
                    <p className="text-[11px] text-slate-500">
                        Port 8000 &bull; {mainHealthy ? "Operational" : "Unreachable"}
                    </p>
                </div>

                {/* YOLO Inference Microservice */}
                <div className="rounded-xl border border-slate-200/80 bg-slate-50 p-3.5">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold uppercase text-slate-500">
                            YOLO Microservice
                        </span>
                        {yoloHealthy ? (
                            <CheckCircle2 size={16} className="text-emerald-600" />
                        ) : (
                            <AlertTriangle size={16} className="text-amber-600" />
                        )}
                    </div>
                    <p className="mt-1 text-sm font-bold text-slate-900">
                        YOLO26s-seg GPU Engine
                    </p>
                    <p className="text-[11px] text-slate-500">
                        Via Backend Gateway &bull; {yoloHealthy ? "Ready" : "Unavailable"}
                    </p>
                    {yoloDetails?.device && (
                        <p className="mt-1 font-mono text-[10px] text-slate-600 truncate">
                            {yoloDetails.device}
                        </p>
                    )}
                </div>

                {/* NLP Classification Engine */}
                <div className="rounded-xl border border-slate-200/80 bg-slate-50 p-3.5">
                    <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold uppercase text-slate-500">
                            NLP Engine
                        </span>
                        {nlpHealthy ? (
                            <CheckCircle2 size={16} className="text-emerald-600" />
                        ) : (
                            <AlertTriangle size={16} className="text-amber-600" />
                        )}
                    </div>
                    <p className="mt-1 text-sm font-bold text-slate-900">
                        TF-IDF & Logistic Reg.
                    </p>
                    <p className="text-[11px] text-slate-500">
                        4-Class AG News &bull; {nlpHealthy ? "Model Loaded" : "Pending"}
                    </p>
                </div>
            </div>
        </div>
    );
}
