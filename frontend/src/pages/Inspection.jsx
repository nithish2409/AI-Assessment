import { useState, useRef } from "react";
import {
    Building2,
    Factory,
    ShieldAlert,
    AlertTriangle,
    CheckCircle2,
    Info,
    FileText,
    Download,
    Eye,
    Layers,
    Cpu,
    RefreshCw,
    Sliders,
    Search,
    AlertCircle,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ErrorState from "../components/ErrorState";
import { analyzeInspection, downloadInspectionPdf, parseApiError } from "../services/api";

const DOMAINS = [
    {
        id: "building",
        label: "Building Structural Inspection",
        icon: Building2,
        description: "Concrete cracks, spallation, exposed rebar, efflorescence & structural components",
        models: ["ResNet18-CODEBRIM", "YOLO26s-seg", "YOLOv8s-Worldv2"],
    },
    {
        id: "industrial",
        label: "Industrial Equipment Inspection",
        icon: Factory,
        description: "Surface corrosion, metal weld cracks, bolt slippage, equipment parts & nameplate OCR",
        models: ["YOLO26s-Industrial", "YOLOv8s-Worldv2", "EasyOCR"],
    },
];

export default function Inspection() {
    const [domain, setDomain] = useState("building");
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [confidence, setConfidence] = useState(0.25);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [results, setResults] = useState(null);
    const [activeTab, setActiveTab] = useState("overlay");
    const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);

    const fileInputRef = useRef(null);

    const handleFileChange = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedFile(file);
            setResults(null);
            setError(null);
            const objectUrl = URL.createObjectURL(file);
            setPreviewUrl(objectUrl);
        }
    };

    const handleClear = () => {
        setSelectedFile(null);
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setPreviewUrl(null);
        setResults(null);
        setError(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;

        setIsLoading(true);
        setError(null);
        setActiveTab("overlay");

        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("domain", domain);
        formData.append("confidence", String(confidence));

        try {
            const data = await analyzeInspection(formData);
            setResults(data);
        } catch (err) {
            setError(parseApiError(err));
        } finally {
            setIsLoading(false);
        }
    };

    const handleDownloadPdf = async () => {
        if (!results) return;
        setIsDownloadingPdf(true);
        try {
            const blob = await downloadInspectionPdf(results);
            const url = window.URL.createObjectURL(new Blob([blob], { type: "application/pdf" }));
            const a = document.createElement("a");
            a.href = url;
            a.download = `${results.inspection_id}_report.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            alert("Failed to download PDF report. Please try again.");
        } finally {
            setIsDownloadingPdf(false);
        }
    };

    const handleDownloadJson = () => {
        if (!results) return;
        const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${results.inspection_id}_findings.json`;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        document.body.removeChild(a);
    };

    // Severity styling helpers
    const getSeverityBadge = (severity) => {
        switch (severity?.toLowerCase()) {
            case "high":
                return {
                    badgeClass: "bg-red-50 text-red-700 border-red-200",
                    indicatorClass: "bg-red-500",
                    label: "HIGH VISUAL SEVERITY",
                    icon: AlertCircle,
                };
            case "moderate":
                return {
                    badgeClass: "bg-amber-50 text-amber-700 border-amber-200",
                    indicatorClass: "bg-amber-500",
                    label: "MODERATE VISUAL SEVERITY",
                    icon: AlertTriangle,
                };
            case "low":
                return {
                    badgeClass: "bg-emerald-50 text-emerald-700 border-emerald-200",
                    indicatorClass: "bg-emerald-500",
                    label: "LOW VISUAL SEVERITY",
                    icon: CheckCircle2,
                };
            default:
                return {
                    badgeClass: "bg-slate-50 text-slate-700 border-slate-200",
                    indicatorClass: "bg-slate-400",
                    label: "UNDETERMINED SEVERITY",
                    icon: Info,
                };
        }
    };

    const getPriorityPill = (priority) => {
        switch (priority?.toLowerCase()) {
            case "immediate_action":
                return <span className="rounded-full bg-red-100 px-2.5 py-0.5 text-xs font-semibold text-red-800">Priority: Immediate Action Required</span>;
            case "review_recommended":
                return <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-800">Priority: Review Recommended</span>;
            case "monitor":
                return <span className="rounded-full bg-emerald-100 px-2.5 py-0.5 text-xs font-semibold text-emerald-800">Priority: Routine Monitoring</span>;
            default:
                return <span className="rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-semibold text-slate-800">Priority: Further Investigation</span>;
        }
    };

    const sev = results?.visual_severity ? getSeverityBadge(results.visual_severity.visual_severity) : null;
    const SevIcon = sev?.icon;

    return (
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
            <PageHeader
                title="AI Damage Inspection & Visual Severity"
                subtitle="Dual-domain civil structural & industrial equipment inspection powered by GPU vision models."
            />

            {/* Domain Selection Tabs */}
            <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
                {DOMAINS.map((item) => {
                    const Icon = item.icon;
                    const isActive = domain === item.id;
                    return (
                        <button
                            key={item.id}
                            type="button"
                            onClick={() => {
                                setDomain(item.id);
                                setResults(null);
                                setError(null);
                            }}
                            className={`flex flex-col text-left rounded-xl border p-5 transition-all ${
                                isActive
                                    ? "border-blue-600 bg-blue-50/50 shadow-sm ring-2 ring-blue-600/20"
                                    : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                            }`}
                        >
                            <div className="flex items-center gap-3">
                                <div className={`rounded-lg p-2.5 ${isActive ? "bg-blue-600 text-white" : "bg-slate-100 text-slate-600"}`}>
                                    <Icon size={22} />
                                </div>
                                <div>
                                    <h2 className="text-base font-semibold text-slate-900">{item.label}</h2>
                                    <p className="text-xs text-slate-500 mt-0.5">{item.description}</p>
                                </div>
                            </div>
                            <div className="mt-4 flex flex-wrap gap-1.5 pt-3 border-t border-slate-100">
                                {item.models.map((m) => (
                                    <span key={m} className="rounded bg-white px-2 py-0.5 text-[11px] font-medium text-slate-600 border border-slate-200">
                                        {m}
                                    </span>
                                ))}
                            </div>
                        </button>
                    );
                })}
            </div>

            {/* Upload & Configuration Section */}
            <div className="mt-6 rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                    {/* Upload Zone */}
                    <div className="lg:col-span-2">
                        <label className="block text-sm font-medium text-slate-700 mb-2">
                            Select Photograph to Inspect (JPEG, PNG, WebP)
                        </label>
                        {!previewUrl ? (
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50/60 p-8 text-center transition-colors hover:border-blue-500 hover:bg-blue-50/30"
                            >
                                <div className="rounded-full bg-blue-100 p-3 text-blue-600 mb-3">
                                    <Layers size={24} />
                                </div>
                                <p className="text-sm font-semibold text-slate-700">Click to upload or drag & drop</p>
                                <p className="text-xs text-slate-400 mt-1">JPEG, PNG, or WebP up to 10 MB</p>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/jpeg,image/png,image/webp"
                                    onChange={handleFileChange}
                                    className="hidden"
                                />
                            </div>
                        ) : (
                            <div className="relative rounded-xl border border-slate-200 bg-slate-900 overflow-hidden">
                                <img
                                    src={previewUrl}
                                    alt="Upload preview"
                                    className="max-h-80 w-full object-contain mx-auto"
                                />
                                <div className="absolute bottom-0 inset-x-0 bg-slate-950/70 p-3 text-white flex justify-between items-center backdrop-blur-sm">
                                    <span className="text-xs truncate max-w-xs">{selectedFile?.name}</span>
                                    <button
                                        type="button"
                                        onClick={handleClear}
                                        className="text-xs text-red-400 hover:text-red-300 font-medium ml-4"
                                    >
                                        Change Image
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Controls & Threshold Settings */}
                    <div className="flex flex-col justify-between border-t lg:border-t-0 lg:border-l border-slate-200 lg:pl-6 pt-4 lg:pt-0">
                        <div>
                            <h3 className="text-sm font-semibold text-slate-900 flex items-center gap-2">
                                <Sliders size={16} className="text-slate-500" />
                                Inspection Thresholds
                            </h3>
                            <p className="text-xs text-slate-500 mt-1">
                                Detection sensitivity confidence cutoff for bounding box and segmentation models.
                            </p>

                            <div className="mt-4">
                                <div className="flex justify-between text-xs text-slate-600 mb-1">
                                    <span>Confidence Threshold</span>
                                    <span className="font-semibold text-slate-900">{confidence.toFixed(2)}</span>
                                </div>
                                <input
                                    type="range"
                                    min="0.10"
                                    max="0.80"
                                    step="0.05"
                                    value={confidence}
                                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                                    className="w-full accent-blue-600"
                                />
                                <div className="flex justify-between text-[10px] text-slate-400 mt-1">
                                    <span>0.10 (High Sensitivity)</span>
                                    <span>0.80 (High Precision)</span>
                                </div>
                            </div>

                            <div className="mt-6 rounded-lg bg-slate-50 p-3 border border-slate-200 text-xs text-slate-600">
                                <span className="font-semibold text-slate-800">Inspection Mode:</span>{" "}
                                {domain === "building" ? "Civil Concrete Damage" : "Industrial Defect & OCR"}
                            </div>
                        </div>

                        <div className="mt-6">
                            <button
                                type="button"
                                onClick={handleAnalyze}
                                disabled={!selectedFile || isLoading}
                                className={`w-full rounded-xl py-3 px-4 font-semibold text-white shadow-sm transition-all flex items-center justify-center gap-2 ${
                                    !selectedFile || isLoading
                                        ? "bg-slate-300 cursor-not-allowed"
                                        : "bg-blue-600 hover:bg-blue-700 active:scale-[0.99]"
                                }`}
                            >
                                {isLoading ? (
                                    <>
                                        <RefreshCw size={18} className="animate-spin" />
                                        <span>Running GPU Vision Fleet...</span>
                                    </>
                                ) : (
                                    <>
                                        <Search size={18} />
                                        <span>Start AI Inspection</span>
                                    </>
                                )}
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Error Display */}
            {error && (
                <div className="mt-6">
                    <ErrorState error={error} onRetry={handleAnalyze} />
                </div>
            )}

            {/* Results Presentation */}
            {results && (
                <div className="mt-8 space-y-6">
                    {/* Visual Severity Screening Banner */}
                    <div className={`rounded-xl border p-6 shadow-sm ${sev?.badgeClass || "bg-white border-slate-200"}`}>
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-current/10">
                            <div className="flex items-center gap-3">
                                {SevIcon && <SevIcon size={28} />}
                                <div>
                                    <div className="flex items-center gap-2.5">
                                        <h3 className="text-xl font-bold tracking-tight">{sev?.label}</h3>
                                        {getPriorityPill(results.visual_severity.inspection_priority)}
                                    </div>
                                    <p className="text-xs opacity-80 mt-0.5">
                                        Deterministic screening estimate based on visual defect extent and co-occurrence.
                                    </p>
                                </div>
                            </div>

                            <div className="flex items-center gap-2">
                                <button
                                    type="button"
                                    onClick={handleDownloadPdf}
                                    disabled={isDownloadingPdf}
                                    className="rounded-lg bg-white px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-sm border border-slate-300 hover:bg-slate-50 flex items-center gap-1.5"
                                >
                                    <FileText size={15} />
                                    <span>{isDownloadingPdf ? "Compiling..." : "Download PDF Report"}</span>
                                </button>
                                <button
                                    type="button"
                                    onClick={handleDownloadJson}
                                    className="rounded-lg bg-white px-3.5 py-2 text-xs font-semibold text-slate-800 shadow-sm border border-slate-300 hover:bg-slate-50 flex items-center gap-1.5"
                                >
                                    <Download size={15} />
                                    <span>Export JSON</span>
                                </button>
                            </div>
                        </div>

                        {/* Rationale and Evidence */}
                        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                            <div>
                                <h4 className="font-semibold uppercase tracking-wider text-[11px] opacity-75 mb-1.5">
                                    Assessment Rationale:
                                </h4>
                                <ul className="space-y-1 list-disc list-inside">
                                    {results.visual_severity.rationale.map((r, i) => (
                                        <li key={i}>{r}</li>
                                    ))}
                                </ul>
                            </div>
                            <div>
                                <h4 className="font-semibold uppercase tracking-wider text-[11px] opacity-75 mb-1.5">
                                    Technical Limitations & Scope:
                                </h4>
                                <ul className="space-y-1 list-disc list-inside opacity-90">
                                    {results.visual_severity.limitations.map((l, i) => (
                                        <li key={i}>{l}</li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>

                    {/* Visual Viewer & Overlay Tabs */}
                    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                        <div className="flex items-center justify-between pb-4 border-b border-slate-200">
                            <div>
                                <h3 className="text-base font-semibold text-slate-900">Visual Findings & Overlays</h3>
                                <p className="text-xs text-slate-500">Inspection ID: {results.inspection_id} ({results.image_width} × {results.image_height} px)</p>
                            </div>

                            <div className="flex gap-1.5 rounded-lg bg-slate-100 p-1">
                                <button
                                    type="button"
                                    onClick={() => setActiveTab("overlay")}
                                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                                        activeTab === "overlay" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                                    }`}
                                >
                                    Annotated Overlay
                                </button>
                                {results.images.mask && (
                                    <button
                                        type="button"
                                        onClick={() => setActiveTab("mask")}
                                        className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                                            activeTab === "mask" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                                        }`}
                                    >
                                        Damage Mask
                                    </button>
                                )}
                                <button
                                    type="button"
                                    onClick={() => setActiveTab("original")}
                                    className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                                        activeTab === "original" ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
                                    }`}
                                >
                                    Original
                                </button>
                            </div>
                        </div>

                        <div className="mt-4 rounded-xl bg-slate-950 p-2 overflow-hidden flex items-center justify-center min-h-[380px]">
                            {activeTab === "overlay" && (
                                <img
                                    src={`data:image/png;base64,${results.images.overlay}`}
                                    alt="Annotated inspection overlay"
                                    className="max-h-[520px] w-auto object-contain rounded"
                                />
                            )}
                            {activeTab === "mask" && (
                                <img
                                    src={`data:image/png;base64,${results.images.mask}`}
                                    alt="Binary damage mask"
                                    className="max-h-[520px] w-auto object-contain rounded"
                                />
                            )}
                            {activeTab === "original" && (
                                <img
                                    src={`data:image/png;base64,${results.images.original}`}
                                    alt="Original photograph"
                                    className="max-h-[520px] w-auto object-contain rounded"
                                />
                            )}
                        </div>

                        {/* Image Metadata Strip */}
                        <div className="mt-3 flex flex-wrap items-center justify-between text-xs text-slate-500">
                            <div>
                                <span>Relative Damage Footprint: </span>
                                <span className="font-semibold text-slate-800">{results.crack_coverage_percentage.toFixed(3)}%</span>
                            </div>
                            <div>
                                <span>Inference Latency: </span>
                                <span className="font-semibold text-slate-800">{results.processing_time_ms} ms</span>
                            </div>
                        </div>
                    </div>

                    {/* Defect Findings Breakdown */}
                    <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                        <h3 className="text-base font-semibold text-slate-900 mb-1">
                            Defect Findings Inventory ({results.defects_count} items)
                        </h3>
                        <p className="text-xs text-slate-500 mb-4">
                            Distinguishes between pixel segmentation, bounding-box localization, and image-level classification findings.
                        </p>

                        {results.defects.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="min-w-full divide-y divide-slate-200 text-left text-xs">
                                    <thead className="bg-slate-50 text-slate-600 font-semibold">
                                        <tr>
                                            <th className="py-2.5 px-3">ID</th>
                                            <th className="py-2.5 px-3">Category</th>
                                            <th className="py-2.5 px-3">Finding Type</th>
                                            <th className="py-2.5 px-3">Source Model</th>
                                            <th className="py-2.5 px-3">Confidence / Prob</th>
                                            <th className="py-2.5 px-3">Component Context</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-100">
                                        {results.defects.map((d) => (
                                            <tr key={d.defect_id} className="hover:bg-slate-50/50">
                                                <td className="py-2.5 px-3 font-mono font-medium text-slate-900">{d.defect_id}</td>
                                                <td className="py-2.5 px-3 font-semibold text-slate-800 capitalize">{d.category}</td>
                                                <td className="py-2.5 px-3">
                                                    <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                                                        d.finding_type === "segmentation"
                                                            ? "bg-red-100 text-red-800"
                                                            : d.finding_type === "bounding_box"
                                                            ? "bg-amber-100 text-amber-800"
                                                            : "bg-purple-100 text-purple-800"
                                                    }`}>
                                                        {d.finding_type.replace("_", " ").toUpperCase()}
                                                    </span>
                                                </td>
                                                <td className="py-2.5 px-3 text-slate-600">{d.source_model}</td>
                                                <td className="py-2.5 px-3 font-semibold text-slate-700">
                                                    {(d.confidence * 100).toFixed(1)}%
                                                </td>
                                                <td className="py-2.5 px-3 text-slate-600 capitalize">
                                                    {d.associated_component || "Unknown"}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <div className="rounded-lg bg-slate-50 p-6 text-center text-xs text-slate-500">
                                No discrete defect findings identified within detector sensitivity.
                            </div>
                        )}
                    </div>

                    {/* Component Context & OCR Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Component Localization */}
                        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                            <h3 className="text-sm font-semibold text-slate-900 mb-1 flex items-center gap-2">
                                <Cpu size={16} className="text-slate-500" />
                                Structural / Equipment Components
                            </h3>
                            <p className="text-xs text-slate-500 mb-3">
                                Localized spatial context derived via zero-shot YOLO-World.
                            </p>
                            {results.components.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                    {results.components.map((c, i) => (
                                        <div key={i} className="rounded-lg bg-slate-50 border border-slate-200 px-3 py-1.5 text-xs">
                                            <span className="font-semibold text-slate-800 capitalize">{c.name}</span>
                                            <span className="ml-1.5 text-slate-400">({(c.confidence * 100).toFixed(0)}%)</span>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-xs text-slate-400 italic">No specific architectural or machinery components localized.</p>
                            )}
                        </div>

                        {/* OCR Text Findings */}
                        <div className="rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
                            <h3 className="text-sm font-semibold text-slate-900 mb-1 flex items-center gap-2">
                                <FileText size={16} className="text-slate-500" />
                                Nameplate & Label OCR Extractions
                            </h3>
                            <p className="text-xs text-slate-500 mb-3">
                                Optical character recognition from visible equipment tags via EasyOCR.
                            </p>
                            {results.ocr_results && results.ocr_results.length > 0 ? (
                                <div className="space-y-1.5">
                                    {results.ocr_results.map((o, i) => (
                                        <div key={i} className="flex justify-between items-center rounded bg-slate-50 px-2.5 py-1 text-xs border border-slate-200 font-mono">
                                            <span className="font-semibold text-slate-900">{o.text}</span>
                                            <span className="text-[10px] text-slate-500">{(o.confidence * 100).toFixed(0)}% conf</span>
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-xs text-slate-400 italic">No legible text or nameplate markings detected in image.</p>
                            )}
                        </div>
                    </div>

                    {/* Mandatory Screening Disclaimer Banner */}
                    <div className="rounded-xl bg-amber-50/80 border border-amber-200/80 p-4 text-xs text-amber-900 flex items-start gap-3">
                        <AlertTriangle size={18} className="text-amber-600 shrink-0 mt-0.5" />
                        <div>
                            <p className="font-semibold">MANDATORY INSPECTION SCREENING NOTICE:</p>
                            <p className="mt-0.5 leading-relaxed text-amber-800">
                                {results.disclaimer}
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
