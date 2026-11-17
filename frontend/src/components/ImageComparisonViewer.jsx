import { useState, useRef, useEffect } from "react";
import {
    Maximize2,
    Eye,
    Sliders,
    Layers,
    SplitSquareVertical,
    Grid3X3,
    X,
} from "lucide-react";

/**
 * Format base64 string to standard data URI if not already prefixed.
 */
const toDataUri = (src) => {
    if (!src) return "";
    if (src.startsWith("data:") || src.startsWith("blob:") || src.startsWith("http")) {
        return src;
    }
    return `data:image/png;base64,${src}`;
};

export default function ImageComparisonViewer({
    assets = [], // Array of { id, label, src, description, category, legend }
    defaultAssetId,
    beforeImageSrc,
    afterImageSrc,
    showSlider = true,
    showLegend = true,
    className = "",
}) {
    const [selectedId, setSelectedId] = useState(
        defaultAssetId || (assets[0] ? assets[0].id : null)
    );
    const [viewMode, setViewMode] = useState("single"); // "single" | "slider" | "sideBySide"
    const [sliderPos, setSliderPos] = useState(50); // percentage [0, 100]
    const [isLightboxOpen, setIsLightboxOpen] = useState(false);

    // Sync defaultAssetId if it changes
    useEffect(() => {
        if (defaultAssetId) setSelectedId(defaultAssetId);
    }, [defaultAssetId]);

    const activeAsset = assets.find((a) => a.id === selectedId) || assets[0];

    // Filter available assets
    const validAssets = assets.filter((a) => Boolean(a.src));

    return (
        <div
            className={`rounded-2xl border border-slate-200 bg-white shadow-xs overflow-hidden ${className}`}
        >
            {/* Top Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 bg-slate-50/80 px-4 py-3">
                <div className="flex items-center gap-1.5 overflow-x-auto">
                    <span className="text-xs font-semibold uppercase text-slate-500 mr-2 shrink-0">
                        View:
                    </span>
                    <button
                        type="button"
                        onClick={() => setViewMode("single")}
                        className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                            viewMode === "single"
                                ? "bg-white text-blue-600 shadow-xs border border-slate-200"
                                : "text-slate-600 hover:bg-slate-200/60"
                        }`}
                    >
                        <Eye size={13} /> Asset Viewer
                    </button>

                    {beforeImageSrc && afterImageSrc && showSlider && (
                        <button
                            type="button"
                            onClick={() => setViewMode("slider")}
                            className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                                viewMode === "slider"
                                    ? "bg-white text-blue-600 shadow-xs border border-slate-200"
                                    : "text-slate-600 hover:bg-slate-200/60"
                            }`}
                        >
                            <Sliders size={13} /> Before/After Slider
                        </button>
                    )}

                    {beforeImageSrc && afterImageSrc && (
                        <button
                            type="button"
                            onClick={() => setViewMode("sideBySide")}
                            className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium transition ${
                                viewMode === "sideBySide"
                                    ? "bg-white text-blue-600 shadow-xs border border-slate-200"
                                    : "text-slate-600 hover:bg-slate-200/60"
                            }`}
                        >
                            <SplitSquareVertical size={13} /> Side-by-Side
                        </button>
                    )}
                </div>

                <div className="flex items-center gap-2">
                    <button
                        type="button"
                        onClick={() => setIsLightboxOpen(true)}
                        className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-medium text-slate-700 shadow-2xs hover:bg-slate-50"
                        title="Open expanded lightbox"
                    >
                        <Maximize2 size={13} /> Expand
                    </button>
                </div>
            </div>

            {/* Asset Selection Tabs (when in single view) */}
            {viewMode === "single" && (
                <div className="border-b border-slate-100 bg-white px-4 py-2">
                    <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-thin">
                        {validAssets.map((asset) => {
                            const isSelected = asset.id === selectedId;
                            return (
                                <button
                                    key={asset.id}
                                    type="button"
                                    onClick={() => setSelectedId(asset.id)}
                                    className={`shrink-0 rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                                        isSelected
                                            ? "bg-blue-600 text-white shadow-2xs"
                                            : "bg-slate-100 text-slate-700 hover:bg-slate-200"
                                    }`}
                                >
                                    {asset.label}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Display Area */}
            <div className="relative min-h-[360px] bg-slate-950 p-4 flex items-center justify-center">
                {viewMode === "single" && activeAsset && (
                    <div className="flex flex-col items-center justify-center w-full">
                        <img
                            src={toDataUri(activeAsset.src)}
                            alt={activeAsset.label}
                            className="max-h-[520px] w-auto max-w-full rounded-lg object-contain shadow-md"
                        />
                        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 w-full max-w-2xl px-2">
                            <span className="text-xs font-semibold text-slate-300">
                                {activeAsset.label}
                            </span>
                            {activeAsset.description && (
                                <span className="text-[11px] text-slate-400">
                                    {activeAsset.description}
                                </span>
                            )}
                        </div>
                    </div>
                )}

                {viewMode === "slider" && beforeImageSrc && afterImageSrc && (
                    <div className="w-full max-w-3xl flex flex-col items-center">
                        <div className="relative aspect-video max-h-[500px] w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-900 select-none">
                            {/* After Image (Background) */}
                            <img
                                src={toDataUri(afterImageSrc)}
                                alt="After Inspection"
                                className="absolute inset-0 h-full w-full object-contain"
                            />
                            {/* Before Image (Clipped Foreground) */}
                            <div
                                className="absolute inset-0 overflow-hidden"
                                style={{ width: `${sliderPos}%` }}
                            >
                                <img
                                    src={toDataUri(beforeImageSrc)}
                                    alt="Before Inspection"
                                    className="absolute inset-0 h-full w-full object-contain max-w-none"
                                    style={{ width: "100%", height: "100%" }}
                                />
                            </div>

                            {/* Divider Line */}
                            <div
                                className="absolute top-0 bottom-0 w-0.5 bg-white shadow-lg pointer-events-none"
                                style={{ left: `${sliderPos}%` }}
                            >
                                <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex h-7 w-7 items-center justify-center rounded-full bg-white text-slate-900 shadow-md">
                                    <Sliders size={14} />
                                </div>
                            </div>

                            {/* Labels */}
                            <span className="absolute top-3 left-3 rounded-md bg-slate-900/80 px-2 py-0.5 text-[11px] font-semibold text-white pointer-events-none">
                                Before ({sliderPos}%)
                            </span>
                            <span className="absolute top-3 right-3 rounded-md bg-slate-900/80 px-2 py-0.5 text-[11px] font-semibold text-white pointer-events-none">
                                After ({100 - sliderPos}%)
                            </span>
                        </div>

                        {/* Interactive Slider Input */}
                        <div className="mt-4 flex w-full max-w-md items-center gap-3">
                            <span className="text-xs text-slate-400">Before</span>
                            <input
                                type="range"
                                min="0"
                                max="100"
                                value={sliderPos}
                                onChange={(e) => setSliderPos(Number(e.target.value))}
                                aria-label="Before and after comparison slider"
                                className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-700 accent-blue-500"
                            />
                            <span className="text-xs text-slate-400">After</span>
                        </div>
                    </div>
                )}

                {viewMode === "sideBySide" && beforeImageSrc && afterImageSrc && (
                    <div className="grid w-full grid-cols-1 gap-4 md:grid-cols-2">
                        <div className="flex flex-col items-center rounded-xl bg-slate-900 p-3 border border-slate-800">
                            <span className="mb-2 text-xs font-semibold text-slate-300">
                                Before Inspection Photograph
                            </span>
                            <img
                                src={toDataUri(beforeImageSrc)}
                                alt="Before Inspection"
                                className="max-h-[420px] w-auto max-w-full rounded-lg object-contain"
                            />
                        </div>

                        <div className="flex flex-col items-center rounded-xl bg-slate-900 p-3 border border-slate-800">
                            <span className="mb-2 text-xs font-semibold text-slate-300">
                                After Inspection Photograph
                            </span>
                            <img
                                src={toDataUri(afterImageSrc)}
                                alt="After Inspection"
                                className="max-h-[420px] w-auto max-w-full rounded-lg object-contain"
                            />
                        </div>
                    </div>
                )}
            </div>

            {/* Bottom Legend */}
            {showLegend && (
                <div className="border-t border-slate-100 bg-white px-5 py-3">
                    <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
                        <span className="font-semibold text-slate-700">
                            Crack Overlay Legend:
                        </span>
                        <div className="flex flex-wrap items-center gap-4">
                            <div className="flex items-center gap-1.5">
                                <span className="h-3 w-3 rounded-full bg-yellow-400 border border-yellow-500" />
                                <span className="text-slate-600">
                                    Persistent crack regions
                                </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="h-3 w-3 rounded-full bg-red-500 border border-red-600" />
                                <span className="text-slate-600">
                                    Potential newly observed cracks
                                </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="h-3 w-3 rounded-full bg-blue-500 border border-blue-600" />
                                <span className="text-slate-600">
                                    No-longer-observed crack regions
                                </span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                <span className="h-3 w-3 rounded-full bg-slate-800 border border-slate-600" />
                                <span className="text-slate-600">
                                    Dimmed: Outside valid overlap
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Expanded Lightbox Modal */}
            {isLightboxOpen && (
                <div
                    role="dialog"
                    aria-modal="true"
                    aria-label="Expanded Image Asset Viewer"
                    className="fixed inset-0 z-50 flex flex-col bg-slate-950/95 p-4 backdrop-blur-sm"
                >
                    <div className="flex items-center justify-between pb-3 text-white">
                        <div className="flex items-center gap-3">
                            <h3 className="font-semibold text-sm">
                                {activeAsset?.label || "Image Inspection Asset"}
                            </h3>
                            {activeAsset?.description && (
                                <span className="text-xs text-slate-400">
                                    — {activeAsset.description}
                                </span>
                            )}
                        </div>
                        <button
                            type="button"
                            onClick={() => setIsLightboxOpen(false)}
                            className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white"
                            aria-label="Close expanded view"
                        >
                            <X size={20} />
                        </button>
                    </div>

                    <div className="flex-1 flex items-center justify-center overflow-auto p-2">
                        {activeAsset && (
                            <img
                                src={toDataUri(activeAsset.src)}
                                alt={activeAsset.label}
                                className="max-h-[85vh] max-w-[90vw] object-contain rounded-lg shadow-2xl"
                            />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
