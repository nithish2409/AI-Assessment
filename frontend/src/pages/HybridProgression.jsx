import { useState, useRef } from "react";
import {
    GitCompareArrows,
    Sparkles,
    RotateCcw,
    Sliders,
    Clock,
    Cpu,
    ShieldAlert,
    AlertTriangle,
    CheckCircle2,
    Layers,
    Info,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ImageUpload from "../components/ImageUpload";
import ResultCard from "../components/ResultCard";
import ImageComparisonViewer from "../components/ImageComparisonViewer";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import ReliabilityNotice from "../components/ReliabilityNotice";
import { compareHybridImages, parseApiError } from "../services/api";

const ALIGNMENT_MODES = [
    { value: "auto", label: "Auto (SIFT → AKAZE → ORB)" },
    { value: "sift", label: "SIFT (Scale-Invariant)" },
    { value: "akaze", label: "AKAZE (Nonlinear Scale)" },
    { value: "orb", label: "ORB (High-Speed)" },
    { value: "direct", label: "Direct (Identity / Zero Warp)" },
];

export default function HybridProgression() {
    const [beforeFile, setBeforeFile] = useState(null);
    const [afterFile, setAfterFile] = useState(null);

    // Hyperparameters
    const [alignmentMode, setAlignmentMode] = useState("auto");
    const [confidence, setConfidence] = useState(0.25);
    const [iou, setIou] = useState(0.45);
    const [tolerancePx, setTolerancePx] = useState(2);

    // Execution state
    const [analyzing, setAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const abortControllerRef = useRef(null);

    // URLs for original files for 11-asset viewer
    const [beforeUrl, setBeforeUrl] = useState(null);
    const [afterUrl, setAfterUrl] = useState(null);

    const handleBeforeSelect = (file) => {
        setBeforeFile(file);
        if (beforeUrl) URL.revokeObjectURL(beforeUrl);
        setBeforeUrl(URL.createObjectURL(file));
        setResult(null);
        setError(null);
    };

    const handleAfterSelect = (file) => {
        setAfterFile(file);
        if (afterUrl) URL.revokeObjectURL(afterUrl);
        setAfterUrl(URL.createObjectURL(file));
        setResult(null);
        setError(null);
    };

    const handleClear = () => {
        if (beforeUrl) URL.revokeObjectURL(beforeUrl);
        if (afterUrl) URL.revokeObjectURL(afterUrl);
        setBeforeFile(null);
        setAfterFile(null);
        setBeforeUrl(null);
        setAfterUrl(null);
        setResult(null);
        setError(null);
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
    };

    const handleAnalyze = async () => {
        if (!beforeFile || !afterFile) {
            setError("Please upload both before and after photographs to perform hybrid crack comparison.");
            return;
        }

        setAnalyzing(true);
        setError(null);
        setResult(null);

        const controller = new AbortController();
        abortControllerRef.current = controller;

        try {
            const data = await compareHybridImages(
                beforeFile,
                afterFile,
                {
                    alignmentMode,
                    confidence,
                    iou,
                    tolerancePx,
                },
                controller.signal
            );
            setResult(data);
        } catch (err) {
            if (err.name !== "CanceledError") {
                setError(parseApiError(err));
            }
        } finally {
            setAnalyzing(false);
            abortControllerRef.current = null;
        }
    };

    const crack = result?.crack_comparison;
    const visual = result?.visual_comparison;
    const alignment = result?.alignment;
    const timing = result?.timing;

    // Build the 11-asset collection for ImageComparisonViewer
    const assets = result
        ? [
              {
                  id: "hybrid_overlay",
                  label: "1. Combined Hybrid Overlay",
                  src: result.images?.hybrid_overlay,
                  description: "Composite overlay with yellow persistent, red new, and blue no-longer-observed cracks",
              },
              {
                  id: "tolerance_mask",
                  label: "2. Tolerance-Adjusted New Mask",
                  src: result.images?.tolerance_adjusted_new_mask || result.images?.newly_observed_mask,
                  description: `Newly observed crack regions outside the ${crack?.tolerance_px ?? tolerancePx} px spatial tolerance buffer`,
              },
              {
                  id: "newly_observed",
                  label: "3. Potential Newly Observed Cracks",
                  src: result.images?.newly_observed_mask,
                  description: "Raw newly observed crack pixels in after mask relative to warped before mask",
              },
              {
                  id: "persistent",
                  label: "4. Persistent Predicted Crack Regions",
                  src: result.images?.persistent_mask,
                  description: "Crack regions detected in both before and after inspection masks",
              },
              {
                  id: "no_longer_observed",
                  label: "5. No-Longer-Observed Crack Regions",
                  src: result.images?.no_longer_observed_mask,
                  description: "Crack regions present in before mask but absent in after inspection",
              },
              {
                  id: "after_mask",
                  label: "6. After Crack Mask",
                  src: result.images?.after_crack_mask,
                  description: "YOLO26s-seg binary segmentation mask for after photograph",
              },
              {
                  id: "before_mask",
                  label: "7. Warped Before Crack Mask",
                  src: result.images?.before_crack_mask,
                  description: "Before crack mask warped into after coordinate system via homography H",
              },
              {
                  id: "visual_difference",
                  label: "8. SSIM Visual Difference",
                  src: result.images?.visual_difference,
                  description: "Photometric SSIM difference heatmap over valid overlap region",
              },
              {
                  id: "aligned_before",
                  label: "9. Aligned Before Photograph",
                  src: result.images?.aligned_before,
                  description: "Before photograph geometrically warped to after photograph coordinate frame",
              },
              {
                  id: "orig_before",
                  label: "10. Original Before Photograph",
                  src: beforeUrl,
                  description: "Unmodified original before photograph",
              },
              {
                  id: "orig_after",
                  label: "11. Original After Photograph",
                  src: afterUrl,
                  description: "Unmodified original after photograph",
              },
          ]
        : [];

    return (
        <div>
            <PageHeader
                title="Hybrid Crack Progression Analysis"
                description="Synergistic multi-modal pipeline integrating feature homography alignment, YOLO26s-seg crack segmentation, spatial tolerance buffering, and calibrated noise filtering to track concrete crack evolution."
                badge="Additional Feature"
                icon={GitCompareArrows}
                actions={
                    result && (
                        <button
                            type="button"
                            onClick={handleClear}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-50"
                        >
                            <RotateCcw size={14} /> Reset Analysis
                        </button>
                    )
                }
            />

            {/* Mandatory Terminology & Scope Banner */}
            <ReliabilityNotice type="caution" className="mb-6">
                <div>
                    <strong className="text-slate-900 font-semibold">Reliability & Interpretation Bounds:</strong> This module reports <em>predicted coverage change</em>, <em>potential newly observed cracks</em>, and <em>potential crack-region expansion</em>. It does not measure physical crack width in millimeters, does not claim confirmed structural damage progression, and never certifies that a building is safe.
                </div>
            </ReliabilityNotice>

            {/* Upload Section */}
            <div className="grid gap-6 md:grid-cols-2">
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                    <ImageUpload
                        id="hybrid-before-upload"
                        label="Before Inspection Photograph"
                        sublabel="Earlier inspection photograph (JPEG, PNG, WebP)"
                        file={beforeFile}
                        onFileSelect={handleBeforeSelect}
                        onFileClear={() => {
                            setBeforeFile(null);
                            if (beforeUrl) URL.revokeObjectURL(beforeUrl);
                            setBeforeUrl(null);
                            setResult(null);
                        }}
                        disabled={analyzing}
                        required
                    />
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                    <ImageUpload
                        id="hybrid-after-upload"
                        label="After Inspection Photograph"
                        sublabel="Later inspection photograph (JPEG, PNG, WebP)"
                        file={afterFile}
                        onFileSelect={handleAfterSelect}
                        onFileClear={() => {
                            setAfterFile(null);
                            if (afterUrl) URL.revokeObjectURL(afterUrl);
                            setAfterUrl(null);
                            setResult(null);
                        }}
                        disabled={analyzing}
                        required
                    />
                </div>
            </div>

            {/* Analysis Configuration Controls */}
            <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                <div className="flex items-center gap-1.5 text-xs font-semibold uppercase text-slate-500 mb-4">
                    <Sliders size={14} /> Analysis Configuration
                </div>

                <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
                    {/* Alignment Mode */}
                    <div>
                        <label
                            htmlFor="hybrid-align-mode"
                            className="block text-xs font-semibold text-slate-700 mb-1"
                        >
                            Alignment Mode
                        </label>
                        <select
                            id="hybrid-align-mode"
                            value={alignmentMode}
                            disabled={analyzing}
                            onChange={(e) => setAlignmentMode(e.target.value)}
                            className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-800 outline-none focus:border-blue-500"
                        >
                            {ALIGNMENT_MODES.map((m) => (
                                <option key={m.value} value={m.value}>
                                    {m.label}
                                </option>
                            ))}
                        </select>
                        <span className="text-[10px] text-slate-400 mt-1 block">
                            Feature detector for homography H.
                        </span>
                    </div>

                    {/* Confidence Threshold */}
                    <div>
                        <div className="flex justify-between text-xs text-slate-700 mb-1 font-semibold">
                            <span>Confidence</span>
                            <span className="font-mono text-blue-600">{confidence.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0.05"
                            max="0.95"
                            step="0.05"
                            value={confidence}
                            disabled={analyzing}
                            onChange={(e) => setConfidence(parseFloat(e.target.value))}
                            className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 accent-blue-600"
                        />
                        <span className="text-[10px] text-slate-400 mt-1 block">
                            YOLO crack detection threshold.
                        </span>
                    </div>

                    {/* NMS IoU Threshold */}
                    <div>
                        <div className="flex justify-between text-xs text-slate-700 mb-1 font-semibold">
                            <span>NMS IoU</span>
                            <span className="font-mono text-blue-600">{iou.toFixed(2)}</span>
                        </div>
                        <input
                            type="range"
                            min="0.10"
                            max="0.90"
                            step="0.05"
                            value={iou}
                            disabled={analyzing}
                            onChange={(e) => setIou(parseFloat(e.target.value))}
                            className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 accent-blue-600"
                        />
                        <span className="text-[10px] text-slate-400 mt-1 block">
                            Non-Maximum Suppression threshold.
                        </span>
                    </div>

                    {/* Spatial Tolerance Radius */}
                    <div>
                        <div className="flex justify-between text-xs text-slate-700 mb-1 font-semibold">
                            <span>Tolerance (px)</span>
                            <span className="font-mono text-blue-600">{tolerancePx} px</span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="10"
                            step="1"
                            value={tolerancePx}
                            disabled={analyzing}
                            onChange={(e) => setTolerancePx(parseInt(e.target.value, 10))}
                            className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-200 accent-blue-600"
                        />
                        <span className="text-[10px] text-slate-400 mt-1 block">
                            Dilation buffer absorbing registration shifts.
                        </span>
                    </div>
                </div>

                {/* Submit Row */}
                <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
                    <div className="text-xs text-slate-500">
                        Operating Noise Floor: <strong className="text-slate-800">0.89 pp (provisional)</strong>
                    </div>

                    <button
                        type="button"
                        onClick={handleAnalyze}
                        disabled={!beforeFile || !afterFile || analyzing}
                        className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-2.5 text-xs font-semibold text-white shadow-xs hover:bg-blue-700 disabled:opacity-50"
                    >
                        <Sparkles size={16} />
                        {analyzing ? "Running Hybrid Analysis..." : "Execute Hybrid Progression"}
                    </button>
                </div>
            </div>

            {/* Error Notification */}
            {error && (
                <div className="mt-6">
                    <ErrorState
                        error={error}
                        title="Hybrid Progression Analysis Failed"
                        onRetry={handleAnalyze}
                    />
                </div>
            )}

            {/* Loading Indicator */}
            {analyzing && (
                <div className="mt-6">
                    <LoadingState
                        title="Executing Hybrid Damage Progression Pipeline..."
                        description="Aligning photographs via homography H, running dual YOLO forward passes, buffering spatial tolerance, and classifying connected components."
                        onCancel={() => {
                            if (abortControllerRef.current) {
                                abortControllerRef.current.abort();
                                setAnalyzing(false);
                            }
                        }}
                    />
                </div>
            )}

            {/* Comprehensive Results Display */}
            {result && !analyzing && (
                <div className="mt-8 space-y-6">
                    {/* Primary Reliability & Progression Interpretation Banner */}
                    <ReliabilityNotice
                        progressionData={{
                            progression_label: crack?.progression_label,
                            uncertainty_level: crack?.uncertainty_level,
                            interpretation_summary: crack?.interpretation_summary,
                            calibrated_noise_floor_pp: crack?.calibrated_noise_floor_pp,
                            is_above_noise_floor: crack?.is_above_noise_floor,
                            coverage_change_percentage_points: crack?.coverage_change_percentage_points,
                        }}
                    />

                    {/* Operational Warnings from Backend */}
                    {result.warnings && result.warnings.length > 0 && (
                        <div className="rounded-2xl border border-amber-200 bg-amber-50/70 p-4 text-xs text-amber-900">
                            <div className="flex items-start gap-2">
                                <AlertTriangle size={16} className="mt-0.5 text-amber-600 shrink-0" />
                                <div>
                                    <strong className="font-semibold">Analysis Warnings:</strong>
                                    <ul className="mt-1 list-inside list-disc space-y-0.5">
                                        {result.warnings.map((w, i) => (
                                            <li key={i}>{w}</li>
                                        ))}
                                    </ul>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Core KPI Cards */}
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        <ResultCard
                            title="Predicted Coverage Change"
                            value={`${(crack?.coverage_change_percentage_points ?? 0) >= 0 ? "+" : ""}${(crack?.coverage_change_percentage_points ?? 0).toFixed(2)} pp`}
                            subtitle={`Before: ${(crack?.before_crack_coverage_percentage ?? 0).toFixed(2)}% → After: ${(crack?.after_crack_coverage_percentage ?? 0).toFixed(2)}%`}
                            valueColor={
                                crack?.is_above_noise_floor
                                    ? "text-amber-600"
                                    : "text-slate-900"
                            }
                            badge={crack?.is_above_noise_floor ? "Above Noise" : "Within Noise"}
                            badgeType={crack?.is_above_noise_floor ? "warning" : "neutral"}
                        />

                        <ResultCard
                            title="Mean SSIM Score"
                            value={(visual?.ssim_score ?? 0).toFixed(4)}
                            subtitle={`Visual difference: ${(visual?.visual_change_percentage ?? 0).toFixed(2)}%`}
                            badge={
                                (visual?.ssim_score ?? 0) > 0.85
                                    ? "High Similarity"
                                    : "Changed"
                            }
                        />

                        <ResultCard
                            title="Valid Overlap Area"
                            value={`${(alignment?.valid_overlap_percentage ?? 0).toFixed(1)}%`}
                            subtitle={`${(alignment?.valid_overlap_pixels ?? 0).toLocaleString()} corresponding pixels`}
                            badge={alignment?.status || "accepted"}
                            badgeType={alignment?.status === "accepted" ? "success" : "neutral"}
                        />

                        <ResultCard
                            title="Total Latency"
                            value={
                                timing?.total_duration_ms
                                    ? `${timing.total_duration_ms.toFixed(0)} ms`
                                    : "N/A"
                            }
                            subtitle={`Device: ${result.model?.name || "YOLO26s-seg"}`}
                            icon={Clock}
                        />
                    </div>

                    {/* Crack Progression Breakdown Matrix */}
                    <div className="grid gap-6 md:grid-cols-2">
                        {/* Pixel-Level Metrics */}
                        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs space-y-3">
                            <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                                <Layers size={18} className="text-blue-600" />
                                <h3 className="text-sm font-semibold text-slate-900">
                                    Pixel-Level Progression Breakdown
                                </h3>
                            </div>

                            <div className="space-y-2 text-xs">
                                <div className="flex justify-between py-1 border-b border-slate-100">
                                    <span className="text-slate-600 flex items-center gap-1.5">
                                        <span className="h-2 w-2 rounded-full bg-yellow-400" />
                                        Persistent predicted crack regions:
                                    </span>
                                    <span className="font-mono font-bold text-slate-800">
                                        {(crack?.persistent_pixels ?? 0).toLocaleString()} px
                                    </span>
                                </div>

                                <div className="flex justify-between py-1 border-b border-slate-100">
                                    <span className="text-slate-600 flex items-center gap-1.5">
                                        <span className="h-2 w-2 rounded-full bg-red-500" />
                                        Potential newly observed cracks (raw):
                                    </span>
                                    <span className="font-mono font-bold text-slate-800">
                                        {(crack?.newly_observed_pixels ?? 0).toLocaleString()} px
                                    </span>
                                </div>

                                <div className="flex justify-between py-1 border-b border-slate-100">
                                    <span className="text-slate-600 flex items-center gap-1.5">
                                        <span className="h-2 w-2 rounded-full bg-blue-500" />
                                        No-longer-observed crack regions:
                                    </span>
                                    <span className="font-mono font-bold text-slate-800">
                                        {(crack?.no_longer_observed_pixels ?? 0).toLocaleString()} px
                                    </span>
                                </div>

                                <div className="flex justify-between py-1 bg-slate-50 px-2 rounded-lg">
                                    <span className="font-semibold text-slate-700">
                                        Tolerance-Adjusted Newly Observed ({crack?.tolerance_px ?? 2} px):
                                    </span>
                                    <span className="font-mono font-bold text-blue-600">
                                        {(crack?.tolerance_adjusted_new_pixels ?? 0).toLocaleString()} px
                                    </span>
                                </div>
                            </div>
                        </div>

                        {/* Connected Component Breakdown */}
                        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs space-y-3">
                            <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
                                <GitCompareArrows size={18} className="text-blue-600" />
                                <h3 className="text-sm font-semibold text-slate-900">
                                    Connected Component Morphometry
                                </h3>
                            </div>

                            <div className="space-y-2 text-xs">
                                <div className="flex justify-between py-1 border-b border-slate-100">
                                    <span className="text-slate-600">
                                        Potential new crack components:
                                    </span>
                                    <span className="font-mono font-bold text-rose-600">
                                        {crack?.potential_new_components ?? 0}
                                    </span>
                                </div>

                                <div className="flex justify-between py-1 border-b border-slate-100">
                                    <span className="text-slate-600">
                                        Potential crack-region expansion:
                                    </span>
                                    <span className="font-mono font-bold text-amber-600">
                                        {crack?.potential_expanded_components ?? 0}
                                    </span>
                                </div>

                                <div className="flex justify-between py-1 border-b border-slate-100">
                                    <span className="text-slate-600">
                                        Persistent crack components:
                                    </span>
                                    <span className="font-mono font-bold text-slate-800">
                                        {crack?.persistent_components ?? 0}
                                    </span>
                                </div>

                                <div className="flex justify-between py-1 border-b border-slate-100">
                                    <span className="text-slate-600">
                                        No-longer-observed components:
                                    </span>
                                    <span className="font-mono font-bold text-slate-800">
                                        {crack?.no_longer_observed_components ?? 0}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 11-Asset Visual Inspection Viewer with Slider and Color Legend */}
                    <div>
                        <div className="mb-2 flex items-center justify-between">
                            <h3 className="text-sm font-bold text-slate-900">
                                Visual Progression Assets (11 Inspection Layers)
                            </h3>
                            <span className="text-xs text-slate-500">
                                Switch tabs or toggle side-by-side & slider views below
                            </span>
                        </div>

                        <ImageComparisonViewer
                            assets={assets}
                            defaultAssetId="hybrid_overlay"
                            beforeImageSrc={beforeUrl}
                            afterImageSrc={afterUrl}
                            showSlider={true}
                            showLegend={true}
                        />
                    </div>

                    {/* Pipeline Execution Telemetry Table */}
                    {timing && (
                        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                            <h3 className="text-xs font-semibold uppercase text-slate-500 mb-3 flex items-center gap-1.5">
                                <Clock size={14} /> Stage-by-Stage Latency Breakdown
                            </h3>
                            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6 text-xs">
                                <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-200/80">
                                    <span className="text-slate-400 block text-[10px]">Alignment</span>
                                    <span className="font-mono font-bold text-slate-800">{timing.alignment_ms.toFixed(1)} ms</span>
                                </div>
                                <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-200/80">
                                    <span className="text-slate-400 block text-[10px]">SSIM Difference</span>
                                    <span className="font-mono font-bold text-slate-800">{timing.ssim_ms.toFixed(1)} ms</span>
                                </div>
                                <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-200/80">
                                    <span className="text-slate-400 block text-[10px]">YOLO Before</span>
                                    <span className="font-mono font-bold text-slate-800">{timing.yolo_before_ms.toFixed(1)} ms</span>
                                </div>
                                <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-200/80">
                                    <span className="text-slate-400 block text-[10px]">YOLO After</span>
                                    <span className="font-mono font-bold text-slate-800">{timing.yolo_after_ms.toFixed(1)} ms</span>
                                </div>
                                <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-200/80">
                                    <span className="text-slate-400 block text-[10px]">Spatial Comparison</span>
                                    <span className="font-mono font-bold text-slate-800">{timing.crack_comparison_ms.toFixed(1)} ms</span>
                                </div>
                                <div className="rounded-xl bg-slate-50 p-2.5 border border-slate-200/80">
                                    <span className="text-slate-400 block text-[10px]">Asset Encoding</span>
                                    <span className="font-mono font-bold text-slate-800">{timing.visualization_ms.toFixed(1)} ms</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
