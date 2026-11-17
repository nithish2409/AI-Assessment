import { useState, useRef } from "react";
import {
    SplitSquareVertical,
    Sparkles,
    CheckCircle2,
    RotateCcw,
    Sliders,
    AlertCircle,
    AlertTriangle,
    Info,
    Eye,
    ShieldAlert,
    Cpu,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import ImageUpload from "../components/ImageUpload";
import ResultCard from "../components/ResultCard";
import ImageComparisonViewer from "../components/ImageComparisonViewer";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import ReliabilityNotice from "../components/ReliabilityNotice";
import {
    validateComparisonImages,
    compareInspectionImages,
    parseApiError,
} from "../services/api";

const ALIGNMENT_MODES = [
    { value: "auto", label: "Auto (SIFT → AKAZE → ORB)", desc: "Attempts detectors in order of quality; never silently falls back to direct." },
    { value: "sift", label: "SIFT (Scale-Invariant)", desc: "High-accuracy floating-point descriptor for scale and rotation changes." },
    { value: "akaze", label: "AKAZE (Nonlinear Scale)", desc: "Fast nonlinear diffusion descriptor preserving crisp boundaries." },
    { value: "orb", label: "ORB (Oriented FAST)", desc: "High-speed binary feature matcher." },
    { value: "direct", label: "Direct (Identity / No Warp)", desc: "Assumes images are already geometrically aligned; no transformation applied." },
];

export default function ImageComparison() {
    const [beforeFile, setBeforeFile] = useState(null);
    const [afterFile, setAfterFile] = useState(null);
    const [alignmentMode, setAlignmentMode] = useState("auto");

    const [validating, setValidating] = useState(false);
    const [validationResult, setValidationResult] = useState(null);

    const [comparing, setComparing] = useState(false);
    const [comparisonResult, setComparisonResult] = useState(null);
    const [error, setError] = useState(null);

    const abortControllerRef = useRef(null);

    const handleClear = () => {
        setBeforeFile(null);
        setAfterFile(null);
        setValidationResult(null);
        setComparisonResult(null);
        setError(null);
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            abortControllerRef.current = null;
        }
    };

    const handleValidate = async () => {
        if (!beforeFile || !afterFile) {
            setError("Please upload both before and after photographs to validate.");
            return;
        }

        setValidating(true);
        setError(null);
        setValidationResult(null);

        try {
            const result = await validateComparisonImages(beforeFile, afterFile);
            setValidationResult(result);
        } catch (err) {
            setError(parseApiError(err));
        } finally {
            setValidating(false);
        }
    };

    const handleCompare = async () => {
        if (!beforeFile || !afterFile) {
            setError("Please upload both before and after photographs before comparing.");
            return;
        }

        setComparing(true);
        setError(null);
        setComparisonResult(null);

        const controller = new AbortController();
        abortControllerRef.current = controller;

        try {
            const data = await compareInspectionImages(
                beforeFile,
                afterFile,
                alignmentMode,
                controller.signal
            );
            setComparisonResult(data);
        } catch (err) {
            if (err.name !== "CanceledError") {
                setError(parseApiError(err));
            }
        } finally {
            setComparing(false);
            abortControllerRef.current = null;
        }
    };

    // Format visual assets for viewer
    const viewerAssets = comparisonResult
        ? [
              {
                  id: "difference",
                  label: "SSIM Difference Heatmap",
                  src: comparisonResult.images?.difference,
                  description: "Pixel-level structural difference intensity (1.0 - SSIM)",
              },
              {
                  id: "highlighted",
                  label: "Highlighted Changes",
                  src: comparisonResult.images?.highlighted,
                  description: "Aligned photograph with red bounding boxes around changed regions",
              },
              {
                  id: "mask",
                  label: "Binary Change Mask",
                  src: comparisonResult.images?.mask,
                  description: "Thresholded and morphologically filtered binary change regions",
              },
              {
                  id: "aligned_after",
                  label: "Aligned Reference",
                  src: comparisonResult.images?.aligned_after,
                  description: "Photograph warped into common coordinate frame",
              },
              {
                  id: "before",
                  label: "Normalized Before",
                  src: comparisonResult.images?.before,
                  description: "Preprocessed before photograph",
              },
          ]
        : [];

    const alignment = comparisonResult?.alignment;

    return (
        <div>
            <PageHeader
                title="Before/After Image Comparison"
                description="Pairwise image registration using feature-based homography alignment (SIFT, AKAZE, ORB) and Structural Similarity Index (SSIM) difference analysis."
                badge="Task 2"
                icon={SplitSquareVertical}
                actions={
                    comparisonResult && (
                        <button
                            type="button"
                            onClick={handleClear}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-50"
                        >
                            <RotateCcw size={14} /> Reset Comparison
                        </button>
                    )
                }
            />

            {/* Scope Distinction Banner */}
            <ReliabilityNotice type="info" className="mb-6">
                <div>
                    <strong className="text-slate-900 font-semibold">General Visual Difference Scope:</strong> Task 2 measures surface-level photometric and structural divergence via SSIM. It flags general visual differences (weathering, paint discoloration, lighting variations) and does not perform crack-specific isolation. For crack progression, use <strong>Hybrid Progression</strong>.
                </div>
            </ReliabilityNotice>

            {/* Upload Section */}
            <div className="grid gap-6 md:grid-cols-2">
                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                    <ImageUpload
                        id="before-image-upload"
                        label="Before Inspection Photograph"
                        sublabel="Baseline reference image (JPEG, PNG, WebP)"
                        file={beforeFile}
                        onFileSelect={(f) => {
                            setBeforeFile(f);
                            setValidationResult(null);
                            setComparisonResult(null);
                            setError(null);
                        }}
                        onFileClear={() => {
                            setBeforeFile(null);
                            setValidationResult(null);
                            setComparisonResult(null);
                        }}
                        disabled={comparing || validating}
                        required
                    />
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                    <ImageUpload
                        id="after-image-upload"
                        label="After Inspection Photograph"
                        sublabel="Subsequent inspection image (JPEG, PNG, WebP)"
                        file={afterFile}
                        onFileSelect={(f) => {
                            setAfterFile(f);
                            setValidationResult(null);
                            setComparisonResult(null);
                            setError(null);
                        }}
                        onFileClear={() => {
                            setAfterFile(null);
                            setValidationResult(null);
                            setComparisonResult(null);
                        }}
                        disabled={comparing || validating}
                        required
                    />
                </div>
            </div>

            {/* Controls Row */}
            <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                    {/* Alignment Mode Selector */}
                    <div className="flex-1 max-w-lg">
                        <label
                            htmlFor="alignment-mode-select"
                            className="block text-xs font-semibold uppercase text-slate-500 mb-1.5"
                        >
                            Feature Alignment Method
                        </label>
                        <select
                            id="alignment-mode-select"
                            value={alignmentMode}
                            disabled={comparing}
                            onChange={(e) => setAlignmentMode(e.target.value)}
                            className="w-full rounded-xl border border-slate-300 bg-white px-3.5 py-2 text-xs font-medium text-slate-800 outline-none focus:border-blue-500"
                        >
                            {ALIGNMENT_MODES.map((m) => (
                                <option key={m.value} value={m.value}>
                                    {m.label}
                                </option>
                            ))}
                        </select>
                        <p className="mt-1 text-[11px] text-slate-400">
                            {ALIGNMENT_MODES.find((m) => m.value === alignmentMode)?.desc}
                        </p>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex flex-wrap items-center gap-3">
                        <button
                            type="button"
                            onClick={handleValidate}
                            disabled={!beforeFile || !afterFile || validating || comparing}
                            className="inline-flex items-center gap-1.5 rounded-xl border border-slate-300 bg-slate-50 px-4 py-2.5 text-xs font-semibold text-slate-700 shadow-2xs hover:bg-slate-100 disabled:opacity-50"
                        >
                            <CheckCircle2 size={15} />
                            {validating ? "Validating..." : "Validate Pair"}
                        </button>

                        <button
                            type="button"
                            onClick={handleCompare}
                            disabled={!beforeFile || !afterFile || comparing}
                            className="inline-flex items-center gap-2 rounded-xl bg-blue-600 px-6 py-2.5 text-xs font-semibold text-white shadow-xs hover:bg-blue-700 disabled:opacity-50"
                        >
                            <Sparkles size={16} />
                            {comparing ? "Aligning & Comparing..." : "Compare Images"}
                        </button>
                    </div>
                </div>

                {/* Pair Validation Result Notice */}
                {validationResult && (
                    <div className="mt-4 rounded-xl bg-emerald-50 p-3.5 text-xs text-emerald-900 border border-emerald-200 flex items-start gap-2">
                        <CheckCircle2 size={16} className="mt-0.5 text-emerald-600 shrink-0" />
                        <div>
                            <strong className="font-semibold">{validationResult.message}</strong>
                            <div className="mt-1 text-emerald-800 text-[11px]">
                                Before: {validationResult.before?.width} &times; {validationResult.before?.height} px ({validationResult.before?.channels} ch) &bull; 
                                After: {validationResult.after?.width} &times; {validationResult.after?.height} px ({validationResult.after?.channels} ch) &bull; 
                                Same Dimensions: {validationResult.same_dimensions ? "Yes" : "No (will be normalized)"}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Error Notification */}
            {error && (
                <div className="mt-6">
                    <ErrorState
                        error={error}
                        title="Image Comparison Failed"
                        onRetry={handleCompare}
                    />
                </div>
            )}

            {/* Loading Indicator */}
            {comparing && (
                <div className="mt-6">
                    <LoadingState
                        title="Executing Feature Alignment & SSIM Comparison..."
                        description={`Extracting keypoint correspondences with ${alignmentMode.toUpperCase()} and computing structural difference map.`}
                        onCancel={() => {
                            if (abortControllerRef.current) {
                                abortControllerRef.current.abort();
                                setComparing(false);
                            }
                        }}
                    />
                </div>
            )}

            {/* Comparison Results */}
            {comparisonResult && !comparing && (
                <div className="mt-8 space-y-6">
                    {/* Metrics Grid */}
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        <ResultCard
                            title="Mean SSIM Score"
                            value={comparisonResult.ssim_score.toFixed(4)}
                            subtitle="Structural similarity over valid overlap"
                            badge={
                                comparisonResult.ssim_score > 0.85
                                    ? "High Similarity"
                                    : comparisonResult.ssim_score > 0.65
                                    ? "Moderate Difference"
                                    : "High Difference"
                            }
                            badgeType={
                                comparisonResult.ssim_score > 0.85
                                    ? "success"
                                    : comparisonResult.ssim_score > 0.65
                                    ? "warning"
                                    : "danger"
                            }
                        />

                        <ResultCard
                            title="Changed Area Percentage"
                            value={`${comparisonResult.changed_percentage.toFixed(2)}%`}
                            subtitle="Percentage of valid overlap exhibiting visual change"
                            valueColor="text-blue-600"
                        />

                        <ResultCard
                            title="Changed Regions Count"
                            value={comparisonResult.changed_regions_count}
                            subtitle="Connected contours exceeding min area"
                        />

                        {(() => {
                            const isDirect =
                                alignment?.status === "direct" ||
                                alignment?.method === "DIRECT" ||
                                alignment?.method_used === "DIRECT" ||
                                alignment?.geometric_status === "ASSUMED_UNVERIFIED";
                            const isIdentical =
                                alignment?.status === "identical" ||
                                alignment?.geometric_status === "IDENTICAL_INPUTS";

                            const qualValue = isDirect
                                ? "N/A"
                                : isIdentical
                                ? "100.0%"
                                : alignment?.confidence != null
                                ? `${(alignment.confidence * 100).toFixed(1)}%`
                                : "N/A";

                            const qualSubtitle = isDirect
                                ? "Direct mode (unverified)"
                                : isIdentical
                                ? "Byte-identical images"
                                : `Detector: ${alignment?.detector || alignment?.method_used || alignment?.method || alignmentMode.toUpperCase()}`;

                            const qualBadge = isDirect
                                ? "Unverified"
                                : isIdentical
                                ? "Identical"
                                : "Verified";

                            const qualBadgeType = isDirect ? "neutral" : "success";

                            return (
                                <ResultCard
                                    title={isDirect ? "Geometric Registration" : "Alignment Quality"}
                                    value={qualValue}
                                    subtitle={qualSubtitle}
                                    badge={qualBadge}
                                    badgeType={qualBadgeType}
                                />
                            );
                        })()}
                    </div>

                    {/* Detailed Alignment Telemetry Card */}
                    {alignment && (() => {
                        const isDirect =
                            alignment?.status === "direct" ||
                            alignment?.method === "DIRECT" ||
                            alignment?.method_used === "DIRECT" ||
                            alignment?.geometric_status === "ASSUMED_UNVERIFIED";
                        const isIdentical =
                            alignment?.status === "identical" ||
                            alignment?.geometric_status === "IDENTICAL_INPUTS";

                        return (
                            <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                                <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
                                    <h3 className="text-xs font-semibold uppercase text-slate-500">
                                        Geometric Alignment Telemetry
                                    </h3>
                                    <span
                                        className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold border ${
                                            isDirect
                                                ? "bg-slate-100 text-slate-700 border-slate-200"
                                                : isIdentical
                                                ? "bg-blue-50 text-blue-700 border-blue-200"
                                                : "bg-emerald-50 text-emerald-700 border-emerald-200"
                                        }`}
                                    >
                                        {isDirect
                                            ? "Assumed / Not Geometrically Verified"
                                            : isIdentical
                                            ? "Byte-Identical Inputs"
                                            : "Feature-based Registration Accepted"}
                                    </span>
                                </div>

                                <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4 text-xs">
                                    <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                                        <span className="text-slate-500">Method Used:</span>
                                        <p className="mt-0.5 font-mono font-bold text-slate-800">
                                            {isDirect ? "DIRECT" : (alignment.detector || alignment.method_used || alignment.method || alignmentMode.toUpperCase())}
                                        </p>
                                    </div>
                                    <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                                        <span className="text-slate-500">Inliers / Matches:</span>
                                        <p className="mt-0.5 font-mono font-bold text-slate-800">
                                            {isDirect ? (
                                                "N/A"
                                            ) : alignment.inliers != null ? (
                                                <>
                                                    {alignment.inliers} / {alignment.matches ?? "N/A"}{" "}
                                                    {alignment.inlier_ratio != null && (
                                                        <span className="text-slate-500 font-normal">
                                                            ({(alignment.inlier_ratio * 100).toFixed(1)}%)
                                                        </span>
                                                    )}
                                                </>
                                            ) : (
                                                "N/A"
                                            )}
                                        </p>
                                    </div>
                                    <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                                        <span className="text-slate-500">Reprojection Error:</span>
                                        <p className="mt-0.5 font-mono font-bold text-slate-800">
                                            {isDirect ? (
                                                "N/A"
                                            ) : alignment.reprojection_error != null ? (
                                                `${alignment.reprojection_error.toFixed(2)} px`
                                            ) : (
                                                "N/A"
                                            )}
                                        </p>
                                    </div>
                                    <div className="rounded-xl bg-slate-50 p-3 border border-slate-200/80">
                                        <span className="text-slate-500">Valid Overlap:</span>
                                        <p className="mt-0.5 font-mono font-bold text-slate-800">
                                            {alignment.overlap_ratio != null
                                                ? `${(alignment.overlap_ratio * 100).toFixed(1)}%`
                                                : "100.0%"}
                                        </p>
                                    </div>
                                </div>

                                {/* Direct Alignment Unverified Warning Banner */}
                                {isDirect && (
                                    <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50/80 p-4 text-xs text-amber-900">
                                        <div className="flex items-start gap-2.5">
                                            <AlertTriangle size={16} className="mt-0.5 shrink-0 text-amber-600" />
                                            <div>
                                                <strong className="font-semibold text-amber-950">
                                                    Direct Alignment Notice:
                                                </strong>{" "}
                                                Direct mode skips geometric registration. Results assume that corresponding pixels represent the same scene locations. Resizing or normalization does not independently verify alignment.
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Dimension Normalization Telemetry / Warnings */}
                                {alignment.dimension_normalized && (
                                    <div className="mt-3 rounded-xl border border-blue-200 bg-blue-50/70 p-3.5 text-xs text-blue-900">
                                        <div className="flex items-start gap-2">
                                            <Info size={15} className="mt-0.5 shrink-0 text-blue-600" />
                                            <div>
                                                <span className="font-semibold text-blue-950">
                                                    Source Dimension Normalization:
                                                </span>{" "}
                                                {alignment.dimension_details ? (
                                                    <span>
                                                        Before photograph was {alignment.dimension_details.before_source?.width}&times;{alignment.dimension_details.before_source?.height} px, 
                                                        After photograph was {alignment.dimension_details.after_source?.width}&times;{alignment.dimension_details.after_source?.height} px. 
                                                        Normalized to reference frame ({alignment.dimension_details.processing_dimensions?.width}&times;{alignment.dimension_details.processing_dimensions?.height} px).
                                                    </span>
                                                ) : (
                                                    <span>Source photographs had different original dimensions and were scaled to the reference coordinate frame.</span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {/* Alignment Warnings List if any additional */}
                                {alignment.warnings && alignment.warnings.length > 0 && (
                                    <div className="mt-3 space-y-1.5">
                                        {alignment.warnings
                                            .filter(w => !isDirect || (!w.includes("geometric alignment was not verified") && !w.includes("different dimensions")))
                                            .map((warn, idx) => (
                                                <div key={idx} className="flex items-center gap-2 text-xs text-amber-800 bg-amber-50/50 rounded-lg p-2 border border-amber-100">
                                                    <AlertTriangle size={14} className="shrink-0 text-amber-600" />
                                                    <span>{warn}</span>
                                                </div>
                                            ))}
                                    </div>
                                )}
                            </div>
                        );
                    })()}

                    {/* Visual Comparison Viewer */}
                    <ImageComparisonViewer
                        assets={viewerAssets}
                        defaultAssetId="difference"
                        beforeImageSrc={comparisonResult.images?.before}
                        afterImageSrc={comparisonResult.images?.aligned_after}
                        showSlider={true}
                        showLegend={false}
                    />

                    {/* Detected Regions Table */}
                    {comparisonResult.regions && comparisonResult.regions.length > 0 && (
                        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs overflow-hidden">
                            <h3 className="text-sm font-semibold text-slate-900 mb-3">
                                Detected Change Regions ({comparisonResult.regions.length})
                            </h3>
                            <div className="overflow-x-auto">
                                <table className="w-full text-left text-xs">
                                    <thead className="border-b border-slate-100 bg-slate-50 text-slate-500 font-semibold uppercase">
                                        <tr>
                                            <th className="py-2.5 px-3">#</th>
                                            <th className="py-2.5 px-3">Position (X, Y)</th>
                                            <th className="py-2.5 px-3">Dimensions (W &times; H)</th>
                                            <th className="py-2.5 px-3">Area (Pixels)</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-slate-100 text-slate-700">
                                        {comparisonResult.regions.slice(0, 15).map((r, i) => (
                                            <tr key={i} className="hover:bg-slate-50/60">
                                                <td className="py-2.5 px-3 font-mono font-medium">{i + 1}</td>
                                                <td className="py-2.5 px-3 font-mono">({r.x}, {r.y})</td>
                                                <td className="py-2.5 px-3 font-mono">{r.width} &times; {r.height} px</td>
                                                <td className="py-2.5 px-3 font-mono font-semibold text-slate-900">
                                                    {r.area_pixels.toLocaleString()} px
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                            {comparisonResult.regions.length > 15 && (
                                <p className="mt-3 text-center text-xs text-slate-400">
                                    Showing top 15 regions sorted by area of {comparisonResult.regions.length} total detected regions.
                                </p>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
