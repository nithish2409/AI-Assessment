import { useNavigate } from "react-router-dom";
import {
    SplitSquareVertical,
    GitCompareArrows,
    Newspaper,
    Building2,
    ArrowRight,
    CheckCircle2,
} from "lucide-react";
import PageHeader from "../components/PageHeader";
import SystemHealthIndicator from "../components/SystemHealthIndicator";

export default function DashboardOverview() {
    const navigate = useNavigate();

    const tasks = [
        {
            id: "inspection",
            title: "Task 1: AI Damage Detection & Visual Inspection",
            tag: "Building & Industrial Inspection",
            route: "/inspection",
            icon: Building2,
            model: "YOLO + CODEBRIM + EasyOCR",
            endpoint: "POST /api/inspection/analyze",
            description:
                "AI-assisted visual inspections of building structures and industrial equipment. Detects visible damage, identifies components, and provides visual severity estimates with annotated findings and downloadable reports.",
            capabilities: [
                "Building structural inspection: cracks, spallation, exposed rebar, and efflorescence",
                "Industrial equipment inspection: corrosion, weld cracks, and bolt slippage",
                "Damage segmentation, component detection, and equipment nameplate OCR",
                "Visual severity estimates, annotated overlays, and PDF / JSON reports",
            ],
            notes: "Visual screening supports inspection review and does not certify structural or equipment safety.",
        },
        {
            id: "image-comparison",
            title: "Task 2: Before/After Image Comparison",
            tag: "Image Registration & SSIM",
            route: "/image-comparison",
            icon: SplitSquareVertical,
            model: "SIFT / AKAZE / ORB + SSIM",
            endpoint: "POST /api/comparison/compare",
            description:
                "Pairwise geometric registration and visual change detection between longitudinal inspection photographs. Aligns perspective variations with RANSAC homography and computes Structural Similarity Index (SSIM) over the valid overlap region.",
            capabilities: [
                "Multi-method feature detectors: SIFT, AKAZE, ORB, and Direct",
                "Robust RANSAC homography estimation with outlier rejection",
                "Valid overlap region calculation and masking",
                "SSIM visual difference intensity map and bounding box regions",
            ],
            notes: "Detects general surface and visual differences; does not distinguish crack progression from non-crack surface changes.",
        },
        {
            id: "news-classification",
            title: "Task 3: NLP News Classification",
            tag: "Natural Language Processing",
            route: "/news-classification",
            icon: Newspaper,
            model: "TF-IDF + Logistic Regression (AG News)",
            endpoint: "POST /api/nlp/classify",
            description:
                "Multiclass text classification of news headlines and articles into four canonical categories: World, Sports, Business, and Sci/Tech. Includes feature interpretation tools such as Bag-of-Words and cosine similarity.",
            capabilities: [
                "Four-class AG News categorical inference",
                "Calibrated class probability distributions",
                "Top TF-IDF and Bag-of-Words n-gram feature extraction",
                "Pairwise cosine text similarity analysis",
            ],
            notes: "Preserves the existing trained model and feature engineering pipeline.",
        },
        {
            id: "hybrid-analysis",
            title: "Hybrid Crack Progression Analysis",
            tag: "Additional Feature",
            route: "/hybrid-analysis",
            icon: GitCompareArrows,
            featured: true,
            model: "Homography Alignment + YOLO26s-seg + SSIM + Spatial Tolerance",
            endpoint: "POST /api/comparison/hybrid",
            description:
                "Synergistic integration of geometric alignment and YOLO segmentation to quantify true crack evolution across inspections. Employs spatial tolerance buffering and connected component analysis to separate genuine crack progression from alignment shifts.",
            capabilities: [
                "Warped before-crack mask registration into after-image reference frame",
                "Pixel classification: persistent, newly observed, and no-longer-observed cracks",
                "Spatial tolerance buffering (0–10 px) to absorb registration jitter",
                "Component progression analysis (potential new vs expanded cracks)",
                "Provisional calibrated noise floor filtering (0.89 pp) with uncertainty level",
                "11-asset visual inspection viewer with color-coded overlay",
            ],
            notes: "Operating threshold of 0.89 pp is provisional from controlled experiments. Does not guarantee safety or structural growth.",
        },
    ];

    return (
        <div>
            <PageHeader
                title="AI Assessment Dashboard"
                description="AI-assisted building and industrial inspections, before/after image comparison, NLP news classification, and additional hybrid crack progression analysis."
                badge="Assessment Tasks"
            />

            {/* Live System Connectivity Strip */}
            <div className="mb-8">
                <SystemHealthIndicator showDetails={true} />
            </div>

            {/* Tasks Grid */}
            <div className="grid gap-6 md:grid-cols-2">
                {tasks.map((task) => {
                    const Icon = task.icon;
                    return (
                        <div
                            key={task.id}
                            className={`flex flex-col justify-between rounded-2xl border bg-white p-6 shadow-xs transition hover:shadow-sm ${
                                task.featured
                                    ? "border-blue-300 ring-2 ring-blue-500/20"
                                    : "border-slate-200"
                            }`}
                        >
                            <div>
                                <div className="flex items-start justify-between gap-3">
                                    <div className="flex items-center gap-3">
                                        <div
                                            className={`rounded-xl p-2.5 ${
                                                task.featured
                                                    ? "bg-blue-600 text-white"
                                                    : "bg-slate-100 text-slate-700"
                                            }`}
                                        >
                                            <Icon size={22} />
                                        </div>
                                        <div>
                                            <span className="text-[10px] font-bold uppercase tracking-wider text-blue-600">
                                                {task.tag}
                                            </span>
                                            <h2 className="text-base font-bold text-slate-900">
                                                {task.title}
                                            </h2>
                                        </div>
                                    </div>
                                    {task.featured && (
                                        <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-[11px] font-semibold text-blue-800 border border-blue-200">
                                            Additional Feature
                                        </span>
                                    )}
                                </div>

                                <p className="mt-3 text-xs text-slate-600 leading-relaxed">
                                    {task.description}
                                </p>

                                <div className="mt-4 rounded-xl bg-slate-50 p-3 text-xs border border-slate-200/80 space-y-1">
                                    <div className="flex items-center justify-between text-slate-500">
                                        <span>Model:</span>
                                        <span className="font-mono font-semibold text-slate-800">
                                            {task.model}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between text-slate-500">
                                        <span>Endpoint:</span>
                                        <span className="font-mono text-[11px] text-slate-700">
                                            {task.endpoint}
                                        </span>
                                    </div>
                                </div>

                                <div className="mt-4">
                                    <h3 className="text-xs font-semibold uppercase text-slate-500 mb-2">
                                        Key Capabilities
                                    </h3>
                                    <ul className="space-y-1.5 text-xs text-slate-600">
                                        {task.capabilities.map((c, i) => (
                                            <li key={i} className="flex items-start gap-2">
                                                <CheckCircle2
                                                    size={14}
                                                    className="mt-0.5 shrink-0 text-emerald-600"
                                                />
                                                <span>{c}</span>
                                            </li>
                                        ))}
                                    </ul>
                                </div>

                                {task.notes && (
                                    <p className="mt-4 rounded-lg bg-amber-50/70 p-2.5 text-[11px] text-amber-900 border border-amber-200/70">
                                        <strong className="font-semibold">Notice:</strong> {task.notes}
                                    </p>
                                )}
                            </div>

                            <div className="mt-6 pt-4 border-t border-slate-100">
                                <button
                                    type="button"
                                    onClick={() => navigate(task.route)}
                                    className={`flex w-full items-center justify-center gap-2 rounded-xl py-2.5 text-xs font-semibold transition ${
                                        task.featured
                                            ? "bg-blue-600 text-white hover:bg-blue-700 shadow-xs"
                                            : "bg-slate-900 text-white hover:bg-slate-800"
                                    }`}
                                >
                                    Open {task.title.split(":")[1]?.trim() || task.title}
                                    <ArrowRight size={14} />
                                </button>
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
