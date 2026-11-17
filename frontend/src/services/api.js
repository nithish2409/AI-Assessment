import axios from "axios";

const BASE_URL =
    (typeof import.meta !== "undefined" && import.meta.env?.VITE_API_URL) ||
    (typeof process !== "undefined" && process.env?.VITE_API_URL) ||
    "http://localhost:8000/api";

const api = axios.create({
    baseURL: BASE_URL,
    timeout: 30000,
});

// --------------------------------------
// Structured Error Parser
// --------------------------------------

function formatErrorTitle(code, status) {
    if (code === "ALIGNMENT_FAILED") return "Image Alignment Failed";
    if (code === "DAMAGE_SERVICE_UNAVAILABLE") return "Damage Service Unavailable";
    if (code === "DAMAGE_SERVICE_TIMEOUT") return "Inference Timeout";
    if (code === "INSUFFICIENT_OVERLAP") return "Insufficient Overlap";
    if (code === "INVALID_ALIGNMENT_MODE") return "Invalid Alignment Mode";
    if (code === "INVALID_IMAGE_TYPE") return "Invalid Image Format";
    if (code === "EMPTY_IMAGE") return "Empty Image Upload";
    if (code === "IMAGE_TOO_LARGE") return "Image File Too Large";
    if (code === "COMPARISON_VALIDATION_ERROR") return "Comparison Validation Error";
    if (code && typeof code === "string") {
        return code
            .toLowerCase()
            .split("_")
            .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
            .join(" ");
    }
    return status ? `HTTP ${status} Error` : "Operation Failed";
}

export const parseApiError = (error) => {
    if (!error) {
        return {
            title: "Unknown Error",
            message: "An unknown error occurred.",
            code: "UNKNOWN",
            guidance: null,
            failureReasons: [],
            attemptedMethods: [],
            __isParsedApiError: true,
        };
    }

    // 1. Idempotency: If error was already parsed by parseApiError, return as-is
    if (
        error.__isParsedApiError ||
        (error.code && error.title && error.message && !error.isAxiosError && !error.response)
    ) {
        return error;
    }

    // 2. Client-side Axios timeout
    if (error.code === "ECONNABORTED" || error.message?.includes("timeout")) {
        return {
            title: "Request Timeout",
            message: "The server took too long to respond. The model inference or alignment may be processing a heavy image.",
            code: "TIMEOUT",
            guidance: "Please try with a smaller image or retry after a moment.",
            failureReasons: ["Client-side request timeout limit reached."],
            attemptedMethods: [],
            isTimeout: true,
            __isParsedApiError: true,
        };
    }

    // 3. Genuine Network Disconnection (Strictly when NO response was received)
    if (!error.response) {
        return {
            title: "Network Connection Failed",
            message: error.message || "Unable to communicate with the FastAPI backend.",
            code: "NETWORK_ERROR",
            guidance: "Ensure the FastAPI backend is running on http://127.0.0.1:8000.",
            failureReasons: ["No HTTP response received from server."],
            attemptedMethods: [],
            isUnavailable: true,
            __isParsedApiError: true,
        };
    }

    // 4. Server returned an HTTP response
    const { status, data } = error.response;
    const detail = data?.detail;

    // 4A. Structured object error (e.g. detail.error, detail.code, detail.error_code)
    if (detail && typeof detail === "object" && !Array.isArray(detail)) {
        const rawCode =
            detail.error ||
            detail.code ||
            detail.error_code ||
            data?.error ||
            data?.code ||
            data?.error_code ||
            (status === 422 ? "VALIDATION_ERROR" : `HTTP_${status}`);

        const message =
            detail.message ||
            detail.msg ||
            data?.message ||
            data?.msg ||
            "An error occurred during request processing.";

        const guidance = detail.guidance || data?.guidance || null;
        const failureReasons = detail.failure_reasons || detail.failureReasons || data?.failure_reasons || [];
        const attemptedMethods = detail.attempted_methods || detail.attemptedMethods || data?.attempted_methods || [];

        return {
            title: formatErrorTitle(rawCode, status),
            message,
            code: rawCode,
            guidance,
            failureReasons,
            attemptedMethods,
            isUnavailable: status === 503,
            isTimeout: status === 504,
            status,
            __isParsedApiError: true,
        };
    }

    // 4B. FastAPI validation error array [{ loc, msg, type }]
    if (Array.isArray(detail)) {
        const msgs = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
        const failureReasons = detail.map((d) => `${d.loc ? d.loc.join(".") + ": " : ""}${d.msg}`);
        return {
            title: "Validation Error",
            message: msgs,
            code: "VALIDATION_ERROR",
            guidance: "Review the uploaded parameters and image formats.",
            failureReasons,
            attemptedMethods: [],
            status,
            __isParsedApiError: true,
        };
    }

    // 4C. Simple string detail
    if (typeof detail === "string") {
        const isAlignment = detail.toLowerCase().includes("alignment failed");
        const code = isAlignment ? "ALIGNMENT_FAILED" : `HTTP_${status}`;
        const title = isAlignment ? "Image Alignment Failed" : status >= 500 ? "Server Error" : "Request Rejected";
        const guidance = isAlignment
            ? "Ensure both images show the same scene from a similar viewpoint with visible shared details. If the images are already pixel-aligned, you may use alignment_mode='direct'."
            : null;
        const failureReasons = isAlignment
            ? ["Insufficient matching features between the two images."]
            : [];
        const attemptedMethods = isAlignment ? ["SIFT", "AKAZE", "ORB"] : [];

        return {
            title,
            message: detail,
            code,
            guidance,
            failureReasons,
            attemptedMethods,
            status,
            __isParsedApiError: true,
        };
    }

    // 4D. Fallback for unexpected response structures
    const rawCode = data?.error || data?.code || data?.error_code || `HTTP_${status}`;
    return {
        title: formatErrorTitle(rawCode, status),
        message: data?.message || error.message || "Unexpected server response.",
        code: rawCode,
        guidance: data?.guidance || null,
        failureReasons: data?.failure_reasons || [],
        attemptedMethods: data?.attempted_methods || [],
        status,
        __isParsedApiError: true,
    };
};

export const getErrorMessage = (error) => {
    const parsed = parseApiError(error);
    if (parsed.guidance) {
        return `${parsed.message} — ${parsed.guidance}`;
    }
    return parsed.message;
};

// --------------------------------------
// Health & Diagnostics
// --------------------------------------

export const checkHealth = async () =>
    (await api.get("/health")).data;

export const checkDamageHealth = async () =>
    (await api.get("/damage/health")).data;

export const checkNlpStatus = async () =>
    (await api.get("/nlp/status")).data;

export const getModelMetrics = async () =>
    (await api.get("/nlp/metrics")).data;

// --------------------------------------
// Task 2: Before/After Image Comparison
// --------------------------------------

export const validateComparisonImages = async (
    beforeFile,
    afterFile,
    signal = null
) => {
    const formData = new FormData();
    formData.append("before", beforeFile);
    formData.append("after", afterFile);

    const response = await api.post("/comparison/validate", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
        signal,
    });
    return response.data;
};

export const compareInspectionImages = async (
    beforeFile,
    afterFile,
    alignmentMode = "auto",
    signal = null
) => {
    const formData = new FormData();
    formData.append("before", beforeFile);
    formData.append("after", afterFile);
    formData.append("alignment_mode", alignmentMode);

    const response = await api.post("/comparison/compare", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
        signal,
    });
    return response.data;
};

// --------------------------------------
// Hybrid Damage Comparison
// --------------------------------------

export const compareHybridImages = async (
    beforeFile,
    afterFile,
    {
        alignmentMode = "auto",
        confidence = 0.25,
        iou = 0.45,
        tolerancePx = 2,
    } = {},
    signal = null
) => {
    const formData = new FormData();
    formData.append("before", beforeFile);
    formData.append("after", afterFile);
    formData.append("alignment_mode", alignmentMode);
    formData.append("confidence", String(confidence));
    formData.append("iou", String(iou));
    formData.append("tolerance_px", String(tolerancePx));

    const response = await api.post("/comparison/hybrid", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
        signal,
    });
    return response.data;
};

// --------------------------------------
// Task 3: NLP News Classification
// --------------------------------------

export const classifyNews = async (text, signal = null) =>
    (await api.post("/nlp/classify", { text }, { signal })).data;

export const analyzeNews = async (text, top_n = 10, signal = null) =>
    (await api.post("/nlp/analyze", { text, top_n }, { signal })).data;

export const extractTfidf = async (text, top_n = 10) =>
    (await api.post("/nlp/tfidf", { text, top_n })).data;

export const extractBow = async (text, top_n = 10) =>
    (await api.post("/nlp/bow", { text, top_n })).data;

export const extractNgrams = async (text, n = 2, top_n = 10) =>
    (await api.post("/nlp/ngrams", { text, n, top_n })).data;

export const extractKeywords = async (text, top_n = 10) =>
    (await api.post("/nlp/keywords", { text, top_n })).data;

export const selectFeatures = async (category, top_n = 10) =>
    (await api.post("/nlp/features", { category, top_n })).data;

export const compareTexts = async (text1, text2) =>
    (await api.post("/nlp/similarity", { text1, text2 })).data;

// --------------------------------------
// AI Inspection & Visual Severity
// --------------------------------------

export const analyzeInspection = async (formData, signal = null) => {
    const response = await api.post("/inspection/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 90000,
        signal,
    });
    return response.data;
};

export const downloadInspectionPdf = async (analysisData) => {
    const response = await api.post("/inspection/report/pdf", analysisData, {
        responseType: "blob",
        timeout: 60000,
    });
    return response.data;
};

export default api;
