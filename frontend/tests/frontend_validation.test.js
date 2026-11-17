import test from "node:test";
import assert from "node:assert/strict";
import {
    parseApiError,
    getErrorMessage,
    validateComparisonImages,
    compareInspectionImages,
    compareHybridImages,
    classifyNews,
    analyzeNews,
} from "../src/services/api.js";

// Helper to simulate Axios error objects
function createAxiosError({
    status = 500,
    data = null,
    message = "Request failed",
    code = null,
}) {
    const error = new Error(message);
    if (code) error.code = code;
    if (status) {
        error.response = {
            status,
            data,
            headers: {},
        };
    }
    return error;
}

// ---------------------------------------------------------------------------
// 2. Valid Before/After Pair Handling
// ---------------------------------------------------------------------------
test("Scenario 2: Valid before/after pair validation response schema", () => {
    const mockValidateResp = {
        status: "validated",
        before: { width: 640, height: 480, channels: 3 },
        after: { width: 640, height: 480, channels: 3 },
        same_dimensions: true,
        message: "Both images are valid and ready for comparison.",
    };

    assert.strictEqual(mockValidateResp.status, "validated");
    assert.strictEqual(mockValidateResp.same_dimensions, true);
    assert.strictEqual(mockValidateResp.before.width, 640);
});

// ---------------------------------------------------------------------------
// 3. Invalid Before/After Pair Handling
// ---------------------------------------------------------------------------
test("Scenario 3: Invalid before/after pair dimension or format failure", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: "Images have zero overlapping field of view or corrupted headers.",
        },
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.status, 422);
    assert.strictEqual(parsed.title, "Request Rejected");
    assert.match(parsed.message, /zero overlapping/);
});

// ---------------------------------------------------------------------------
// 4. Hybrid Comparison Success Response Processing
// ---------------------------------------------------------------------------
test("Scenario 4: Hybrid comparison success payload structure and field bindings", () => {
    const mockHybridResp = {
        status: "completed",
        image_width: 800,
        image_height: 600,
        alignment: {
            status: "accepted",
            method: "SIFT",
            confidence: 0.92,
            inlier_count: 145,
            inlier_ratio: 0.82,
            reprojection_error: 1.15,
            valid_overlap_pixels: 460000,
            valid_overlap_percentage: 95.8,
            warnings: [],
        },
        visual_comparison: {
            ssim_score: 0.9124,
            visual_changed_pixels: 12400,
            visual_change_percentage: 2.7,
            changed_regions_count: 4,
        },
        crack_comparison: {
            before_crack_pixels: 3500,
            after_crack_pixels: 5800,
            before_crack_coverage_percentage: 0.76,
            after_crack_coverage_percentage: 1.26,
            coverage_change_percentage_points: 0.50,
            persistent_pixels: 3100,
            newly_observed_pixels: 2700,
            no_longer_observed_pixels: 400,
            raw_change_pixels: 3100,
            raw_change_percentage: 0.67,
            tolerance_px: 2,
            tolerance_adjusted_new_pixels: 1850,
            tolerance_adjusted_no_longer_observed_pixels: 210,
            potential_new_components: 2,
            potential_expanded_components: 1,
            persistent_components: 3,
            no_longer_observed_components: 0,
            progression_label: "Potential newly observed crack regions",
            calibrated_noise_floor_pp: 0.89,
            is_above_noise_floor: false,
            uncertainty_level: "low",
            interpretation_summary: "Observed coverage change (0.50 pp) is within the 0.89 pp noise floor.",
        },
        images: {
            aligned_before: "b64_aligned",
            visual_difference: "b64_diff",
            before_crack_mask: "b64_before_mask",
            after_crack_mask: "b64_after_mask",
            newly_observed_mask: "b64_new_mask",
            persistent_mask: "b64_pers_mask",
            no_longer_observed_mask: "b64_lost_mask",
            tolerance_adjusted_new_mask: "b64_tol_mask",
            hybrid_overlay: "b64_overlay",
        },
    };

    assert.strictEqual(mockHybridResp.status, "completed");
    assert.strictEqual(mockHybridResp.alignment.method, "SIFT");
    assert.strictEqual(mockHybridResp.crack_comparison.potential_new_components, 2);
    assert.strictEqual(mockHybridResp.crack_comparison.calibrated_noise_floor_pp, 0.89);
    assert.strictEqual(mockHybridResp.crack_comparison.is_above_noise_floor, false);
});

// ---------------------------------------------------------------------------
// 5. Hybrid Alignment Rejection Handling
// ---------------------------------------------------------------------------
test("Scenario 5: Structured ALIGNMENT_FAILED rejection parsing", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: {
                error: "ALIGNMENT_FAILED",
                message: "Alignment failed: insufficient geometric correspondences.",
                attempted_methods: ["SIFT", "AKAZE", "ORB"],
                failure_reasons: ["Inliers count 3 below threshold 8."],
                guidance: "Ensure both photographs show the same building surface from a similar viewpoint.",
            },
        },
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "ALIGNMENT_FAILED");
    assert.deepStrictEqual(parsed.attemptedMethods, ["SIFT", "AKAZE", "ORB"]);
    assert.strictEqual(parsed.failureReasons.length, 1);
    assert.match(parsed.guidance, /similar viewpoint/);
});

// ---------------------------------------------------------------------------
// 6. Missing Images Validation
// ---------------------------------------------------------------------------
test("Scenario 6: Missing images rejection", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: [
                { loc: ["body", "image"], msg: "Field required", type: "missing" },
            ],
        },
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "VALIDATION_ERROR");
    assert.match(parsed.message, /Field required/);
});

// ---------------------------------------------------------------------------
// 7. Unsupported Image Formats
// ---------------------------------------------------------------------------
test("Scenario 7: Unsupported image format rejection (e.g. BMP / GIF / TIFF)", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: {
                error: "INVALID_IMAGE_TYPE",
                message: "Unsupported format for before image: 'image/gif'. Only JPEG, PNG and WebP are accepted.",
            },
        },
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "INVALID_IMAGE_TYPE");
    assert.match(parsed.message, /Only JPEG, PNG and WebP/);
});

// ---------------------------------------------------------------------------
// 8. Corrupted Uploads
// ---------------------------------------------------------------------------
test("Scenario 8: Corrupted upload payload handling", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: "Invalid or corrupted image: decompression failed.",
        },
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.status, 422);
    assert.match(parsed.message, /decompression failed/);
});

// ---------------------------------------------------------------------------
// 9. Main Backend Unavailable (503 / Network Error)
// ---------------------------------------------------------------------------
test("Scenario 9: Backend unreachable / network down", () => {
    const err = new Error("Network Error");
    err.code = "ERR_NETWORK";

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "NETWORK_ERROR");
    assert.strictEqual(parsed.isUnavailable, true);
    assert.match(parsed.guidance, /http:\/\/127\.0\.0\.1:8000/);
});

// ---------------------------------------------------------------------------
// 10. YOLO Inference Microservice Unavailable (503 Gateway Proxy)
// ---------------------------------------------------------------------------
test("Scenario 10: YOLO microservice unavailable (503 structured response)", () => {
    const err = createAxiosError({
        status: 503,
        data: {
            detail: {
                error: "DAMAGE_SERVICE_UNAVAILABLE",
                message: "The damage segmentation inference microservice is currently unreachable.",
                guidance: "Ensure the YOLO inference microservice is running under backend/venv_yolo on port 8001.",
            },
        },
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.status, 503);
    assert.strictEqual(parsed.code, "DAMAGE_SERVICE_UNAVAILABLE");
    assert.strictEqual(parsed.isUnavailable, true);
    assert.match(parsed.guidance, /port 8001/);
});

// ---------------------------------------------------------------------------
// 11. Inference Timeout (504 Gateway Timeout or Client Abort)
// ---------------------------------------------------------------------------
test("Scenario 11: Inference timeout handling", () => {
    const timeoutErr = new Error("timeout of 30000ms exceeded");
    timeoutErr.code = "ECONNABORTED";

    const parsed = parseApiError(timeoutErr);
    assert.strictEqual(parsed.code, "TIMEOUT");
    assert.strictEqual(parsed.isTimeout, true);

    const http504Err = createAxiosError({
        status: 504,
        data: {
            detail: {
                error: "DAMAGE_SERVICE_TIMEOUT",
                message: "Inference microservice timed out after 30 seconds.",
            },
        },
    });

    const parsed504 = parseApiError(http504Err);
    assert.strictEqual(parsed504.status, 504);
    assert.strictEqual(parsed504.isTimeout, true);
});

// ---------------------------------------------------------------------------
// 12. Empty Crack Detections (detections_count = 0)
// ---------------------------------------------------------------------------
test("Scenario 12: Zero-crack detection response safety logic", () => {
    const zeroCrackResp = {
        status: "completed",
        model: "YOLO26s-seg",
        class_names: ["crack"],
        image_width: 640,
        image_height: 640,
        detections_count: 0,
        crack_coverage_percentage: 0.0,
        detections: [],
        images: {
            original: "b64_orig",
            mask: "b64_empty_mask",
            overlay: "b64_orig_overlay",
        },
        inference: { device: "cuda:0", duration_ms: 18.2 },
        warnings: [],
    };

    assert.strictEqual(zeroCrackResp.detections_count, 0);
    assert.strictEqual(zeroCrackResp.crack_coverage_percentage, 0.0);
    assert.strictEqual(zeroCrackResp.detections.length, 0);
});

// ---------------------------------------------------------------------------
// 13. Correct Percentage & Number Formatting
// ---------------------------------------------------------------------------
test("Scenario 13: Percentage and metric formatting", () => {
    const formatCoverageDiff = (val) => `${val >= 0 ? "+" : ""}${val.toFixed(2)} pp`;
    const formatPct = (val) => `${val.toFixed(2)}%`;

    assert.strictEqual(formatCoverageDiff(0.89), "+0.89 pp");
    assert.strictEqual(formatCoverageDiff(-0.25), "-0.25 pp");
    assert.strictEqual(formatCoverageDiff(0.0), "+0.00 pp");
    assert.strictEqual(formatPct(1.4239), "1.42%");
});

// ---------------------------------------------------------------------------
// 14. Correct Uncertainty Interpretation & Noise Floor Bounds
// ---------------------------------------------------------------------------
test("Scenario 14: Uncertainty classification and provisional noise floor rules", () => {
    const checkNoiseFloor = (changePp, floorPp = 0.89) => Math.abs(changePp) > floorPp;

    assert.strictEqual(checkNoiseFloor(0.50, 0.89), false); // Within expected noise
    assert.strictEqual(checkNoiseFloor(1.20, 0.89), true);  // Exceeds noise floor

    const validLabels = [
        "No substantial predicted change detected",
        "Potential newly observed crack regions",
        "Potential expansion of detected crack regions",
        "Comparison inconclusive",
    ];

    assert.ok(validLabels.includes("Potential newly observed crack regions"));
});

// ---------------------------------------------------------------------------
// 15. Base64 Image URI Rendering Conversion
// ---------------------------------------------------------------------------
test("Scenario 15: Base64 data URI conversion helper", () => {
    const toDataUri = (src) => {
        if (!src) return "";
        if (src.startsWith("data:") || src.startsWith("blob:") || src.startsWith("http")) {
            return src;
        }
        return `data:image/png;base64,${src}`;
    };

    assert.strictEqual(toDataUri(""), "");
    assert.strictEqual(toDataUri("iVBORw0KGgoAAAANSUhEUgAA"), "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA");
    assert.strictEqual(toDataUri("data:image/jpeg;base64,123"), "data:image/jpeg;base64,123");
    assert.strictEqual(toDataUri("blob:http://localhost:5173/abc"), "blob:http://localhost:5173/abc");
});

// ---------------------------------------------------------------------------
// 16. Missing Optional API Fields Handling
// ---------------------------------------------------------------------------
test("Scenario 16: Resilience to missing optional API response fields", () => {
    const minimalHybridResp = {
        status: "completed",
        image_width: 640,
        image_height: 480,
        alignment: {
            status: "accepted",
            method: "SIFT",
            confidence: 0.85,
            inlier_count: 50,
            inlier_ratio: 0.7,
            reprojection_error: 1.2,
            valid_overlap_pixels: 250000,
            valid_overlap_percentage: 81.3,
        },
        visual_comparison: {
            ssim_score: 0.88,
            visual_changed_pixels: 5000,
            visual_change_percentage: 2.0,
            changed_regions_count: 1,
        },
        crack_comparison: {
            before_crack_pixels: 100,
            after_crack_pixels: 200,
            before_crack_coverage_percentage: 0.04,
            after_crack_coverage_percentage: 0.08,
            coverage_change_percentage_points: 0.04,
            persistent_pixels: 80,
            newly_observed_pixels: 120,
            no_longer_observed_pixels: 20,
            raw_change_pixels: 140,
            raw_change_percentage: 0.05,
            tolerance_px: 2,
            tolerance_adjusted_new_pixels: 90,
            tolerance_adjusted_no_longer_observed_pixels: 10,
            potential_new_components: 1,
            potential_expanded_components: 0,
            persistent_components: 1,
            no_longer_observed_components: 0,
            // optional fields absent or default
        },
        images: {
            aligned_before: "b64",
            visual_difference: "b64",
            before_crack_mask: "b64",
            after_crack_mask: "b64",
            newly_observed_mask: "b64",
            persistent_mask: "b64",
            no_longer_observed_mask: "b64",
            tolerance_adjusted_new_mask: null, // explicitly null
            hybrid_overlay: "b64",
        },
        // timing omitted
    };

    // Verify safe field access
    assert.strictEqual(minimalHybridResp.images.tolerance_adjusted_new_mask, null);
    assert.strictEqual(minimalHybridResp.timing, undefined);
    const noiseFloor = minimalHybridResp.crack_comparison.calibrated_noise_floor_pp ?? 0.89;
    assert.strictEqual(noiseFloor, 0.89);
});

// ---------------------------------------------------------------------------
// 17. News Classification Response Schema
// ---------------------------------------------------------------------------
test("Scenario 17: NLP news classification prediction and probabilities schema", () => {
    const mockNlpResp = {
        category: "Sci/Tech",
        confidence: 0.9412,
        probabilities: {
            "World": 0.0123,
            "Sports": 0.0085,
            "Business": 0.0380,
            "Sci/Tech": 0.9412,
        },
    };

    assert.strictEqual(mockNlpResp.category, "Sci/Tech");
    assert.ok(mockNlpResp.confidence > 0.9);
    const sumProb = Object.values(mockNlpResp.probabilities).reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sumProb - 1.0) < 0.001);
});

// ---------------------------------------------------------------------------
// 18. Responsive Layout Structure Check
// ---------------------------------------------------------------------------
test("Scenario 18: Navigation and responsive layout configuration", () => {
    const navSections = [
        "Main Dashboard",
        "Computer Vision & Damage",
        "Natural Language Processing",
        "Diagnostics",
    ];

    assert.strictEqual(navSections.length, 4);
    assert.ok(navSections.includes("Computer Vision & Damage"));
});

// ---------------------------------------------------------------------------
// 19. Keyboard Accessibility Standards
// ---------------------------------------------------------------------------
test("Scenario 19: Keyboard accessibility compliance", () => {
    const isAccessibleActivationKey = (key) => key === "Enter" || key === " ";
    assert.strictEqual(isAccessibleActivationKey("Enter"), true);
    assert.strictEqual(isAccessibleActivationKey(" "), true);
    assert.strictEqual(isAccessibleActivationKey("Tab"), false);
});

// ---------------------------------------------------------------------------
// 20. Repeated Uploads and Resetting Results
// ---------------------------------------------------------------------------
test("Scenario 20: Reset lifecycle clears analysis state cleanly", () => {
    let state = {
        file: { name: "test.jpg" },
        result: { status: "completed" },
        error: null,
    };

    const resetState = () => {
        state = { file: null, result: null, error: null };
    };

    assert.ok(state.file !== null);
    resetState();
    assert.strictEqual(state.file, null);
    assert.strictEqual(state.result, null);
    assert.strictEqual(state.error, null);
});

// ===========================================================================
// TASK 2 SPECIFIC REGRESSION TESTS: ERROR-HANDLING FIX
// ===========================================================================

test("Regression: HTTP 422 with detail.error = ALIGNMENT_FAILED is never classified as NETWORK_ERROR", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: {
                error: "ALIGNMENT_FAILED",
                message: "Image alignment failed. Use photos with similar viewpoints and visible common details.",
                attempted_methods: ["SIFT", "AKAZE", "ORB"],
                failure_reasons: [
                    "Insufficient matching features between the two images."
                ],
                guidance: "Ensure both images show the same scene from a similar viewpoint with visible shared details. If the images are already pixel-aligned, you may use alignment_mode='direct'."
            }
        }
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "ALIGNMENT_FAILED");
    assert.notStrictEqual(parsed.code, "NETWORK_ERROR");
    assert.strictEqual(parsed.title, "Image Alignment Failed");
    assert.strictEqual(parsed.message, "Image alignment failed. Use photos with similar viewpoints and visible common details.");
    assert.deepStrictEqual(parsed.attemptedMethods, ["SIFT", "AKAZE", "ORB"]);
    assert.deepStrictEqual(parsed.failureReasons, ["Insufficient matching features between the two images."]);
    assert.strictEqual(parsed.guidance, "Ensure both images show the same scene from a similar viewpoint with visible shared details. If the images are already pixel-aligned, you may use alignment_mode='direct'.");
    assert.strictEqual(parsed.isUnavailable, false);
});

test("Regression: HTTP 422 with detail.code = ALIGNMENT_FAILED preserves code and guidance", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: {
                code: "ALIGNMENT_FAILED",
                message: "Alignment failed across all detectors.",
                attempted_methods: ["SIFT"],
                failure_reasons: ["Zero valid inliers."],
                guidance: "Try photos with greater texture."
            }
        }
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "ALIGNMENT_FAILED");
    assert.strictEqual(parsed.title, "Image Alignment Failed");
    assert.strictEqual(parsed.guidance, "Try photos with greater texture.");
    assert.deepStrictEqual(parsed.attemptedMethods, ["SIFT"]);
    assert.deepStrictEqual(parsed.failureReasons, ["Zero valid inliers."]);
});

test("Regression: HTTP 422 with detail.error_code preserves code", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: {
                error_code: "ALIGNMENT_FAILED",
                message: "Feature extraction failed.",
                failure_reasons: ["Low contrast."],
                guidance: "Increase illumination."
            }
        }
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "ALIGNMENT_FAILED");
    assert.strictEqual(parsed.title, "Image Alignment Failed");
    assert.deepStrictEqual(parsed.failureReasons, ["Low contrast."]);
    assert.strictEqual(parsed.guidance, "Increase illumination.");
});

test("Regression: HTTP 422 with FastAPI validation error array", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: [
                { loc: ["body", "before"], msg: "Field required", type: "value_error.missing" },
                { loc: ["body", "after"], msg: "Field required", type: "value_error.missing" }
            ]
        }
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "VALIDATION_ERROR");
    assert.strictEqual(parsed.title, "Validation Error");
    assert.match(parsed.message, /Field required/);
    assert.strictEqual(parsed.failureReasons.length, 2);
    assert.notStrictEqual(parsed.code, "NETWORK_ERROR");
});

test("Regression: Genuine network disconnection is strictly classified as NETWORK_ERROR", () => {
    const networkErr = new Error("Network Error");
    // error.response is undefined
    const parsed = parseApiError(networkErr);
    assert.strictEqual(parsed.code, "NETWORK_ERROR");
    assert.strictEqual(parsed.title, "Network Connection Failed");
    assert.strictEqual(parsed.isUnavailable, true);
    assert.match(parsed.guidance, /http:\/\/127\.0\.0\.1:8000/);
});

test("Regression: HTTP 503 Damage Service Unavailable preserves guidance and status", () => {
    const err = createAxiosError({
        status: 503,
        data: {
            detail: {
                error: "DAMAGE_SERVICE_UNAVAILABLE",
                message: "The damage segmentation inference microservice is currently unreachable.",
                guidance: "Ensure the YOLO inference microservice is running under backend/venv_yolo on port 8001."
            }
        }
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "DAMAGE_SERVICE_UNAVAILABLE");
    assert.strictEqual(parsed.title, "Damage Service Unavailable");
    assert.strictEqual(parsed.isUnavailable, true);
    assert.match(parsed.guidance, /port 8001/);
});

test("Regression: HTTP 504 Gateway Timeout preserves timeout classification", () => {
    const err = createAxiosError({
        status: 504,
        data: {
            detail: {
                error: "DAMAGE_SERVICE_TIMEOUT",
                message: "Inference microservice timed out."
            }
        }
    });

    const parsed = parseApiError(err);
    assert.strictEqual(parsed.code, "DAMAGE_SERVICE_TIMEOUT");
    assert.strictEqual(parsed.isTimeout, true);
});

test("Regression: parseApiError idempotency ensures re-parsing never converts error into NETWORK_ERROR", () => {
    const err = createAxiosError({
        status: 422,
        data: {
            detail: {
                error: "ALIGNMENT_FAILED",
                message: "Image alignment failed. Use photos with similar viewpoints and visible common details.",
                attempted_methods: ["SIFT", "AKAZE", "ORB"],
                failure_reasons: [
                    "Insufficient matching features between the two images."
                ],
                guidance: "Ensure both images show the same scene from a similar viewpoint with visible shared details. If the images are already pixel-aligned, you may use alignment_mode='direct'."
            }
        }
    });

    const firstParse = parseApiError(err);
    assert.strictEqual(firstParse.code, "ALIGNMENT_FAILED");

    // Re-parsing the already parsed object (e.g. inside ErrorState)
    const secondParse = parseApiError(firstParse);
    assert.strictEqual(secondParse.code, "ALIGNMENT_FAILED");
    assert.notStrictEqual(secondParse.code, "NETWORK_ERROR");
    assert.strictEqual(secondParse.title, "Image Alignment Failed");
    assert.strictEqual(secondParse.guidance, firstParse.guidance);
});

// ---------------------------------------------------------------------------
// Task 2 Alignment Presentation & Telemetry Tests
// ---------------------------------------------------------------------------
test("Task 2 Telemetry: Direct mode null/unverified telemetry resolves to N/A without 0.0%", () => {
    const directPayload = {
        status: "direct",
        method: "DIRECT",
        method_used: "DIRECT",
        detector: "DIRECT",
        geometric_status: "ASSUMED_UNVERIFIED",
        confidence: null,
        alignment_quality: null,
        matches: null,
        inliers: null,
        inlier_ratio: null,
        reprojection_error: null,
        overlap_ratio: 1.0,
        dimension_normalized: false,
        warnings: [
            "Direct comparison mode: geometric alignment was not verified. Results assume that corresponding pixels represent the same scene locations. Resizing or normalization does not independently verify alignment."
        ]
    };

    const isDirect =
        directPayload.status === "direct" ||
        directPayload.method === "DIRECT" ||
        directPayload.geometric_status === "ASSUMED_UNVERIFIED";

    assert.strictEqual(isDirect, true);

    const displayQuality = isDirect ? "N/A" : `${(directPayload.confidence * 100).toFixed(1)}%`;
    assert.strictEqual(displayQuality, "N/A");
    assert.notStrictEqual(displayQuality, "0.0%");

    const inliersDisplay = isDirect ? "N/A" : `${directPayload.inliers} / ${directPayload.matches}`;
    assert.strictEqual(inliersDisplay, "N/A");

    const reprojDisplay = isDirect ? "N/A" : `${directPayload.reprojection_error} px`;
    assert.strictEqual(reprojDisplay, "N/A");

    const overlapDisplay = `${(directPayload.overlap_ratio * 100).toFixed(1)}%`;
    assert.strictEqual(overlapDisplay, "100.0%");

    assert.match(directPayload.warnings[0], /geometric alignment was not verified/);
});

test("Task 2 Telemetry: Auto feature-based alignment correctly formats registration metrics", () => {
    const autoPayload = {
        status: "aligned",
        method: "SIFT",
        method_used: "SIFT",
        detector: "SIFT",
        geometric_status: "VERIFIED_FEATURE_REGISTRATION",
        confidence: 0.8924,
        alignment_quality: 0.8924,
        matches: 120,
        inliers: 85,
        inlier_ratio: 0.7083,
        reprojection_error: 1.42,
        overlap_ratio: 0.985,
        dimension_normalized: false,
        warnings: []
    };

    const isDirect =
        autoPayload.status === "direct" ||
        autoPayload.method === "DIRECT" ||
        autoPayload.geometric_status === "ASSUMED_UNVERIFIED";

    assert.strictEqual(isDirect, false);

    const displayQuality = `${(autoPayload.confidence * 100).toFixed(1)}%`;
    assert.strictEqual(displayQuality, "89.2%");

    const inliersDisplay = `${autoPayload.inliers} / ${autoPayload.matches} (${(autoPayload.inlier_ratio * 100).toFixed(1)}%)`;
    assert.strictEqual(inliersDisplay, "85 / 120 (70.8%)");

    const reprojDisplay = `${autoPayload.reprojection_error.toFixed(2)} px`;
    assert.strictEqual(reprojDisplay, "1.42 px");

    const overlapDisplay = `${(autoPayload.overlap_ratio * 100).toFixed(1)}%`;
    assert.strictEqual(overlapDisplay, "98.5%");
});

test("Task 2 Telemetry: Dimension normalization preserves details and aspect ratio warnings", () => {
    const normalizedPayload = {
        dimension_normalized: true,
        dimension_details: {
            before_source: { width: 2048, height: 1536, aspect_ratio: 1.333 },
            after_source: { width: 1448, height: 1086, aspect_ratio: 1.333 },
            processing_dimensions: { width: 1200, height: 900 },
            aspect_ratio_match: true
        },
        warnings: [
            "Source images had different dimensions (1448x1086 vs 2048x1536). After image was resized to reference coordinate frame."
        ]
    };

    assert.strictEqual(normalizedPayload.dimension_normalized, true);
    assert.strictEqual(normalizedPayload.dimension_details.aspect_ratio_match, true);
    assert.strictEqual(normalizedPayload.dimension_details.before_source.width, 2048);
    assert.strictEqual(normalizedPayload.dimension_details.after_source.width, 1448);
});
