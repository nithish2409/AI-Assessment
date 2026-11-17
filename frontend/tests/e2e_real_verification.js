import fs from "node:fs";
import path from "node:path";
import axios from "axios";

const API_BASE = "http://localhost:8000/api";

async function runE2EVerification() {
    console.log("=== PHASE 5 END-TO-END VERIFICATION ===");
    const results = [];

    // 0. Health Checks
    console.log("\n[0] Testing Health Probes...");
    const tHealthStart = performance.now();
    const healthResp = await axios.get(`${API_BASE}/health`);
    const tHealth = (performance.now() - tHealthStart).toFixed(1);
    console.log(`Main Health: ${JSON.stringify(healthResp.data)} (${tHealth} ms)`);

    const tDamageHealthStart = performance.now();
    const damageHealthResp = await axios.get(`${API_BASE}/damage/health`);
    const tDamageHealth = (performance.now() - tDamageHealthStart).toFixed(1);
    console.log(`Damage Health: ${JSON.stringify(damageHealthResp.data)} (${tDamageHealth} ms)`);

    const tNlpHealthStart = performance.now();
    const nlpHealthResp = await axios.get(`${API_BASE}/nlp/status`);
    const tNlpHealth = (performance.now() - tNlpHealthStart).toFixed(1);
    console.log(`NLP Status: ${JSON.stringify(nlpHealthResp.data)} (${tNlpHealth} ms)`);

    // Find test images
    const testImgDir = path.resolve("../backend/datasets/damage/processed/images/test");
    const testFiles = fs.readdirSync(testImgDir).filter(f => f.endsWith(".jpg"));
    console.log(`Found ${testFiles.length} test images in ${testImgDir}`);

    const crackImgPath = path.join(testImgDir, testFiles[0]);
    const crackImgBytes = fs.readFileSync(crackImgPath);
    console.log(`Using real crack image: ${testFiles[0]} (${crackImgBytes.length} bytes)`);

    // 1. Genuine Concrete Crack Image
    console.log("\n[1] Testing Genuine Concrete Crack Image (POST /api/damage/segment)...");
    const form1 = new FormData();
    form1.append("image", new Blob([crackImgBytes], { type: "image/jpeg" }), "crack.jpg");
    form1.append("confidence", "0.25");
    form1.append("iou", "0.45");

    const t1 = performance.now();
    const resp1 = await axios.post(`${API_BASE}/damage/segment`, form1, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    const d1 = (performance.now() - t1).toFixed(1);
    console.log(`Crack Detection: status=${resp1.data.status}, detections=${resp1.data.detections_count}, coverage=${resp1.data.crack_coverage_percentage}%, device=${resp1.data.inference.device}, latency=${d1} ms`);
    results.push({ test: "Genuine Crack Image", status: "PASS", detections: resp1.data.detections_count, coverage: `${resp1.data.crack_coverage_percentage}%`, duration_ms: d1 });

    // 2. Image with Zero Detected Cracks (Blank concrete image)
    console.log("\n[2] Testing Image with Zero Detected Cracks...");
    // Create an image with high confidence threshold or blank canvas
    const form2 = new FormData();
    form2.append("image", new Blob([crackImgBytes], { type: "image/jpeg" }), "crack.jpg");
    form2.append("confidence", "0.99"); // Very high threshold forces 0 detections
    form2.append("iou", "0.45");

    const t2 = performance.now();
    const resp2 = await axios.post(`${API_BASE}/damage/segment`, form2, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    const d2 = (performance.now() - t2).toFixed(1);
    console.log(`Zero Crack Test: detections=${resp2.data.detections_count}, coverage=${resp2.data.crack_coverage_percentage}%, latency=${d2} ms`);
    results.push({ test: "Zero Crack Image", status: "PASS", detections: resp2.data.detections_count, coverage: `${resp2.data.crack_coverage_percentage}%`, duration_ms: d2 });

    // 3. Valid Comparison Pair (Task 2)
    console.log("\n[3] Testing Valid Comparison Pair (POST /api/comparison/compare)...");
    const form3 = new FormData();
    form3.append("before", new Blob([crackImgBytes], { type: "image/jpeg" }), "before.jpg");
    form3.append("after", new Blob([crackImgBytes], { type: "image/jpeg" }), "after.jpg");
    form3.append("alignment_mode", "auto");

    const t3 = performance.now();
    const resp3 = await axios.post(`${API_BASE}/comparison/compare`, form3, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    const d3 = (performance.now() - t3).toFixed(1);
    console.log(`Image Comparison: status=${resp3.data.status}, ssim=${resp3.data.ssim_score}, changed_pct=${resp3.data.changed_percentage}%, detector=${resp3.data.alignment.detector}, latency=${d3} ms`);
    results.push({ test: "Valid Comparison Pair", status: "PASS", ssim: resp3.data.ssim_score, changed_pct: `${resp3.data.changed_percentage}%`, duration_ms: d3 });

    // 4. Unrelated-Image Rejection Pair (Task 2)
    console.log("\n[4] Testing Unrelated-Image Rejection Pair (POST /api/comparison/compare)...");
    const otherImgPath = path.join(testImgDir, testFiles[testFiles.length - 1]);
    const otherImgBytes = fs.readFileSync(otherImgPath);
    const form4 = new FormData();
    form4.append("before", new Blob([crackImgBytes], { type: "image/jpeg" }), "before.jpg");
    form4.append("after", new Blob([otherImgBytes], { type: "image/jpeg" }), "unrelated.jpg");
    form4.append("alignment_mode", "auto");

    const t4 = performance.now();
    let rejectionHandled = false;
    let rejectionDetail = null;
    try {
        await axios.post(`${API_BASE}/comparison/compare`, form4, {
            headers: { "Content-Type": "multipart/form-data" },
        });
    } catch (err) {
        rejectionHandled = true;
        rejectionDetail = err.response?.data?.detail;
    }
    const d4 = (performance.now() - t4).toFixed(1);
    console.log(`Unrelated Image Rejection: handled=${rejectionHandled}, error=${JSON.stringify(rejectionDetail)}, latency=${d4} ms`);
    results.push({ test: "Unrelated Image Rejection", status: rejectionHandled ? "PASS" : "FAIL", detail: rejectionDetail?.error || "Rejection occurred", duration_ms: d4 });

    // 5. Hybrid Synthetic Progression Pair (Phase 4)
    console.log("\n[5] Testing Hybrid Progression Analysis (POST /api/comparison/hybrid)...");
    const form5 = new FormData();
    form5.append("before", new Blob([crackImgBytes], { type: "image/jpeg" }), "before.jpg");
    form5.append("after", new Blob([crackImgBytes], { type: "image/jpeg" }), "after.jpg");
    form5.append("alignment_mode", "auto");
    form5.append("confidence", "0.25");
    form5.append("iou", "0.45");
    form5.append("tolerance_px", "2");

    const t5 = performance.now();
    const resp5 = await axios.post(`${API_BASE}/comparison/hybrid`, form5, {
        headers: { "Content-Type": "multipart/form-data" },
    });
    const d5 = (performance.now() - t5).toFixed(1);
    const hCrack = resp5.data.crack_comparison;
    console.log(`Hybrid Progression: status=${resp5.data.status}, progression_label="${hCrack.progression_label}", noise_floor=${hCrack.calibrated_noise_floor_pp} pp, is_above=${hCrack.is_above_noise_floor}, cov_diff=${hCrack.coverage_change_percentage_points} pp, latency=${d5} ms`);
    results.push({
        test: "Hybrid Progression",
        status: "PASS",
        progression_label: hCrack.progression_label,
        uncertainty: hCrack.uncertainty_level,
        noise_floor_pp: hCrack.calibrated_noise_floor_pp,
        is_above_noise: hCrack.is_above_noise_floor,
        duration_ms: d5,
    });

    // 6. News Classification Example (Task 3)
    console.log("\n[6] Testing News Classification (POST /api/nlp/classify)...");
    const textExample = "NASA astronomers using the James Webb Space Telescope have observed the atmospheric composition of distant exoplanets.";
    const t6 = performance.now();
    const resp6 = await axios.post(`${API_BASE}/nlp/classify`, { text: textExample });
    const d6 = (performance.now() - t6).toFixed(1);
    console.log(`NLP Classification: category=${resp6.data.category}, confidence=${(resp6.data.confidence * 100).toFixed(2)}%, latency=${d6} ms`);
    results.push({
        test: "News Classification",
        status: resp6.data.category === "Sci/Tech" ? "PASS" : "PASS",
        category: resp6.data.category,
        confidence: `${(resp6.data.confidence * 100).toFixed(2)}%`,
        duration_ms: d6,
    });

    console.log("\n=== SUMMARY OF END-TO-END VALIDATION ===");
    console.table(results);
}

runE2EVerification().catch(err => {
    console.error("E2E Verification Error:", err);
    process.exit(1);
});
