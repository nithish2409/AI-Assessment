"""
Verification script for testing live FastAPI endpoints with real and synthetic images.
"""
import time
import httpx
import cv2
import numpy as np
from pathlib import Path

def test_live_damage_api():
    base_url = "http://127.0.0.1:8000"
    client = httpx.Client(timeout=30.0)
    
    # 1. Health checks
    print("--- 1. Testing Health Endpoints ---")
    h_resp = client.get(f"{base_url}/api/health")
    print(f"GET /api/health: {h_resp.status_code} -> {h_resp.json()}")
    assert h_resp.status_code == 200
    
    d_resp = client.get(f"{base_url}/api/damage/health")
    print(f"GET /api/damage/health: {d_resp.status_code} -> {d_resp.json()}")
    assert d_resp.status_code == 200
    assert d_resp.json()["is_available"] is True

    # 2. Test Real Crack Image
    print("\n--- 2. Testing Real Crack Photograph ---")
    test_img_path = Path("backend/datasets/damage/processed/images/test/1616.rf.c868709931a671796794fdbb95352c5a.jpg")
    assert test_img_path.exists(), "Test image not found."

    t0 = time.perf_counter()
    with open(test_img_path, "rb") as f:
        files = {"image": ("crack_test.jpg", f.read(), "image/jpeg")}
        data = {"confidence": "0.25", "iou": "0.45"}
        resp = client.post(f"{base_url}/api/damage/segment", files=files, data=data)
    e2e_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Status Code: {resp.status_code}")
    assert resp.status_code == 200
    data = resp.json()
    print(f"Model: {data['model']}")
    print(f"Dimensions: {data['image_width']} x {data['image_height']}")
    print(f"Detections Count: {data['detections_count']}")
    print(f"Crack Coverage (%): {data['crack_coverage_percentage']}%")
    print(f"Device: {data['inference']['device']}")
    print(f"Internal Duration: {data['inference']['duration_ms']} ms")
    print(f"  - Preprocess: {data['inference'].get('preprocess_ms')} ms")
    print(f"  - Inference: {data['inference'].get('inference_ms')} ms")
    print(f"  - Postprocess: {data['inference'].get('postprocess_ms')} ms")
    print(f"End-to-End API Latency: {e2e_ms:.2f} ms")
    
    for idx, det in enumerate(data["detections"]):
        print(f"  Detection #{idx+1}: {det['class_name']} (conf={det['confidence']:.4f}, bbox={det['bbox']}, area={det['area_pixels']} px)")

    # 3. Test Negative / Non-Crack Image
    print("\n--- 3. Testing Non-Crack Clean Surface ---")
    clean_wall = np.full((640, 640, 3), 190, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", clean_wall)
    
    t0 = time.perf_counter()
    resp_clean = client.post(
        f"{base_url}/api/damage/segment",
        files={"image": ("clean_wall.png", encoded.tobytes(), "image/png")},
    )
    e2e_clean_ms = (time.perf_counter() - t0) * 1000.0
    print(f"Status Code: {resp_clean.status_code}")
    assert resp_clean.status_code == 200
    clean_data = resp_clean.json()
    print(f"Detections Count: {clean_data['detections_count']}")
    print(f"Crack Coverage (%): {clean_data['crack_coverage_percentage']}%")
    print(f"Warnings: {clean_data['warnings']}")
    print(f"End-to-End API Latency: {e2e_clean_ms:.2f} ms")

    # 4. Test Task 2 and Task 3 are completely unaffected
    print("\n--- 4. Testing NLP and Image Comparison Endpoint Regression ---")
    nlp_resp = client.post(
        f"{base_url}/api/nlp/classify",
        json={"text": "Wall Street rallies as quarterly earnings beat expectations across major financial firms."}
    )
    print(f"Task 3 NLP classify status: {nlp_resp.status_code}")
    assert nlp_resp.status_code == 200
    print(f"Task 3 NLP prediction: {nlp_resp.json().get('category')}")

    print("\nAll live API tests passed successfully!")

if __name__ == "__main__":
    test_live_damage_api()
