import easyocr
import cv2
import numpy as np
import torch
import time

print('PyTorch CUDA available:', torch.cuda.is_available())
print('Initializing EasyOCR Reader (GPU mode)...')
t0 = time.perf_counter()
reader = easyocr.Reader(['en'], gpu=True)
print(f'Reader initialized in {time.perf_counter() - t0:.2f} s')

# Test 1: Image WITH text
img_text = np.full((200, 600, 3), 255, dtype=np.uint8)
cv2.putText(img_text, 'INSPECTION TAG #402', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)

t0 = time.perf_counter()
res_text = reader.readtext(img_text)
dt = (time.perf_counter() - t0) * 1000
print(f'\nTest 1 (With Text) - Latency: {dt:.1f} ms')
print(f'Extracted {len(res_text)} items:')
for bbox, text, conf in res_text:
    print(f'   Text: "{text}", Conf: {conf:.4f}')

# Test 2: Image WITHOUT text
img_blank = np.zeros((200, 600, 3), dtype=np.uint8)
t0 = time.perf_counter()
res_blank = reader.readtext(img_blank)
dt = (time.perf_counter() - t0) * 1000
print(f'\nTest 2 (Blank Image) - Latency: {dt:.1f} ms')
print(f'Extracted items: {len(res_blank)} (Expected: 0)')

print('\nCUDA Memory allocated (MB):', round(torch.cuda.memory_allocated() / (1024**2), 2))
print('CUDA Memory reserved (MB):', round(torch.cuda.memory_reserved() / (1024**2), 2))
