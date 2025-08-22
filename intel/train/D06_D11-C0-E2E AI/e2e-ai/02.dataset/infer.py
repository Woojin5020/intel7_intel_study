import torch
from pathlib import Path
from PIL import Image

# 1. 모델 불러오기 (학습 완료된 가중치 .pt 파일 경로 지정)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp2/weights/best.pt')

# 2. 테스트할 이미지 경로 설정
test_img_path = '/home/gabriel/workspace/ai/cup/WIN_20250626_15_17_34_Pro.jpg'

# 3. 이미지 로드 (선택 사항 - PIL, OpenCV 등 원하는 방식 사용 가능)
img = Image.open(test_img_path)

# 4. 추론 실행
results = model(img)

# 5. 결과 이미지 저장 (box, label 등이 그려진 이미지)
output_folder = 'inference_results'
Path(output_folder).mkdir(parents=True, exist_ok=True)

results.save(save_dir=output_folder)

print(f"Inference 결과가 {output_folder} 폴더에 저장되었습니다.")

