## Steps to run training and inferencing

```sh
labelme2yolo --json_dir ./cup/

ls -ahl cup/YOLODataset/dataset.yaml

git clone https://github.com/ultralytics/yolov5.git

python3 –m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 train.py --img 640 --batch 16 --epochs 50 --data /<YOLO dataset full path>/dataset.yaml --weights yolov5s.pt

python3 infer.py
```

