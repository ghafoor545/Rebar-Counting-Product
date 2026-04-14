from ultralytics import YOLO

model = YOLO("yolo26x-seg.pt")     # ← YOLO26 medium segmentation

model.train(
    data=r"C:\Users\RYZEN\Desktop\Dataset Rebar\Rebar-Counting-Segmentation-4\data.yaml",
    
    epochs=60,           # 50-70 enough for deployment
    imgsz=1280,
    batch=2,
    
    lr0=0.0007,          # thoda balanced (kam nahi, tez bhi nahi)
    lrf=0.01,
               
    
    hsv_v=0.3,
    
    degrees=5.0,         
    fliplr=0.5,
    
    
    workers=8,
    patience=20,
    optimizer="AdamW",
    amp=True,
    
    project="runs/deploy_rebar",
    name="yolo26m_seg_factory1",
    exist_ok=True,
    save=True,
    save_period=1,
    val=True,
)