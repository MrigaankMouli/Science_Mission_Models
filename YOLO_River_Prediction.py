from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("best_rivers.pt")

image_path = "/media/summer/ubuntu 2tb/ISDC_MAIN/River_Valleys_Seg_YOLO/Data/train/images/42.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  

# Perform inference
results = model(image)

for result in results[0].boxes:
    cls = int(result.cls.item()) 
    conf = float(result.conf.item())  
    print(f"Class: {cls}, Confidence: {conf:.2f}")

annotated_image = results[0].plot()
plt.imshow(annotated_image)
plt.axis("off")
plt.show()
