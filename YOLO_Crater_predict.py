from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

model = YOLO("best_craters.pt")

image_path = "/media/summer/ubuntu 2tb/ISDC_MAIN/YOLO_Crater/object-detection-isdc-4/extra_images/frame8_jpg.rf.3b7a4468dd2995e5c42386226c8d9c8f.jpg"
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
