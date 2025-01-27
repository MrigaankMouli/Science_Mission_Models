from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

input_dir = '/media/summer/ubuntu 2tb/ISDC_MAIN/Semantic-Segmentation-on-Martian-Terrain/ABC_Copy'
output_dir = '/media/summer/ubuntu 2tb/ISDC_MAIN/Semantic-Segmentation-on-Martian-Terrain/ABC_Copy_Results_Craters'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    model = YOLO("best_craters.pt")

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            image = image[:, 80:]

            results = model(image)

            for result in results:
                for box in result.boxes.data.tolist():  
                    cls, conf, x1, y1, x2, y2 = box[5], box[4], box[0], box[1], box[2], box[3]
                    print(f"Class: {int(cls)}, Confidence: {conf:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

            annotated_image = np.array(results[0].plot())  
            output_path = os.path.join(output_dir, filename)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)  
            cv2.imwrite(output_path, annotated_image)

            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.show()

if __name__ == "__main__":
    main()
