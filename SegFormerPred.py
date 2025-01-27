import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

input_dir = '/media/summer/ubuntu 2tb/ISDC_MAIN/Semantic-Segmentation-on-Martian-Terrain/ABC_Copy'
output_dir = '/media/summer/ubuntu 2tb/ISDC_MAIN/Semantic-Segmentation-on-Martian-Terrain/ABC_Copy_Results_Terrain_Classification'  

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def main():
    model = torch.load("SegFormerMars.pth")
    
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing {filename}...")

            img = cv2.imread(img_path)
            img = img[:,80:]

            input_img = Image.fromarray(img).resize((512, 512))
            im_arr = np.array(input_img)
            im = torch.tensor(im_arr, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            im = im / 255.0
            im = (im - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

            logits = model(im)[0]

            upsampled_logits = F.interpolate(
                logits,
                size=(512, 512),
                mode="bilinear",
                align_corners=False
            )

            predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()  

            color_map = {
                0: (0, 0, 0, 100),  # Nothing=Black
                1: (255, 0, 0, 100),  # Sand=Red
                2: (0, 255, 0, 100),  # Soil=Green
                3: (0, 0, 255, 100)  # Rock=Blue
            }

            color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 4), dtype=np.uint8)

            for cls, color in color_map.items():
                color_mask[predicted_mask == cls] = color

            mask_img = Image.fromarray(color_mask, mode='RGBA')

            input_img_rgba = input_img.convert("RGBA")

            overlay_img = Image.alpha_composite(input_img_rgba, mask_img)

            plt.figure(figsize=(10, 10))
            plt.imshow(overlay_img)
            plt.axis("off")
            plt.show()

            output_img_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_result.png")
            overlay_img.save(output_img_path)
            print(f"Saved result image as {output_img_path}")

if __name__ == '__main__':
    main()
