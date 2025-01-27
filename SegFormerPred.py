import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F


def main():
    model =torch.load("SegFormerMars.pth")
    img_path = '/media/summer/ubuntu 2tb/ISDC_MAIN/Semantic-Segmentation-on-Martian-Terrain/ABC/image_634_1737025989.911667.png'
    img = cv2.imread(img_path)  
    img = img[:, 80:, :]  

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

    predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)

    color_map = {
        0:(0,0,0,100), #Nothing
        1:(255,0,0,100), #Sand
        2:(0,255,0,100), #Soil
        3:(0,0,255,100) #Rock
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

    #overlay_img.save("segmentation_overlay.png")
    #Optional save

if __name__ == '__main__':
    main()
