import argparse
import os
import shutil
import cv2
import numpy as np
import torch
from torchvision import transforms
from face_detection.scrfd.detector import SCRFD
import albumentations as A
import random
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize detectors and recognizers
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_10g_bnkps.onnx")
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

def get_augmentation_transforms(h, w):
    """Enhanced augmentation pipeline from your latest version."""
    scale_factor = min(h, w) / 112.0

    transform_group_1 = A.Compose([
        A.Downscale(scale_min=0.1, scale_max=0.5, interpolation=cv2.INTER_AREA, p=1.0),
        A.OneOf([
            A.GaussNoise(var_limit=(20.0, 100.0), p=0.8),
            A.ISONoise(color_shift=(0.05, 0.1), intensity=(0.1, 0.5), p=0.6),
        ], p=0.7),
        A.ImageCompression(quality_lower=10, quality_upper=40, p=0.6)
    ])

    transform_group_2 = A.Compose([
        A.RandomResizedCrop(height=h, width=w, scale=(0.3, 1.0), ratio=(0.75, 1.33), p=1.0),
        A.Perspective(scale=(0.05, 0.25), p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.6),
            A.MotionBlur(blur_limit=(5, 15), p=0.4)
        ], p=0.5)
    ])

    transform_group_3 = A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.8, 0.2), contrast_limit=(-0.6, 0.4), brightness_by_max=True, p=1.0
        ),
        A.OneOf([
            A.Solarize(threshold=(50, 150), p=0.5),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_upper=3, p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.5, alpha_coef=0.08, p=0.3)
        ], p=0.7)
    ])

    transform_group_4 = A.Compose([
        A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=50, val_shift_limit=50, p=1.0),
        A.OneOf([
            A.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=0.4),
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.3),
            A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.1, snow_point_upper=0.3, p=0.3)
        ], p=0.6)
    ])

    transform_group_5 = A.Compose([
        A.Affine(scale=(0.5, 1.5), rotate=(-30, 30), shear=(-25, 25), translate_percent=(-0.2, 0.2), p=1.0),
        A.ElasticTransform(alpha=scale_factor*50, sigma=scale_factor*5, p=0.5)
    ])

    transform_group_6 = A.Compose([
        A.MotionBlur(blur_limit=(7, 25), p=1.0),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.7),
        A.RandomGridShuffle(grid=(3, 3), p=0.3)
    ])

    transform_group_7 = A.Compose([
        A.Downscale(scale_min=0.1, scale_max=0.3, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=(-0.9, 0.1), contrast_limit=(-0.7, 0.3), p=0.8),
        A.OneOf([
            A.GaussNoise(var_limit=(50.0, 150.0), p=0.7),
            A.MultiplicativeNoise(multiplier=(0.5, 1.5), p=0.6)
        ], p=0.7)
    ])

    return [transform_group_1, transform_group_2, transform_group_3, transform_group_4, 
            transform_group_5, transform_group_6, transform_group_7]

def mixup(image1, image2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_image = (lam * image1 + (1 - lam) * image2).clip(0, 255).astype(np.uint8)
    return mixed_image

def cutmix(image1, image2, alpha=0.4):
    h, w, _ = image1.shape
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(w * cut_rat), int(h * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    return mixed_image

def apply_augmentation(face_image, save_path, name, count):
    h, w, _ = face_image.shape
    augmented_faces = [face_image]
    transforms_list = get_augmentation_transforms(h, w)
    aug_save_dir = os.path.join(save_path, name)
    os.makedirs(aug_save_dir, exist_ok=True)

    for i in range(30):
        transform = random.choice(transforms_list)
        augmented = transform(image=face_image)['image']
        augmented_faces.append(augmented)
        aug_save_path = os.path.join(aug_save_dir, f"{count}_{i}.jpg")
        cv2.imwrite(aug_save_path, augmented)

    for i in range(15):
        img1 = random.choice(augmented_faces)
        img2 = random.choice(augmented_faces)
        mixed_img = mixup(img1, img2)
        cutmix_img = cutmix(img1, img2)
        cv2.imwrite(os.path.join(aug_save_dir, f"{count}_mixup_{i}.jpg"), mixed_img)
        cv2.imwrite(os.path.join(aug_save_dir, f"{count}_cutmix_{i}.jpg"), cutmix_img)

    return augmented_faces

def get_feature(face_image):
    """Revert to original single-scale feature extraction."""
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = face_preprocess(face_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb = recognizer(face_tensor)[0].detach().cpu().numpy()
        normalized_emb = emb / np.linalg.norm(emb)
    
    return normalized_emb

def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    images_name = []
    images_emb = []

    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        count = 0
        for image_name in os.listdir(person_image_path):
            if image_name.lower().endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))
                bboxes, _ = detector.detect(image=input_image)

                for bbox in bboxes:
                    x1, y1, x2, y2, _ = map(int, bbox)
                    face_image = input_image[y1:y2, x1:x2]
                    augmented_faces = apply_augmentation(face_image, faces_save_dir, name_person, count)

                    for aug_face in augmented_faces:
                        emb = get_feature(face_image=aug_face)
                        images_emb.append(emb)
                        images_name.append(name_person)

                    count += 1
                    print(f"Processed {name_person}: image {count}")

    np.savez_compressed(features_path, images_name=np.array(images_name), images_emb=np.array(images_emb))
    print("Added new persons successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-dir", type=str, default="./datasets/backup")
    parser.add_argument("--add-persons-dir", type=str, default="./datasets/new_persons")
    parser.add_argument("--faces-save-dir", type=str, default="./datasets/data/")
    parser.add_argument("--features-path", type=str, default="./datasets/face_features/feature")
    opt = parser.parse_args()

    add_persons(**vars(opt))