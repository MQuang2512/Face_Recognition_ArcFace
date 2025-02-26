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

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the face detector (Choose one of the detectors)
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_10g_bnkps.onnx")

# Initialize the face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)


def get_augmentation_transforms(h, w):
    """Trả về danh sách các nhóm Augmentation."""
    transform_group_1 = A.Compose([
        A.CoarseDropout(max_holes=5, max_height=int(h * 0.2), max_width=int(w * 0.2), p=1.0),
        A.OneOf([
            A.GaussianBlur(blur_limit=(5, 15), p=0.7),
            A.MotionBlur(blur_limit=(5, 15), p=0.5),
            A.GlassBlur(sigma=0.7, max_delta=3, iterations=2, p=0.3)
        ], p=0.5)
    ])

    transform_group_2 = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.3, 0.3), p=1.0),
        A.RandomGamma(gamma_limit=(60, 150), p=0.8)
    ])

    transform_group_3 = A.Compose([
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=40, p=1.0),
        A.FancyPCA(alpha=0.1, p=0.5),
        A.ChannelShuffle(p=0.5),
        A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.5)
    ])

    transform_group_4 = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=(-0.3, 0.3), rotate_limit=0, p=1.0),
        A.Perspective(scale=(0.05, 0.2), p=0.8)
    ])

    transform_group_5 = A.Compose([
        A.Affine(shear=(-20, 20), rotate=(-15, 15), p=1.0),
        A.Perspective(scale=(0.05, 0.15), p=0.8)
    ])

    transform_group_6 = A.Compose([
        A.MotionBlur(blur_limit=(5, 15), p=1.0),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.7)
    ])

    transform_group_7 = A.Compose([
        A.Downscale(scale_min=0.3, scale_max=0.7, p=1.0),
        A.ImageCompression(quality_lower=20, quality_upper=50, p=0.8),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.7)
    ])

    transform_group_8 = A.Compose([
        A.Downscale(scale_min=0.3, scale_max=0.5, p=1.0),  # Giảm độ phân giải mạnh hơn
        A.GaussNoise(var_limit=(20.0, 100.0), p=0.8),  # Nhiễu Gaussian mạnh hơn
        A.RandomBrightnessContrast(brightness_limit=(-0.6, 0.3), contrast_limit=(-0.5, 0.5), p=0.7)
    ])

    transform_group_9 = A.Compose([
        A.MotionBlur(blur_limit=(15, 39), p=0.75),  # Làm mờ do chuyển động nhanh hơn
        A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.7)  # Biến dạng quang học
    ])


    return [transform_group_1, transform_group_2, transform_group_3, transform_group_4, 
            transform_group_5, transform_group_6, transform_group_7, transform_group_8, transform_group_9]


def mixup(image1, image2, alpha=0.4):
    """Mixup augmentation: trộn 2 ảnh với trọng số ngẫu nhiên."""
    lam = np.random.beta(alpha, alpha)
    mixed_image = (lam * image1 + (1 - lam) * image2).astype(np.uint8)
    return mixed_image


def cutmix(image1, image2, alpha=0.4):
    """CutMix augmentation: cắt một phần ảnh này để trộn vào ảnh kia."""
    h, w, _ = image1.shape
    lam = np.random.beta(alpha, alpha)
    cx, cy = np.random.randint(w), np.random.randint(h)
    w_box, h_box = int(w * np.sqrt(1 - lam)), int(h * np.sqrt(1 - lam))
    x1, y1 = max(cx - w_box // 2, 0), max(cy - h_box // 2, 0)
    x2, y2 = min(cx + w_box // 2, w), min(cy + h_box // 2, h)
    
    mixed_image = image1.copy()
    mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    return mixed_image


def apply_augmentation(face_image, save_path, name, count):
    """Áp dụng augmentation và lưu ảnh."""
    h, w, _ = face_image.shape
    augmented_faces = [face_image]
    transforms_list = get_augmentation_transforms(h, w)

    for i in range(20):
        transform = random.choice(transforms_list)
        augmented = transform(image=face_image)['image']
        augmented_faces.append(augmented)

        aug_save_dir = os.path.join(save_path, name)
        os.makedirs(aug_save_dir, exist_ok=True)
        aug_save_path = os.path.join(aug_save_dir, f"{count}_{i}.jpg")
        cv2.imwrite(aug_save_path, augmented)

    # Áp dụng Mixup & CutMix trên ảnh gốc và ảnh đã augmented
    for i in range(10):
        img1 = random.choice(augmented_faces)
        img2 = random.choice(augmented_faces)
        mixed_img = mixup(img1, img2)
        cutmix_img = cutmix(img1, img2)

        cv2.imwrite(os.path.join(save_path, name, f"{count}_mixup_{i}.jpg"), mixed_img)
        cv2.imwrite(os.path.join(save_path, name, f"{count}_cutmix_{i}.jpg"), cutmix_img)

    return augmented_faces


def get_feature(face_image):
    """Trích xuất đặc trưng của khuôn mặt."""
    face_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    face_tensor = face_preprocess(face_rgb).unsqueeze(0).to(device)
    emb = recognizer(face_tensor)[0].detach().cpu().numpy()
    return emb / np.linalg.norm(emb)


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

    np.savez_compressed(features_path, images_name=np.array(images_name), images_emb=np.array(images_emb))
    print("Added new persons successfully!")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default="./datasets/new_persons",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    add_persons(**vars(opt))
