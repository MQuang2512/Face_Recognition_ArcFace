import streamlit as st
import cv2
import numpy as np
import torch
import threading
import time
import os
import shutil
from PIL import Image
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features, compare_encodings
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking
from face_alignment.alignment import norm_crop  # Import norm_crop
from torchvision import transforms
import uuid
from augmentation import apply_augmentation  # Import hàm từ augmentation.py

# Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo detector và recognizer
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_10g_bnkps.onnx")
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)

# Đường dẫn thư mục
NEW_PERSONS_DIR = "./datasets/new_persons"
DATA_DIR = "./datasets/data"
FEATURES_PATH = "./datasets/face_features/feature"

# Shared data for real-time recognition
data_mapping = {
    "raw_image": np.array([]),
    "tracking_ids": [],
    "detection_bboxes": np.array([]),
    "detection_landmarks": np.array([]),
    "tracking_bboxes": [],
}
data_lock = threading.Lock()
id_face_mapping = {}

# Face capture instructions
INSTRUCTIONS = [
    "Nhìn thẳng vào camera",
    "Nhìn sang trái",
    "Nhìn sang phải",
    "Nhìn lên trên",
    "Nhìn xuống dưới"
]

# Hàm trích xuất đặc trưng khuôn mặt
def get_feature(face_image):
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

# Hàm thêm người mới với augmentation, norm_crop, và lưu đặc trưng
def add_person(name, images):
    new_persons_dir = os.path.join(NEW_PERSONS_DIR, name)
    data_dir = os.path.join(DATA_DIR, name)
    os.makedirs(new_persons_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    images_name = []
    images_emb = []
    
    # Thanh tiến trình cho quá trình augmentation và lưu đặc trưng
    progress_bar = st.progress(0)
    total_steps = len(images) * 46  # Mỗi ảnh gốc tạo 45 ảnh augmentation + 1 ảnh gốc (theo logic trong apply_augmentation)
    
    for i, img in enumerate(images):
        # Phát hiện khuôn mặt và lấy landmarks
        bboxes, landmarks = detector.detect(image=img)
        if bboxes is not None and len(bboxes) > 0 and len(landmarks) > 0:
            bbox = bboxes[0]  # Lấy bounding box đầu tiên
            landmark = landmarks[0]  # Lấy landmarks đầu tiên
            x1, y1, x2, y2, _ = map(int, bbox)
            
            # Cắt khuôn mặt bằng norm_crop
            cropped_face = norm_crop(img=img, landmark=landmark)
            if cropped_face is not None:
                # Lưu ảnh gốc (bounding box) vào new_persons
                cv2.imwrite(os.path.join(new_persons_dir, f"original_{i}.jpg"), cropped_face)
                # Trích xuất đặc trưng từ ảnh gốc đã cắt
                emb = get_feature(cropped_face)
                images_emb.append(emb)
                images_name.append(name)
                progress_bar.progress((i * 46 + 1) / total_steps)
                
                # Augmentation: sử dụng hàm apply_augmentation từ augmentation.py
                augmented_faces = apply_augmentation(cropped_face, save_path=data_dir, name=name, count=i)
                
                # Trích xuất đặc trưng từ các ảnh augmentation
                for j, aug_img in enumerate(augmented_faces):
                    if isinstance(aug_img, np.ndarray):  # Đảm bảo aug_img là mảng numpy
                        emb = get_feature(aug_img)
                        images_emb.append(emb)
                        images_name.append(name)
                    progress_bar.progress((i * 46 + j + 2) / total_steps)
    
    # Lưu đặc trưng vào file feature
    if os.path.exists(FEATURES_PATH):
        data = np.load(FEATURES_PATH)
        existing_names = data['images_name'].tolist()
        existing_embs = data['images_emb'].tolist()
        images_name = existing_names + images_name
        images_emb = existing_embs + images_emb
    np.savez_compressed(FEATURES_PATH, images_name=np.array(images_name), images_emb=np.array(images_emb))
    st.success(f"Đã thêm {name} với augmentation và lưu đặc trưng thành công!")

# Hàm chụp ảnh khuôn mặt với bounding box và lưu chỉ bounding box làm ảnh gốc
def capture_faces(name):
    if f"step_{name}" not in st.session_state:
        st.session_state[f"step_{name}"] = 0
    if f"images_{name}" not in st.session_state:
        st.session_state[f"images_{name}"] = []
    
    step = st.session_state[f"step_{name}"]
    images = st.session_state[f"images_{name}"]
    
    if step < len(INSTRUCTIONS):
        st.write(f"**Bước {step + 1}: {INSTRUCTIONS[step]}**")
        img_file = st.camera_input(f"Chụp ảnh cho bước {step + 1}", key=f"camera_{name}_{step}")
        
        if img_file is not None:
            img = Image.open(img_file)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Phát hiện khuôn mặt và vẽ bounding box
            bboxes, landmarks = detector.detect(image=frame)
            if bboxes is not None and len(bboxes) > 0 and len(landmarks) > 0:
                bbox = bboxes[0]  # Lấy bounding box đầu tiên
                x1, y1, x2, y2, _ = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                st.image(frame, caption="Khuôn mặt với bounding box", use_container_width=True)
                
                # Lưu chỉ phần bounding box làm ảnh gốc
                cropped_face = frame[y1:y2, x1:x2]
                images.append(cropped_face)
                st.session_state[f"step_{name}"] = step + 1
                st.session_state[f"images_{name}"] = images
                st.success(f"Đã chụp ảnh bounding box cho bước {step + 1}")
            else:
                st.warning("Không phát hiện khuôn mặt, vui lòng thử lại!")
            st.rerun()
        
        if step > 0 and st.button("Chụp lại ảnh này", key=f"retake_{name}_{step}_{uuid.uuid4()}"):
            step -= 1
            images.pop()
            st.session_state[f"step_{name}"] = step
            st.session_state[f"images_{name}"] = images
            st.rerun()
        
        if st.button("Chụp lại từ đầu", key=f"restart_{name}_{step}_{uuid.uuid4()}"):
            st.session_state[f"step_{name}"] = 0
            st.session_state[f"images_{name}"] = []
            st.rerun()
    else:
        if len(images) == len(INSTRUCTIONS):
            add_person(name, images)
            st.session_state[f"step_{name}"] = 0
            st.session_state[f"images_{name}"] = []
            st.success("Đã hoàn tất quá trình chụp ảnh, augmentation và lưu đặc trưng!")
        else:
            st.error("Quá trình chụp ảnh chưa hoàn tất!")

# Hàm xử lý video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "output_video.mp4"
    out = cv2.VideoWriter(out_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    
    tracker = BYTETracker(args={"track_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8, "aspect_ratio_thresh": 1.6, "min_box_area": 10}, frame_rate=30)
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        outputs, img_info, bboxes, _ = detector.detect_tracking(image=frame)
        if outputs is not None:
            online_targets = tracker.update(outputs, [img_info["height"], img_info["width"]], (128, 128))
            tracking_tlwhs = []
            tracking_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > 10:
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(tid)
            tracking_image = plot_tracking(img_info["raw_img"], tracking_tlwhs, tracking_ids, names=id_face_mapping, frame_id=frame_id)
            out.write(tracking_image)
        else:
            out.write(img_info["raw_img"])
        frame_id += 1
    
    cap.release()
    out.release()
    return out_path

# Hàm nhận diện thời gian thực
def real_time_recognition():
    # Tải đặc trưng khi bắt đầu nhận diện thời gian thực
    images_names, images_embs = read_features(feature_path=FEATURES_PATH)
    
    cap = cv2.VideoCapture(0)
    tracker = BYTETracker(args={"track_thresh": 0.5, "track_buffer": 30, "match_thresh": 0.8, "aspect_ratio_thresh": 1.6, "min_box_area": 10}, frame_rate=30)
    frame_id = 0
    
    frame_placeholder = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame)
        if outputs is not None:
            online_targets = tracker.update(outputs, [img_info["height"], img_info["width"]], (128, 128))
            tracking_tlwhs = []
            tracking_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                if tlwh[2] * tlwh[3] > 10:
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(tid)
            
            for i, tid in enumerate(tracking_ids):
                if tid not in id_face_mapping:
                    for j, bbox in enumerate(bboxes):
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        face_image = frame[y1:y2, x1:x2]
                        query_emb = get_feature(face_image)
                        score, id_min = compare_encodings(query_emb, images_embs)
                        name = images_names[id_min]
                        
                        score_value = score if np.isscalar(score) else score[0]
                        if score_value > 0.5:
                            id_face_mapping[tid] = f"{name}:{score_value:.2f}"
                        else:
                            id_face_mapping[tid] = "UN_KNOWN"
            
            tracking_image = plot_tracking(img_info["raw_img"], tracking_tlwhs, tracking_ids, names=id_face_mapping, frame_id=frame_id)
            frame_placeholder.image(tracking_image, channels="BGR")
        frame_id += 1

# Hàm chính
def main():
    st.title("Ứng dụng Nhận diện Khuôn mặt")
    option = st.sidebar.selectbox("Chọn chức năng", ["Thêm khuôn mặt", "Nhận diện từ video", "Nhận diện thời gian thực"])

    if option == "Thêm khuôn mặt":
        st.header("Thêm khuôn mặt mới")
        name = st.text_input("Nhập tên người dùng")
        if name:
            capture_faces(name)
    
    elif option == "Nhận diện từ video":
        st.header("Nhận diện khuôn mặt từ video")
        uploaded_file = st.file_uploader("Tải lên video", type=["mp4", "avi"])
        if uploaded_file:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write("Đang xử lý video...")
            output_path = process_video("temp_video.mp4")
            st.video(output_path)
    
    elif option == "Nhận diện thời gian thực":
        st.header("Nhận diện khuôn mặt thời gian thực")
        st.write("Nhấn để bắt đầu nhận diện thời gian thực.")
        if st.button("Bắt đầu"):
            real_time_recognition()

if __name__ == "__main__":
    main()