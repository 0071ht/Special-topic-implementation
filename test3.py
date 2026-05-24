# -*- coding: utf-8 -*-
import os
import sys
import time
import uuid
import random
import shutil
import threading

import cv2
import dxcam
import torch
import torch.nn as nn
import numpy as np
import win32gui  
from ultralytics import YOLO

from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QGraphicsScene, QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(BASE_DIR, "test2.ui")
YOLO_MODEL_PATH = r"C:\Users\littl\runs\detect\train10\weights\best.pt"

FACE_DIR = os.path.join(BASE_DIR, "face")
REAL_DIR = os.path.join(BASE_DIR, "real")
FAKE_DIR = os.path.join(BASE_DIR, "fake")

FAKEFORMER_DIR = r"D:\專題實作測試\FakeFormer-main"
CONFIG_PATH = r"D:\專題實作測試\FakeFormer-main\configs\spatial\swin_sbi_base.yaml"

os.chdir(FAKEFORMER_DIR)

if FAKEFORMER_DIR not in sys.path:
    sys.path.append(FAKEFORMER_DIR)

from package_utils.transform import final_transform, get_center_scale, get_affine_transform
from configs.get_config import load_config
from models import *
from package_utils.image_utils import load_image


class FakeFormerDetector:
    def __init__(self):
        self.cfg = load_config(CONFIG_PATH)
        seed = self.cfg.SEED
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            torch.cuda.manual_seed(seed)
        self.device_count = torch.cuda.device_count()
        self.model = build_model(self.cfg.MODEL, MODELS).to(torch.float64)
        self.model = load_pretrained(self.model, self.cfg.TEST.pretrained)
        if self.use_cuda:
            if self.device_count >= 1:
                self.model = nn.DataParallel(self.model, device_ids=self.cfg.TEST.gpus).cuda()
            else:
                self.model = self.model.cuda()
        self.model.eval()
        self.aspect_ratio = self.cfg.DATASET.IMAGE_SIZE[1] * 1.0 / self.cfg.DATASET.IMAGE_SIZE[0]
        self.pixel_std = 200
        self.rot = 0
        self.transforms = final_transform(self.cfg.DATASET)

    def predict(self, image_path):
        img = load_image(image_path)
        c, s = get_center_scale(img.shape[:2], self.aspect_ratio, self.pixel_std)
        trans = get_affine_transform(c, s, self.rot, self.cfg.DATASET.IMAGE_SIZE)
        input_img = cv2.warpAffine(
            img, trans, (int(self.cfg.DATASET.IMAGE_SIZE[0]), int(self.cfg.DATASET.IMAGE_SIZE[1])), flags=cv2.INTER_LINEAR
        )
        with torch.no_grad():
            st = time.time()
            img_trans = self.transforms(input_img / 255).to(torch.float64)
            img_trans = torch.unsqueeze(img_trans, 0)
            if self.use_cuda:
                img_trans = img_trans.cuda(non_blocking=True)
            outputs = self.model(img_trans)
            cls_outputs = outputs[0]["cls"].sigmoid()
            label_pred = cls_outputs.cpu().numpy()
            score = float(label_pred[0][-1])
            label = "Fake" if score > self.cfg.TEST.threshold else "Real"
            infer_time = time.time() - st
        return label, score, infer_time

# 🛑 視窗 1：專門給「自動監控模式」使用的紅色警告視窗 (只有 Fake 會叫出它)
class FakeWarningDialog(QDialog):
    def __init__(self, parent, image_path, score):
        super().__init__(parent)
        self.setWindowTitle("🚨 系統警報：發現偽造臉孔！")
        self.setFixedSize(400, 500)  
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a1a; }
            QLabel { color: white; }
        """)

        layout = QVBoxLayout(self)

        self.img_label = QLabel()
        pixmap = QPixmap(image_path).scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label)

        self.text_label = QLabel(f"⚠️ 警告！偵測到 Deepfake 影像\n偽造機率：{score:.6f}")
        self.text_label.setStyleSheet("color: #ff3333; font-size: 20px; font-weight: bold;")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.ok_btn = QPushButton("解除警報並繼續偵測")
        self.ok_btn.setStyleSheet("""
            QPushButton { 
                background-color: #aa0000; 
                color: white; 
                font-size: 16px; 
                border-radius: 10px; 
                padding: 10px; 
            }
            QPushButton:pressed { background-color: #ff0000; }
        """)
        self.ok_btn.clicked.connect(self.accept) 
        layout.addWidget(self.ok_btn)

# 🔄 視窗 2：專門給「手動選擇圖片」使用的結果視窗 (會根據 Real/Fake 變色)
class ManualResultDialog(QDialog):
    def __init__(self, parent, image_path, label, score):
        super().__init__(parent)
        self.setFixedSize(400, 500)  
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        
        if label == "Fake":
            self.setWindowTitle("🚨 手動分析：發現偽造影像！")
            theme_color = "#ff3333"
            btn_bg = "#aa0000"
            btn_hover = "#ff0000"
            title_text = f"⚠️ 警告！偵測到 Deepfake\n偽造機率：{score:.4f}"
        else:
            self.setWindowTitle("✅ 手動分析：影像安全")
            theme_color = "#00d4ff"  
            btn_bg = "#007b8f"
            btn_hover = "#00d4ff"
            title_text = f"✅ 安全！此為真實影像\n真實機率：{1 - score:.4f}"

        self.setStyleSheet(f"""
            QDialog {{ background-color: #1a1a1a; }}
            QLabel {{ color: white; }}
            QPushButton {{ 
                background-color: {btn_bg}; 
                color: white; 
                font-size: 16px; 
                border-radius: 10px; 
                padding: 10px; 
                font-weight: bold;
            }}
            QPushButton:pressed {{ background-color: {btn_hover}; color: black; }}
        """)

        layout = QVBoxLayout(self)

        self.img_label = QLabel()
        pixmap = QPixmap(image_path).scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pixmap)
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label)

        self.text_label = QLabel(title_text)
        self.text_label.setStyleSheet(f"color: {theme_color}; font-size: 22px; font-weight: bold;")
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.ok_btn = QPushButton("確認並關閉")
        self.ok_btn.clicked.connect(self.accept) 
        layout.addWidget(self.ok_btn)

class ManualDetectThread(QThread):
    finished = Signal(str, str, float)
    error = Signal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            detector = FakeFormerDetector()
            label, score, _ = detector.predict(self.image_path)
            self.finished.emit(self.image_path, label, score)
        except Exception as e:
            self.error.emit(str(e))

class YoloFakeFormerThread(QThread):
    status_changed = Signal(str)
    fake_detected = Signal(str, float)
    image_classified = Signal(str, str, float)
    mode_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.stop_flag = False
        self.alert_closed_event = threading.Event()
        self.save_cooldown = 2.0  

    def stop(self):
        self.stop_flag = True
        self.alert_closed_event.set()

    def continue_after_alert(self):
        self.alert_closed_event.set()

    def check_paths(self):
        if not os.path.exists(UI_PATH): raise FileNotFoundError(f"找不到 UI 檔案：{UI_PATH}")
        if not os.path.exists(YOLO_MODEL_PATH): raise FileNotFoundError(f"找不到 YOLO 模型：{YOLO_MODEL_PATH}")
        if not os.path.exists(FAKEFORMER_DIR): raise FileNotFoundError(f"找不到 FakeFormer 資料夾：{FAKEFORMER_DIR}")
        if not os.path.exists(CONFIG_PATH): raise FileNotFoundError(f"找不到 FakeFormer 設定檔：{CONFIG_PATH}")

    def prepare_dirs(self):
        os.makedirs(FACE_DIR, exist_ok=True)
        os.makedirs(REAL_DIR, exist_ok=True)
        os.makedirs(FAKE_DIR, exist_ok=True)

    def clear_face_dir(self):
        for file_name in os.listdir(FACE_DIR):
            file_path = os.path.join(FACE_DIR, file_name)
            if os.path.isfile(file_path):
                try: os.remove(file_path)
                except Exception: pass

    def classify_saved_image(self, detector, image_path):
        label, score, infer_time = detector.predict(image_path)
        file_name = os.path.basename(image_path)
        target_path = os.path.join(FAKE_DIR if label == "Fake" else REAL_DIR, file_name)
        shutil.copy2(image_path, target_path)
        self.image_classified.emit(target_path, label, score)

        if label == "Fake":
            self.clear_face_dir()
            self.status_changed.emit("偵測到 Fake，程式暫停中...")
            self.alert_closed_event.clear()
            self.fake_detected.emit(target_path, score)
            self.alert_closed_event.wait()
            self.status_changed.emit("程式繼續偵測...")
            time.sleep(2)

        return label, score, infer_time

    def run(self):
        camera = None
        try:
            self.check_paths()
            self.prepare_dirs()
            self.status_changed.emit("正在載入 YOLO 模型...")
            yolo_model = YOLO(YOLO_MODEL_PATH)
            yolo_device = 0 if torch.cuda.is_available() else "cpu"
            self.status_changed.emit("正在載入 FakeFormer 模型...")
            fakeformer_detector = FakeFormerDetector()
            self.status_changed.emit("模型載入完成，開始偵測...")

            camera = dxcam.create(output_color="BGR")
            camera.start(target_fps=60)
            monitor_window_name = "Monitor View"
            cv2.namedWindow(monitor_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(monitor_window_name, 800, 600)

            face_count = 0
            last_save_time = 0
            current_mode_text = ""

            while not self.stop_flag:
                frame = camera.get_latest_frame()
                if frame is None:
                    cv2.waitKey(1)
                    continue

                try:
                    active_window = win32gui.GetForegroundWindow()
                    window_title = win32gui.GetWindowText(active_window)
                    new_mode_text = current_mode_text 
                    if "系統警報" in window_title or "手動分析" in window_title or "Form" in window_title or "python" in window_title.lower():
                        pass
                    elif "YouTube" in window_title or "Netflix" in window_title or "Twitch" in window_title:
                        self.save_cooldown = 0.5   
                        new_mode_text = "目前狀態：影片模式 (極速偵測)" 
                    elif "Facebook" in window_title or "Instagram" in window_title:
                        self.save_cooldown = 2.0   
                        new_mode_text = "目前狀態：社群模式 (中速偵測)"
                    elif "照片" in window_title or "Photos" in window_title:
                        self.save_cooldown = 5.0   
                        new_mode_text = "目前狀態：圖片模式 (慢速偵測)"
                    if new_mode_text != current_mode_text and new_mode_text != "":
                        current_mode_text = new_mode_text
                        self.mode_changed.emit(current_mode_text)
                except Exception:
                    pass 

                results = yolo_model.predict(frame, conf=0.6, verbose=False, device=yolo_device)
                current_time = time.time()
                can_save = (current_time - last_save_time) > self.save_cooldown

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        padding_ratio = 0.20  
                        box_w, box_h = x2 - x1, y2 - y1
                        pad_x, pad_y = int(box_w * padding_ratio), int(box_h * padding_ratio)
                        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                        if x2 <= x1 or y2 <= y1: continue
                        face_roi = frame[y1:y2, x1:x2]

                        if face_roi.size > 0 and can_save:
                            face_count += 1
                            unique_id = uuid.uuid4().hex[:6]
                            file_name = f"face_{face_count}_{int(current_time)}_{unique_id}.jpg"
                            image_path = os.path.join(FACE_DIR, file_name)
                            cv2.imwrite(image_path, face_roi)
                            last_save_time = current_time
                            can_save = False

                            label, score, _ = self.classify_saved_image(fakeformer_detector, image_path)
                            color = (0, 0, 255) if label == "Fake" else (0, 255, 0)
                            cv2.putText(frame, f"{label} {score:.3f}", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow(monitor_window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            self.status_changed.emit(f"偵測結束，共存入 {face_count} 張人臉圖片")
        except Exception as e:
            self.status_changed.emit("發生錯誤：\n" + str(e))
        finally:
            if camera is not None: camera.stop()
            cv2.destroyAllWindows()


class HBMainWindow:
    def __init__(self):
        loader = QUiLoader()
        self.ui = loader.load(UI_PATH)
        
        self.ui.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.ui.setFixedSize(300, 550) 

        self.detect_thread = None
        self.manual_thread = None
        self.current_cooldown = 2.0  

        self.ui.StartButton.setEnabled(True)
        self.ui.StopButton.setEnabled(False)
        self.ui.ManualButton.setEnabled(True) 
        
        self.ui.ROF.setText("雷達待命中...")
        self.ui.ROF.setWordWrap(True)

        self.ui.StartButton.clicked.connect(self.start_detection)
        self.ui.StopButton.clicked.connect(self.stop_detection)
        self.ui.ManualButton.clicked.connect(self.manual_detection) 
        self.ui.freqSlider.valueChanged.connect(self.change_frequency)

        self.change_frequency()

    def manual_detection(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self.ui, 
            "選擇要分析的圖片", 
            "", 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return 

        self.ui.ROF.setText("手動模式：正在載入模型並分析圖片...")
        self.ui.ManualButton.setEnabled(False) 
        
        self.manual_thread = ManualDetectThread(file_path)
        self.manual_thread.finished.connect(self.show_manual_result)
        self.manual_thread.error.connect(self.show_manual_error)
        self.manual_thread.start()

    def show_manual_result(self, image_path, label, score):
        self.ui.ManualButton.setEnabled(True)
        self.ui.ROF.setText(f"手動判斷完成: {label}")
        
        # 呼叫專屬於「手動模式」的變色視窗
        dialog = ManualResultDialog(self.ui, image_path, label, score)
        dialog.exec()

    def show_manual_error(self, error_msg):
        self.ui.ManualButton.setEnabled(True)
        self.ui.ROF.setText(f"手動判斷失敗: {error_msg}")

    def change_frequency(self):
        val = self.ui.freqSlider.value()
        if val == 1:
            self.ui.freqLabel.setText("通知頻率：慢速 (每 5 秒)")
            self.current_cooldown = 5.0
        elif val == 2:
            self.ui.freqLabel.setText("通知頻率：中速 (每 2 秒)")
            self.current_cooldown = 2.0
        elif val == 3:
            self.ui.freqLabel.setText("通知頻率：極速 (每 0.5 秒)")
            self.current_cooldown = 0.5

        if self.detect_thread is not None:
            self.detect_thread.save_cooldown = self.current_cooldown

    def start_detection(self):
        if self.detect_thread is not None and self.detect_thread.isRunning(): return
        self.ui.StartButton.setEnabled(False)
        self.ui.StopButton.setEnabled(True)
        self.ui.ManualButton.setEnabled(False) 
        self.ui.ROF.setText("準備啟動 YOLO + FakeFormer...")

        self.detect_thread = YoloFakeFormerThread()
        self.detect_thread.save_cooldown = self.current_cooldown 
        self.detect_thread.status_changed.connect(self.update_status)
        self.detect_thread.fake_detected.connect(self.show_fake_warning)
        self.detect_thread.image_classified.connect(self.handle_classified_image)
        self.detect_thread.finished.connect(self.detection_finished)
        self.detect_thread.mode_changed.connect(self.update_mode_label)
        self.detect_thread.start()

    def stop_detection(self):
        if self.detect_thread is not None and self.detect_thread.isRunning():
            self.detect_thread.stop()
            self.ui.ROF.setText("正在關閉雷達...")
            self.ui.StopButton.setEnabled(False)

    def update_status(self, text):
        self.ui.ROF.setText(text)

    def update_mode_label(self, mode_text):
        self.ui.freqLabel.setText(mode_text)

    def handle_classified_image(self, image_path, label, score):
        self.ui.ROF.setText(f"最新結果: {label} ({score:.4f})")

    def show_fake_warning(self, image_path, score):
        # 呼叫專屬於「自動模式」的純紅色警告視窗
        dialog = FakeWarningDialog(self.ui, image_path, score)
        dialog.exec()
        if self.detect_thread is not None:
            self.detect_thread.continue_after_alert()

    def detection_finished(self):
        self.ui.StartButton.setEnabled(True)
        self.ui.StopButton.setEnabled(False)
        self.ui.ManualButton.setEnabled(True)
        self.ui.ROF.setText("雷達已關閉")
        self.detect_thread = None

if __name__ == "__main__":
    app = QApplication([])
    hb_window = HBMainWindow()
    hb_window.ui.show()
    app.exec()
    
