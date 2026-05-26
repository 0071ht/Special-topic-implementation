import os  # 匯入 os 模組，用於進行作業系統層級的檔案與目錄路徑操作
import sys  # 匯入 sys 模組，用於操作 Python 執行環境及系統路徑變數
import time  # 匯入 time 模組，用於計算推論時間以及處理冷卻等待時間
import uuid  # 匯入 uuid 模組，用於產生唯一的亂數 ID，避免圖片檔名重複
import random  # 匯入 random 模組，用於設定隨機種子，確保模型預測結果具備可重現性
import threading  # 匯入 threading 模組，用於多執行緒控制中的事件鎖 (Event)，避免介面卡死

import cv2  # 匯入 OpenCV (cv2) 模組，用於影像處理、寫入實體檔案
import dxcam  # 匯入 dxcam 模組，這是一款高效能的 Windows 螢幕擷取工具，適合高幀率截圖
import torch  # 匯入 PyTorch 核心模組，用於深度學習模型的張量運算與推論
import numpy as np  # 匯入 NumPy 模組，用於處理高效能的矩陣與數值運算
import win32gui  # 匯入 win32gui 模組，用於獲取 Windows 當前最上層 (前景) 視窗的標題
from ultralytics import YOLO  # 從 ultralytics 套件中匯入 YOLO 類別，用於載入人臉偵測模型

# 引入 PySide6 UI 相關套件，用於建構圖形化使用者介面 (GUI)
from PySide6.QtUiTools import QUiLoader  # 匯入 QUiLoader，用於動態載入 Qt Designer 製作的 .ui 檔案
from PySide6.QtGui import QPixmap  # 匯入 QPixmap，用於在介面上處理與顯示圖片
from PySide6.QtCore import Qt, QThread, Signal  # 匯入 Qt 核心元件、子執行緒 (QThread) 及自訂訊號 (Signal)
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog, QVBoxLayout, QLabel, QPushButton, QFileDialog  # 匯入各種 UI 視窗與控制元件

# --- 1. 路徑與環境初始化設定 --- 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 取得目前這支 Python 程式檔所在的絕對目錄路徑
UI_PATH = os.path.join(BASE_DIR, "test2.ui")  # 組合出主介面 UI 檔 (test2.ui) 的完整路徑
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "Weights", "best.pt")  # 設定訓練好的 YOLO 人臉偵測模型權重路徑

# 🟢 修正：只保留唯一的戰利品資料夾 "fake"，其餘全數拔除
FAKE_DIR = os.path.join(BASE_DIR, "fake")  

# FakeFormer 開源專案的核心路徑設定 
FAKEFORMER_DIR = os.path.join(BASE_DIR, "FakeFormer-main")  # 設定 FakeFormer 專案的根目錄位置
CONFIG_PATH = os.path.join(FAKEFORMER_DIR, "configs", "spatial", "swin_sbi_base.yaml")  # 設定 FakeFormer 專屬的 YAML 模型架構設定檔路徑

# 將工作目錄切換至 FakeFormer 資料夾，並加入環境變數，以便順利 import 其內部套件 
if os.path.exists(FAKEFORMER_DIR):
    os.chdir(FAKEFORMER_DIR)  # 將 Python 當前的工作目錄強制切換到 FakeFormer 專案資料夾內
    if FAKEFORMER_DIR not in sys.path:  # 檢查 FakeFormer 目錄是否已在 Python 系統路徑中
        sys.path.append(FAKEFORMER_DIR)  # 如果不在系統路徑中，則將其加入，確保可匯入該目錄下的模組

    # 自 FakeFormer 專案中 import 預處理與模型架構 
    from package_utils.transform import final_transform, get_center_scale, get_affine_transform  # 匯入影像轉換與仿射變換所需的函式
    from configs.get_config import load_config  # 匯入讀取 YAML 設定檔的專用函式
    from models import * # 匯入 models 資料夾下的所有模型架構定義
    from package_utils.image_utils import load_image  # 匯入讀取影像的專用工具函式
else:
    print(f"警告：找不到 FakeFormer 目錄 {FAKEFORMER_DIR}")


# --- 2. FakeFormer 辨識核心類別 --- 
class FakeFormerDetector:  
    def __init__(self):  
        self.cfg = load_config(CONFIG_PATH)  
                
        self.use_cuda = torch.cuda.is_available()  
                
        self.model = build_model(self.cfg.MODEL, MODELS).to(torch.float64)  
        self.model = load_pretrained(self.model, self.cfg.TEST.pretrained)  
        
        if self.use_cuda:  
            self.model = self.model.cuda()  
                
        self.model.eval()  
        
        self.aspect_ratio = self.cfg.DATASET.IMAGE_SIZE[1] * 1.0 / self.cfg.DATASET.IMAGE_SIZE[0]  
        self.pixel_std = 200  
        self.rot = 0  
        self.transforms = final_transform(self.cfg.DATASET)  

    def predict_array(self, img_array):  
        """ 直接對記憶體內的影像陣列進行預測，避開 Real 影像寫入硬碟的開銷 """
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
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

    def predict(self, image_path):  
        """ 手動模式專用：讀取實體檔案進行預測 """  
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


# --- 3. UI 視窗類別設定 --- 

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


class ManualResultDialog(QDialog):  
    def __init__(self, parent, image_path, label, score):  
        super().__init__(parent)  
        self.setFixedSize(400, 500)  
        self.setWindowFlag(Qt.WindowStaysOnTopHint) 
        
        if label == "Fake":  
            self.setWindowTitle("手動分析：發現偽造影像！")  
            theme_color = "#ff3333"  
            btn_bg = "#aa0000"  
            btn_hover = "#ff0000"  
            title_text = f"⚠️ 警告！偵測到 Deepfake\n偽造機率：{score:.4f}"  
        else:  
            self.setWindowTitle("手動分析：影像安全")  
            theme_color = "#00d4ff"  
            btn_bg = "#007b8f"  
            btn_hover = "#00d4ff"  
            title_text = f"安全！此為真實影像\n真實機率：{1 - score:.4f}"  

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


# --- 4. 子執行緒 (QThread) 類別實作 --- 

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
    error_occurred = Signal(str)  

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
        # 🟢 修正：只確保唯一的 fake 資料夾存在
        os.makedirs(FAKE_DIR, exist_ok=True)  

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
            self.status_changed.emit("模型載入完成，開始背景偵測...")  

            camera = dxcam.create(output_color="BGR")  
            camera.start(target_fps=60)  

            face_count = 0  
            last_save_time = 0  
            current_mode_text = ""  

            while not self.stop_flag:  
                frame = camera.get_latest_frame()  
                if frame is None:  
                    time.sleep(0.001)  
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
                        new_mode_text = "目前狀態：社群模式 (中速監控)"  
                    elif "照片" in window_title or "Photos" in window_title:  
                        self.save_cooldown = 5.0  
                        new_mode_text = "目前狀態：圖片模式 (慢速節能)"  
                        
                    if new_mode_text != current_mode_text and new_mode_text != "":  
                        current_mode_text = new_mode_text  
                        self.mode_changed.emit(current_mode_text)  
                except Exception:  
                    pass  

                results = yolo_model.predict(frame, conf=0.8, verbose=False, device=yolo_device)  
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
                            label, score, _ = fakeformer_detector.predict_array(face_roi)  
                            
                            last_save_time = current_time  
                            can_save = False 

                            if label == "Fake":
                                face_count += 1  
                                unique_id = uuid.uuid4().hex[:6]  
                                file_name = f"fake_{face_count}_{int(current_time)}_{unique_id}.jpg"  
                                image_path = os.path.join(FAKE_DIR, file_name)  
                                
                                cv2.imwrite(image_path, face_roi)  
                                
                                self.image_classified.emit(image_path, label, score)  
                                self.status_changed.emit("偵測到 Fake，程式暫停中...")  
                                
                                self.alert_closed_event.clear()  
                                self.fake_detected.emit(image_path, score)  
                                self.alert_closed_event.wait()  
                                
                                self.status_changed.emit("程式繼續偵測...")  
                                time.sleep(2)  
                            else:
                                self.image_classified.emit("", "Real", score)
                
            self.status_changed.emit(f"偵測結束，本輪共截獲 {face_count} 張 Fake 圖片")  
        except Exception as e:  
            self.error_occurred.emit(str(e))  
        finally:  
            if camera is not None: camera.stop()  


# --- 5. 主視窗 UI 控制器 --- 
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
        
        if not file_path: return  

        self.ui.ROF.setText("手動模式：正在載入模型並分析圖片...")  
        self.ui.ManualButton.setEnabled(False)  
        
        self.manual_thread = ManualDetectThread(file_path)  
        self.manual_thread.finished.connect(self.show_manual_result)  
        self.manual_thread.error.connect(self.show_manual_error)  
        self.manual_thread.start()  

    def show_manual_result(self, image_path, label, score):  
        self.ui.ManualButton.setEnabled(True)  
        self.ui.ROF.setText(f"手動判斷完成: {label}")  
        dialog = ManualResultDialog(self.ui, image_path, label, score)  
        dialog.exec()  

    def show_manual_error(self, error_msg):  
        self.ui.ManualButton.setEnabled(True)  
        self.ui.ROF.setText(f"手動判斷失敗")  
        QMessageBox.critical(self.ui, "手動模式錯誤", f"手動分析發生異常：\n{error_msg}")

    def change_frequency(self):  
        val = self.ui.freqSlider.value()  
        if val == 1:  
            self.ui.freqLabel.setText("通知頻率：慢速")  
            self.current_cooldown = 5.0  
        elif val == 2:  
            self.ui.freqLabel.setText("通知頻率：中速")  
            self.current_cooldown = 2.0  
        elif val == 3:  
            self.ui.freqLabel.setText("通知頻率：極速")  
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
        self.detect_thread.error_occurred.connect(self.show_auto_error)  
        
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
        if "極速" in mode_text or "影片" in mode_text:
            self.ui.freqSlider.setValue(3)
        elif "中速" in mode_text or "社群" in mode_text:
            self.ui.freqSlider.setValue(2)
        elif "慢速" in mode_text or "圖片" in mode_text:
            self.ui.freqSlider.setValue(1)

    def handle_classified_image(self, image_path, label, score):  
        self.ui.ROF.setText("雷達運作中... 掃描螢幕人臉")  

    def show_fake_warning(self, image_path, score):  
        dialog = FakeWarningDialog(self.ui, image_path, score)  
        dialog.exec()  
        if self.detect_thread is not None:  
            self.detect_thread.continue_after_alert()  

    def show_auto_error(self, error_msg):  
        self.ui.ROF.setText("系統異常中斷")  
        QMessageBox.critical(self.ui, "自動模式核心錯誤", f"雷達無法正常工作，具體原因如下：\n\n{error_msg}")

    def detection_finished(self):  
        self.ui.StartButton.setEnabled(True)  
        self.ui.StopButton.setEnabled(False)  
        self.ui.ManualButton.setEnabled(True)  
        if self.ui.ROF.text() != "系統異常中斷":
            self.ui.ROF.setText("雷達已關閉")  
        self.detect_thread = None  


# --- 6. 程式進入點 --- 
if __name__ == "__main__":  
    app = QApplication([])  
    hb_window = HBMainWindow()  
    hb_window.ui.show()  
    app.exec()
    
