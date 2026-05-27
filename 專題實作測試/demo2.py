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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
UI_PATH = os.path.join(BASE_DIR, "test2.ui")  
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "Weights", "best.pt")  

FAKE_DIR = os.path.join(BASE_DIR, "fake")  

FAKEFORMER_DIR = os.path.join(BASE_DIR, "FakeFormer-main")  
CONFIG_PATH = os.path.join(FAKEFORMER_DIR, "configs", "spatial", "swin_sbi_base.yaml")  

# 將工作目錄切換至 FakeFormer 資料夾，並加入環境變數，以便順利 import 其內部套件
if os.path.exists(FAKEFORMER_DIR):  
    os.chdir(FAKEFORMER_DIR)  
    if FAKEFORMER_DIR not in sys.path:  
        sys.path.append(FAKEFORMER_DIR)  

    # 自 FakeFormer 專案中 import 預處理與模型架構
    from package_utils.transform import final_transform, get_center_scale, get_affine_transform  # 匯入影像轉換與仿射變換所需的函式
    from configs.get_config import load_config  # 匯入讀取 YAML 設定檔的專用函式
    from models import * # 匯入 models 資料夾下的所有模型架構定義
    from package_utils.image_utils import load_image  # 匯入讀取影像的專用工具函式
else:  
    print(f"警告：找不到 FakeFormer 目錄 {FAKEFORMER_DIR}") 


# --- 2. FakeFormer 辨識核心 ---
class FakeFormerDetector:  
    def __init__(self):  
        self.cfg = load_config(CONFIG_PATH)  # (FakeFormer 官方用法) 呼叫自訂函式載入 YAML 模型設定檔
        self.use_cuda = torch.cuda.is_available()  
                
        self.model = build_model(self.cfg.MODEL, MODELS).to(torch.float64)  # (FakeFormer 官方用法) 建立模型架構，並將張量轉為 float64 精度
        self.model = load_pretrained(self.model, self.cfg.TEST.pretrained)  # (FakeFormer 官方用法) 載入預先訓練好的 FakeFormer 模型權重
        
        if self.use_cuda:  
            self.model = self.model.cuda()  
            
        self.model.eval()  
        
        self.aspect_ratio = self.cfg.DATASET.IMAGE_SIZE[1] * 1.0 / self.cfg.DATASET.IMAGE_SIZE[0]  # 計算影像的長寬比
        self.pixel_std = 200  # 設定影像正規化用的像素標準差常數
        self.rot = 0  # 設定影像旋轉角度初始值為 0
        self.transforms = final_transform(self.cfg.DATASET)  # (FakeFormer 官方用法) 初始化影像預處理轉換流程

    def predict_array(self, img_array):  # 定義傳入 NumPy 陣列進行預測的方法
        """ 直接對記憶體內的影像陣列進行預測 """
        img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)  # cv2.cvtColor 將 BGR 色彩空間轉換為 RGB
        
        c, s = get_center_scale(img.shape[:2], self.aspect_ratio, self.pixel_std)  # (FakeFormer 官方用法) 取得影像的中心點與縮放比例
        trans = get_affine_transform(c, s, self.rot, self.cfg.DATASET.IMAGE_SIZE)  # (FakeFormer 官方用法) 計算仿射變換矩陣
        input_img = cv2.warpAffine(  # cv2.warpAffine 執行影像的仿射幾何變換
            img, trans, (int(self.cfg.DATASET.IMAGE_SIZE[0]), int(self.cfg.DATASET.IMAGE_SIZE[1])), flags=cv2.INTER_LINEAR  # 設定輸出尺寸與雙線性插值法
        )  # 結束 warpAffine 呼叫
        
        with torch.no_grad():  # 停用梯度計算，減少推論時的記憶體消耗並提升速度
            st = time.time()  # 記錄推論開始的時間點
            img_trans = self.transforms(input_img / 255).to(torch.float64)  # 將像素值歸一化至 0~1 並套用預處理，轉為 float64
            img_trans = torch.unsqueeze(img_trans, 0)  # 在第 0 維增加 Batch 維度，形狀變為 (1, C, H, W)
            if self.use_cuda: 
                img_trans = img_trans.cuda(non_blocking=True)  
                
            outputs = self.model(img_trans)  # 將影像輸入模型進行前向傳播推論
            cls_outputs = outputs[0]["cls"].sigmoid()  # (FakeFormer 官方用法) 提取分類輸出並套用 Sigmoid 函數轉為 0~1 的機率值
            label_pred = cls_outputs.cpu().numpy()  # 將結果移回 CPU 記憶體並轉換為 NumPy 陣列
            
            score = float(label_pred[0][-1])  # 提取最終的偽造機率分數並轉為 Python 原生 float
            label = "Fake" if score > self.cfg.TEST.threshold else "Real"  # 依據設定檔的閾值判定為偽造或真實
            infer_time = time.time() - st  # 計算總推論花費時間
            
        return label, score, infer_time  # 回傳預測標籤、機率分數與推論時間

    def predict(self, image_path):  # 定義傳入實體檔案路徑進行預測的方法
        """ 讀取實體檔案進行預測 (手動模式使用，不經由 YOLO 裁切) """  
        img = load_image(image_path)  # (FakeFormer 官方用法) 呼叫自訂工具函式讀取硬碟中的影像檔案
        
        c, s = get_center_scale(img.shape[:2], self.aspect_ratio, self.pixel_std)  # (FakeFormer 官方用法) 取得影像的中心點與縮放比例
        trans = get_affine_transform(c, s, self.rot, self.cfg.DATASET.IMAGE_SIZE)  # (FakeFormer 官方用法) 計算仿射變換矩陣
        input_img = cv2.warpAffine(  # 執行影像的仿射變換
            img, trans, (int(self.cfg.DATASET.IMAGE_SIZE[0]), int(self.cfg.DATASET.IMAGE_SIZE[1])), flags=cv2.INTER_LINEAR  # 設定輸出尺寸與插值參數
        )  # 結束 warpAffine 呼叫
        
        with torch.no_grad():  # 停用梯度計算以節省資源
            st = time.time()  # 記錄起始時間
            img_trans = self.transforms(input_img / 255).to(torch.float64)  # 歸一化、套用轉換並調整精度
            img_trans = torch.unsqueeze(img_trans, 0)  # 增加 Batch 維度
            if self.use_cuda:  # 若啟用 GPU
                img_trans = img_trans.cuda(non_blocking=True)  # 轉移至 GPU 運算
                
            outputs = self.model(img_trans)  # 進行模型推論
            cls_outputs = outputs[0]["cls"].sigmoid()  # (FakeFormer 官方用法) 套用 Sigmoid 取得機率
            label_pred = cls_outputs.cpu().numpy()  # 轉回 CPU 及 NumPy 格式
            
            score = float(label_pred[0][-1])  # 提取機率分數
            label = "Fake" if score > self.cfg.TEST.threshold else "Real"  # 判斷真偽標籤
            infer_time = time.time() - st  # 計算花費時間
            
        return label, score, infer_time  


# --- 3. UI 視窗類別設定 ---

class FakeWarningDialog(QDialog):  
    def __init__(self, parent, image_path, score):  
        super().__init__(parent)  
        self.setWindowTitle("系統警報：發現偽造臉孔！")  
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

        self.text_label = QLabel(f"警告!偵測到 Deepfake 影像\n偽造機率：{score:.6f}")  
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

# --- 4. 定義手動模式結果的對話框 ---
class ManualResultDialog(QDialog):  
    def __init__(self, parent, image_path, label, score):  
        super().__init__(parent)  
        self.setFixedSize(400, 500)  
        self.setWindowFlag(Qt.WindowStaysOnTopHint)  
        
        if label == "Fake":  
            self.setWindowTitle("手動分析：發現偽造影像!")  
            theme_color = "#ff3333"  
            btn_bg = "#aa0000" 
            btn_hover = "#ff0000"  
            title_text = f"警告!偵測到 Deepfake\n偽造機率：{score:.4f}"  
        else:  
            self.setWindowTitle("手動分析：影像安全")  
            theme_color = "#00d4ff"  
            btn_bg = "#007b8f"  
            btn_hover = "#00d4ff"  
            title_text = f"此為真實影像\n真實機率：{1 - score:.4f}"  

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
        
        if image_path:  
            pixmap = QPixmap(image_path).scaled(360, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)  
            self.img_label.setPixmap(pixmap)  
        else:  
            self.img_label.setText("未偵測到人臉")  
            self.img_label.setStyleSheet("font-size: 18px; font-weight: bold;")  
        self.img_label.setAlignment(Qt.AlignCenter)  
        layout.addWidget(self.img_label) 
        
        self.text_label = QLabel(title_text)  
        self.text_label.setStyleSheet(f"color: {theme_color}; font-size: 22px; font-weight: bold;")  
        self.text_label.setAlignment(Qt.AlignCenter)  
        layout.addWidget(self.text_label)  

        self.ok_btn = QPushButton("確認並關閉")  
        self.ok_btn.clicked.connect(self.accept)  
        layout.addWidget(self.ok_btn)  


# --- 5. 子執行緒 (QThread) 類別實作 ---
class ManualDetectThread(QThread):  # 定義繼承自 QThread 的手動偵測執行緒類別
    finished = Signal(str, str, float)  # 定義完整訊號 (圖片路徑, 標籤, 分數)
    error = Signal(str)  # 定義錯誤訊號，傳遞錯誤訊息字

    def __init__(self, image_path):  
        super().__init__()  # 呼叫 QThread 父類別初始化
        self.image_path = image_path  # 儲存要分析的圖片路徑

    def run(self):  # QThread 必須實作的執行緒主體方法
        try:  # 使用 try-except 捕捉潛在錯誤
            # 初始化 YOLO 模型
            yolo_model = YOLO(YOLO_MODEL_PATH)  # (YOLO 官方用法) 載入 ultralytics 的 YOLO 模型
            yolo_device = 0 if torch.cuda.is_available() else "cpu"  

            # 使用 imdecode 讀取影像，避免中文路徑讀取失敗
            img_data = np.fromfile(self.image_path, dtype=np.uint8)  # Numpy 讀取二進位檔案資料
            frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)  # OpenCV 解碼記憶體中的影像資料
            if frame is None: raise FileNotFoundError(f"無法讀取影像檔案：{self.image_path}")  

            results = yolo_model.predict(frame, conf=0.8, verbose=False, device=yolo_device)  # (YOLO 官方用法) 執行 YOLO 人臉偵測，設定信心度 0.8 

            # 檢查是否偵測到人臉
            if len(results) == 0 or len(results[0].boxes) == 0:  # (YOLO 官方用法) 檢查 YOLO 回傳結果的框數量
                self.finished.emit(None, "Real", 0.0)  
                return  

            # 取得第一個人臉框 ( results[0].boxes[0] )
            box = results[0].boxes[0]  # (YOLO 官方用法) 取出第一個偵測到的物件框
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # (YOLO 官方用法) 將座標張量轉換為整數格式
            
            
            padding_ratio = 0.10  
            box_w, box_h = x2 - x1, y2 - y1  
            pad_x, pad_y = int(box_w * padding_ratio), int(box_h * padding_ratio)  
            x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y  
            
            x1, y1 = max(0, x1), max(0, y1)  
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)  

            if x2 <= x1 or y2 <= y1:  
                self.finished.emit(None, "Real", 0.0)  
                return  
            
            face_roi = frame[y1:y2, x1:x2]  # 裁切出人臉的座標

            # 如果成功裁切人臉，則使用 FakeFormer 進行預測
            if face_roi.size > 0:  
                detector = FakeFormerDetector()  
                
                label, score, _ = detector.predict_array(face_roi)  # 傳入記憶體陣列進行真偽預測
                
                current_time = int(time.time())  
                unique_id = uuid.uuid4().hex[:6] 
                
                prefix = "manual_fake" if label == "Fake" else "manual_real"
                file_name = f"{prefix}_{current_time}_{unique_id}.jpg"  
                image_path = os.path.join(FAKE_DIR, file_name)  
                
                # 確保資料夾存在，避免存檔失敗
                os.makedirs(FAKE_DIR, exist_ok=True)  
                
                is_success, im_buf_arr = cv2.imencode(".jpg", face_roi)  
                if is_success:  
                    im_buf_arr.tofile(image_path)  
                else:  
                    print(f"警告：圖片編碼失敗，無法儲存至 {image_path}")  
                
                # 發送訊號給 UI 顯示
                self.finished.emit(image_path, label, score)  # 發送完成訊號觸發 UI 更新
            else:  
                self.finished.emit(None, "Real", 0.0)  

        except Exception as e:  # 檢查所有未預期的錯誤
            self.error.emit(str(e))  # 發送錯誤訊號給主執行緒

# --- 6. 定義手動模式結果的對話框 ---
class YoloFakeFormerThread(QThread):  
    status_changed = Signal(str)  # 定義狀態變更訊號
    fake_detected = Signal(str, float)  # 定義發現偽造的訊號
    image_classified = Signal(str, str, float)  # 定義影像分類完成的訊號
    mode_changed = Signal(str)  # 定義模式變更的訊號
    error_occurred = Signal(str)  # 定義錯誤發生的訊號

    def __init__(self):  
        super().__init__()  
        self.stop_flag = False  
        self.alert_closed_event = threading.Event()  # 建立執行緒事件鎖，用於等待 UI 關閉
        self.save_cooldown = 2.0  # 預設儲存與判斷的冷卻時間為 2 秒

    def stop(self):  
        self.stop_flag = True  
        self.alert_closed_event.set()  # 釋放事件鎖，避免執行緒卡死在

    def continue_after_alert(self):  # 警報解除後繼續的方法
        self.alert_closed_event.set()  # 釋放事件鎖，讓執行緒繼續運作

    def check_paths(self):  # 檢查檔案
        if not os.path.exists(UI_PATH): raise FileNotFoundError(f"找不到 UI 檔案：{UI_PATH}") 
        if not os.path.exists(YOLO_MODEL_PATH): raise FileNotFoundError(f"找不到 YOLO 模型：{YOLO_MODEL_PATH}")  
        if not os.path.exists(FAKEFORMER_DIR): raise FileNotFoundError(f"找不到 FakeFormer 資料夾：{FAKEFORMER_DIR}")  
        if not os.path.exists(CONFIG_PATH): raise FileNotFoundError(f"找不到 FakeFormer 設定檔：{CONFIG_PATH}")  

    def prepare_dirs(self):  
        os.makedirs(FAKE_DIR, exist_ok=True)  # 建立處存Fake影像的資料夾

    def run(self):  
        camera = None  
        try:  # 錯誤檢查區塊
            self.check_paths()  
            self.prepare_dirs()  
            
            self.status_changed.emit("正在載入 YOLO 模型...")  # 發送狀態更新給 UI
            yolo_model = YOLO(YOLO_MODEL_PATH)  # (YOLO 官方用法) 載入 YOLO 模型
            yolo_device = 0 if torch.cuda.is_available() else "cpu"  
            
            self.status_changed.emit("正在載入 FakeFormer 模型...")  
            fakeformer_detector = FakeFormerDetector()  # 實例化 FakeFormer 類別
            self.status_changed.emit("模型載入完成，開始背景偵測...")  

            camera = dxcam.create(output_color="BGR")  # 建立 DXcam 截圖物件，輸出 BGR 格式
            camera.start(target_fps=60)  # 開始背景非同步擷取螢幕

            face_count = 0  
            last_save_time = 0  
            current_mode_text = ""  

            while not self.stop_flag:  # 停止指令迴圈
                frame = camera.get_latest_frame()  # 取得最新一張螢幕截圖
                if frame is None:  # 如果畫面尚未準備好
                    time.sleep(0.001)  # 短暫休眠避免佔用過多 CPU
                    continue  

                try:  # 取得最上層視窗標題的區塊
                    active_window = win32gui.GetForegroundWindow()  
                    window_title = win32gui.GetWindowText(active_window)  
                    new_mode_text = current_mode_text  
                    
                    if "系統警報" in window_title or "手動分析" in window_title or "Form" in window_title or "python" in window_title.lower():  
                        pass  
                    elif "YouTube" in window_title or "Netflix" in window_title or "Twitch" in window_title:  # 如果是影音平台
                        self.save_cooldown = 0.5  # 設定截圖等待速度為急速
                        new_mode_text = "目前狀態：影片模式"  
                    elif "Facebook" in window_title or "Instagram" in window_title:  # 如果是社群平台
                        self.save_cooldown = 2.0  # 設定截圖等待速度為中速
                        new_mode_text = "目前狀態：社群模式"  # 更新模式文字
                    elif "照片" in window_title or "Photos" in window_title:  # 如果是圖片檢視器
                        self.save_cooldown = 5.0  # 設定截圖等待速度為慢速
                        new_mode_text = "目前狀態：圖片模式"  # 更新模式文字
                        
                    if new_mode_text != current_mode_text and new_mode_text != "":  
                        current_mode_text = new_mode_text  
                        self.mode_changed.emit(current_mode_text)  # 發送模式變更訊號給 UI
                except Exception:  
                    pass  

                results = yolo_model.predict(frame, conf=0.8, verbose=False, device=yolo_device)  # (YOLO 官方用法) 執行 YOLO 偵測
                current_time = time.time()  
                can_save = (current_time - last_save_time) > self.save_cooldown 

                for result in results:  # (YOLO 官方用法) 走訪偵測結果 (通常只有一個 result 代表當前 frame)
                    for box in result.boxes:  # (YOLO 官方用法) 走訪畫面中的所有偵測框
                        x1, y1, x2, y2 = map(int, box.xyxy[0])  # (YOLO 官方用法) 取得框的左上與右下座標
                        
                        padding_ratio = 0.20  
                        box_w, box_h = x2 - x1, y2 - y1  
                        pad_x, pad_y = int(box_w * padding_ratio), int(box_h * padding_ratio)  
                        x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y  
                        
                        x1, y1 = max(0, x1), max(0, y1)  
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)  

                        if x2 <= x1 or y2 <= y1: continue  
                        face_roi = frame[y1:y2, x1:x2]  # 裁切出臉部座標

                        if face_roi.size > 0 and can_save:  
                            label, score, _ = fakeformer_detector.predict_array(face_roi)  # 進行真偽檢測
                            
                            last_save_time = current_time  # 更新最後判斷時間
                            can_save = False  
                            
                            if label == "Fake":  # 紀錄判定為Fake的圖片
                                face_count += 1  
                                unique_id = uuid.uuid4().hex[:6]  
                                file_name = f"fake_{face_count}_{int(current_time)}_{unique_id}.jpg"  
                                image_path = os.path.join(FAKE_DIR, file_name)  
                                is_success, im_buf_arr = cv2.imencode(".jpg", face_roi)  
                                if is_success:  
                                    im_buf_arr.tofile(image_path)  
                                
                                self.image_classified.emit(image_path, label, score)  
                                self.status_changed.emit("偵測到 Fake，程式暫停中...")  
                                
                                self.alert_closed_event.clear()  # 重置事件鎖為阻塞狀態
                                self.fake_detected.emit(image_path, score)  # 發送發現偽造的訊號的警示視窗
                                self.alert_closed_event.wait()  # 停止當前執行緒直到警示視窗關閉
                                
                                self.status_changed.emit("程式繼續偵測...")  # UI 關閉後更新狀態
                                time.sleep(2)  
                            else:  # 如果判定為真實
                                self.image_classified.emit("", "Real", score)  # 發送安全
                
            self.status_changed.emit(f"偵測結束，本輪共截獲 {face_count} 張 Fake 圖片")  
        except Exception as e:  # 檢查意外錯誤
            self.error_occurred.emit(str(e)) 
        finally:  
            if camera is not None: camera.stop() 


# --- 7. 主視窗 UI 控制器 --- 
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

        self.ui.StartButton.clicked.connect(self.start_detection)  # 設定開始按鈕點擊事件
        self.ui.StopButton.clicked.connect(self.stop_detection)  # 設定停止按鈕點擊事件
        self.ui.ManualButton.clicked.connect(self.manual_detection)  # 設定手動按鈕點擊事件
        self.ui.freqSlider.valueChanged.connect(self.change_frequency)  # 設定滑桿數值變更事件
        self.change_frequency() 

    def manual_detection(self):  # 處理手動偵測的方法
        file_path, _ = QFileDialog.getOpenFileName(  
            self.ui,  # 傳入父視窗
            "選擇要分析的圖片",  
            "",  
            "Images (*.png *.jpg *.jpeg *.bmp)"  
        )  
        
        if not file_path: return  

        self.ui.ROF.setText("手動模式：正在載入模型並分析圖片...")  
        self.ui.ManualButton.setEnabled(False)  
        
        self.manual_thread = ManualDetectThread(file_path)  # 建立手動執行緒實例
        self.manual_thread.finished.connect(self.show_manual_result)  # 綁定完成訊號
        self.manual_thread.error.connect(self.show_manual_error)  # 綁定錯誤訊號
        self.manual_thread.start()  # 啟動執行緒

    def show_manual_result(self, image_path, label, score):  # 顯示手動結果
        self.ui.ManualButton.setEnabled(True)  
        self.ui.ROF.setText(f"手動判斷完成: {label}")  
        
        dialog = ManualResultDialog(self.ui, image_path, label, score)  # 建立檢測結果視窗
        dialog.exec()  

    def show_manual_error(self, error_msg):  # 顯示手動模式錯誤訊息
        self.ui.ManualButton.setEnabled(True)  
        self.ui.ROF.setText(f"手動判斷失敗")  
        QMessageBox.critical(self.ui, "手動模式錯誤", f"手動分析發生異常：\n{error_msg}")  

    def change_frequency(self):  # 變更偵測頻率
        val = self.ui.freqSlider.value()  # 取得滑桿數值
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
            self.detect_thread.save_cooldown = self.current_cooldown  # 即時更新執行緒內的滑桿數值

    def start_detection(self):  # 處理自動偵測的方法
        if self.detect_thread is not None and self.detect_thread.isRunning(): return  
        
        self.ui.StartButton.setEnabled(False)  # 鎖定開始按鈕
        self.ui.StopButton.setEnabled(True)  # 啟用停止按鈕
        self.ui.ManualButton.setEnabled(False)  # 鎖定手動按鈕
        self.ui.ROF.setText("準備啟動 YOLO + FakeFormer...")  

        self.detect_thread = YoloFakeFormerThread()  
        self.detect_thread.save_cooldown = self.current_cooldown  
        
        self.detect_thread.status_changed.connect(self.update_status)  # 綁定狀態更新
        self.detect_thread.fake_detected.connect(self.show_fake_warning)  # 綁定偽造警報
        self.detect_thread.image_classified.connect(self.handle_classified_image)  # 綁定分類結果
        self.detect_thread.finished.connect(self.detection_finished)  # 綁定執行結束
        self.detect_thread.mode_changed.connect(self.update_mode_label)  # 綁定模式動態切換
        self.detect_thread.error_occurred.connect(self.show_auto_error)  # 綁定錯誤處理
        
        self.detect_thread.start()  # 啟動自動偵測執行緒

    def stop_detection(self):  # 停止自動偵測的方法
        if self.detect_thread is not None and self.detect_thread.isRunning():  
            self.detect_thread.stop()  # 呼叫停止
            self.ui.ROF.setText("正在關閉雷達...")  
            self.ui.StopButton.setEnabled(False)  

    def update_status(self, text):  # 更新狀態列
        self.ui.ROF.setText(text)  

    def update_mode_label(self, mode_text):  # 更新頻率模式 UI 的方法
        self.ui.freqLabel.setText(mode_text)  
        if "極速" in mode_text or "影片" in mode_text:  
            self.ui.freqSlider.setValue(3)  
        elif "中速" in mode_text or "社群" in mode_text:  
            self.ui.freqSlider.setValue(2)  
        elif "慢速" in mode_text or "圖片" in mode_text:  
            self.ui.freqSlider.setValue(1)  

    def handle_classified_image(self, image_path, label, score):  # 處理無偽造時的事件
        self.ui.ROF.setText("雷達運作中... 掃描螢幕人臉")  

    def show_fake_warning(self, image_path, score):  # 觸發警告視窗的方法
        dialog = FakeWarningDialog(self.ui, image_path, score)  # 建立警示視窗
        dialog.exec()  # 顯示警示視窗
        if self.detect_thread is not None:  
            self.detect_thread.continue_after_alert()  

    def show_auto_error(self, error_msg):  # 顯示自動模式的錯誤
        self.ui.ROF.setText("系統異常中斷")  
        QMessageBox.critical(self.ui, "自動模式核心錯誤", f"雷達無法正常工作，具體原因如下：\n\n{error_msg}")  # 彈出嚴重錯誤

    def detection_finished(self):  # 執行緒結束後的清理方法
        self.ui.StartButton.setEnabled(True)  # 恢復開始按鈕
        self.ui.StopButton.setEnabled(False)  # 鎖定停止按鈕
        self.ui.ManualButton.setEnabled(True)  # 恢復手動按鈕
        if self.ui.ROF.text() != "系統異常中斷":  
            self.ui.ROF.setText("雷達已關閉")  
        self.detect_thread = None  


# --- 8. 程式進入點 --- 
if __name__ == "__main__":  
    app = QApplication([])  # 初始化 PySide6
    hb_window = HBMainWindow()  # 初始化主視窗控制器
    hb_window.ui.show()  # 顯示控制器
    app.exec()  # 進入 Qt 的主事件迴圈，保持視窗常駐
    
