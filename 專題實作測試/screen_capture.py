import dxcam
import cv2
import time
import os
import uuid
import numpy as np
from ultralytics import YOLO

def main():
    # 1. 初始化路徑與模型
    save_path = "face"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"已建立資料夾: {save_path}")

    # 載入 YOLO 模型
    model = YOLO(r'C:\Users\littl\Desktop\專題實作測試\train10\weights\best.pt')
    
    # 初始化螢幕擷取
    camera = dxcam.create(output_color="BGR")
    camera.start(target_fps=30)
    
    # ---------------------------------------------------------
    # 2. 建立「遙控器」視窗與按鈕邏輯
    # ---------------------------------------------------------
    exit_flag = False  # 控制程式是否該關閉的全域變數
    
    # 滑鼠點擊事件的處理函式
    def remote_click(event, x, y, flags, param):
        nonlocal exit_flag
        # 當滑鼠左鍵點下時
        if event == cv2.EVENT_LBUTTONDOWN:
            # 判斷點擊位置是否在我們的「按鈕」範圍內 (x: 50~250, y: 20~80)
            if 50 <= x <= 250 and 20 <= y <= 80:
                print("收到遙控器關閉指令！")
                exit_flag = True

    # 建立遙控器視窗並綁定滑鼠事件
    remote_window_name = "Remote Control"
    cv2.namedWindow(remote_window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(remote_window_name, remote_click)
    
    # 繪製遙控器的靜態畫面 (300x100 的深灰色背景)
    remote_img = np.ones((100, 300, 3), dtype=np.uint8) * 50 
    # 畫一個紅色的按鈕
    cv2.rectangle(remote_img, (50, 20), (250, 80), (0, 0, 200), -1)
    # 寫上白色的 EXIT 文字
    cv2.putText(remote_img, "EXIT PROGRAM", (75, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 先顯示一次遙控器畫面
    cv2.imshow(remote_window_name, remote_img)

    # ---------------------------------------------------------
    # 3. 建立「主螢幕監控」視窗
    # ---------------------------------------------------------
    monitor_window_name = "Monitor View"
    cv2.namedWindow(monitor_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(monitor_window_name, 800, 600)

    face_count = 0  
    save_cooldown = 0.5  
    last_save_time = 0
    
    print("服務啟動... (請點擊遙控器上的 EXIT 按鈕來關閉)")
    
    try:
        while not exit_flag:  # 當 exit_flag 變成 True 時跳出迴圈
            frame = camera.get_latest_frame()
            if frame is None:
                # 即使沒擷取到畫面，也要維持 OpenCV UI 的更新
                cv2.waitKey(1) 
                continue
                
            # 執行偵測
            results = model.predict(frame, conf=0.6, verbose=False, device=0)
            current_time = time.time()
            can_save = (current_time - last_save_time) > save_cooldown
            
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        if can_save:
                            face_count += 1
                            unique_id = uuid.uuid4().hex[:6]
                            file_name = f"face_{face_count}_{int(current_time)}_{unique_id}.jpg"
                            cv2.imwrite(os.path.join(save_path, file_name), face_roi)
                            last_save_time = current_time
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 顯示監控畫面
            cv2.imshow(monitor_window_name, frame)
            
            # 每回合都要呼叫 waitKey 讓 OpenCV 處理滑鼠點擊與視窗更新
            # 如果使用者習慣按 q，一樣保留按 q 離開的功能
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(f"服務已安全關閉。總共存儲了 {face_count} 張人臉。")

if __name__ == "__main__":
    main()
    
