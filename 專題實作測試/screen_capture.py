import dxcam
import cv2
import time
import os
from ultralytics import YOLOv10

def main():
    # 1. 初始化路徑與模型
    save_path = "face"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"已建立資料夾: {save_path}")

    model = YOLOv10(r'C:\Users\littl\Desktop\專題實作測試\train10\weights\best.pt')
    camera = dxcam.create(output_color="BGR")
    camera.start(target_fps=30)
    
    face_count = 0  # 用於命名檔案
    print("人臉追蹤與儲存服務啟動... (按 'q' 鍵退出)")
    
    try:
        while True:
            frame = camera.get_latest_frame()
            if frame is None:
                continue
                
            # 執行偵測
            results = model.predict(frame, conf=0.6, verbose=False, device=0)
            
            for r in results:
                for box in r.boxes:
                    # 取得座標
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # --- 核心：人臉裁切 (ROI) ---
                    # 這裡使用 Numpy 的切片功能 [y_start:y_end, x_start:x_end]
                    face_roi = frame[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        face_count += 1
                        # 儲存圖片 (檔名包含編號與時間戳記)
                        file_name = f"face_{face_count}_{int(time.time())}.jpg"
                        cv2.imwrite(os.path.join(save_path, file_name), face_roi)
                    
                    # 在主畫面上畫框（提示用）
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 顯示即時畫面
            cv2.imshow("Real-time Face Cropping", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print(f"服務關閉。總共存儲了 {face_count} 張人臉。")

if __name__ == "__main__":
    main()
