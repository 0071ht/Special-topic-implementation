import time
import mss
import mss.tools
import numpy as np
import cv2
# 確保所有函式庫都在這裡被引入

def screen_capture_loop(monitor_index=1):
    """即時連續擷取主螢幕 (Monitor 1) 的迴圈"""
    
    with mss.mss() as sct:
        # 獲取主螢幕的完整座標
        monitor = sct.monitors[monitor_index]
        
        # 限制擷取一個較小的區域 (如 640x480)，以提高速度
        target_monitor = {
            "top": monitor["top"] + 50, 
            "left": monitor["left"] + 50,
            "width": 1600,
            "height": 900,
            "mon": monitor_index,
        }

        print("開始即時螢幕監控，按 'q' 鍵退出...")

        last_time = time.time()
        
        while True:
            # 1. 擷取畫面
            sct_img = sct.grab(target_monitor)
            
            # 2. 轉換格式 (BGRA -> BGR)
            img_array = np.array(sct_img)[:, :, :3]
            
            # --- 這是您未來要整合 AI 偵測的地方 ---
            # 這裡就是您流程圖中的「臉部偵測」步驟的輸入
            # 偵測到的臉部將從 img_array 中被提取出來
            # PASS 
            # ----------------------------------------
            
            # 3. 顯示即時畫面 (可以略過此步驟，節省資源)
            cv2.imshow("Real-time Monitor (640x480)", img_array)
            
            # 4. 偵測效能 (可選)
            fps = 1 / (time.time() - last_time)
            last_time = time.time()
            # print(f"FPS: {fps:.2f}")

            # 5. 退出條件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    screen_capture_loop()
