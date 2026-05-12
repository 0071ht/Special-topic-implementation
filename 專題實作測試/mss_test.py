import mss
import mss.tools
import numpy as np
import cv2

# 設定擷取區域 (只擷取主螢幕)
# 如果您有多個螢幕，需要指定要擷取的螢幕編號，或指定座標
# sct.monitors[0] 是整個虛擬螢幕, sct.monitors[1] 通常是主螢幕
monitor_number = 1 

with mss.mss() as sct:
    # 獲取主螢幕的資訊
    mon = sct.monitors[monitor_number] 

    # 定義擷取區域 (x, y, 寬度, 高度)
    # 這裡擷取螢幕中央 500x500 的區域
    monitor = {
        "top": mon["top"] + 100, 
        "left": mon["left"] + 100,
        "width": 1600,
        "height": 900,
        "mon": monitor_number,
    }

    # 擷取畫面
    sct_img = sct.grab(monitor)
    
    # 將 mss 的結果轉換為 NumPy 陣列
    img_array = np.array(sct_img)
    
    # 由於 mss 擷取的格式是 BGRA，cv2 預設是 BGR，需要轉換
    # 丟棄 Alpha (A) 通道
    img_bgr = img_array[:, :, :3]

    # 顯示擷取的圖像
    cv2.imshow("Test Screenshot", img_bgr)
    cv2.waitKey(0) # 按下任意鍵關閉視窗
    cv2.destroyAllWindows()
