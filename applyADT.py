import cv2
import numpy as np
import rasterio
import cbor2
import os
import argparse
import matplotlib.pyplot as plt

def _apply_cbor_lut(data_dn, cbor_path):
    """Apply CBOR IR105 LUT to a DN array, returning temperature in Celsius."""
    if not os.path.exists(cbor_path):
        raise FileNotFoundError(f"Không tìm thấy CBOR file: {cbor_path}")

    with open(cbor_path, 'rb') as f:
        cbor_data = cbor2.load(f)

    lut_list = cbor_data['calibration']['IR105']

    lut_array = np.zeros(256)
    for dn_val, temp_k in lut_list:
        lut_array[int(dn_val)] = temp_k

    temp_k_data = lut_array[data_dn]
    temp_c_data = temp_k_data - 273.15
    return temp_c_data


def load_temperature_data_from_j2k(j2k_path, cbor_path):
    """
    Đọc dữ liệu Digital Number (DN) từ file J2K và map với Look-Up Table (LUT)
    từ file CBOR để tính ra nhiệt độ bề mặt / đỉnh mây (Celsius).
    """
    if not os.path.exists(j2k_path):
        raise FileNotFoundError(f"Không tìm thấy J2K file: {j2k_path}")

    with rasterio.open(j2k_path) as src:
        data_dn = src.read(1)

    return _apply_cbor_lut(data_dn, cbor_path)


def load_temperature_data_from_png(png_path, cbor_path):
    """
    Đọc dữ liệu Digital Number (DN) từ file PNG grayscale và map với LUT
    từ file CBOR để tính ra nhiệt độ bề mặt / đỉnh mây (Celsius).
    """
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"Không tìm thấy PNG file: {png_path}")

    data_dn = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)
    if data_dn is None:
        raise ValueError(f"Không đọc được PNG file: {png_path}")

    return _apply_cbor_lut(data_dn, cbor_path)

def auto_detect_storm_and_apply_adt(temp_c_data, box_size=240, output_img="cropped_storm_temp.png", plot=False):
    """
    Tự động tìm bão, cắt ảnh và tính nhãn ADT dựa trên dữ liệu Nhiệt độ thực (°C).
    """
    h, w = temp_c_data.shape
    
    # 1. Tìm vùng mây bão
    # Mây đối lưu sâu thường rất lạnh (dưới -50°C)
    cloud_mask = (temp_c_data < -50.0).astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "Không tìm thấy khối mây lạnh nào (T < -50°C) nghi ngờ là bão!"}
    
    # Lọc bỏ các khối mây quá nhỏ (nhiễu), ví dụ diện tích < 500 pixels
    valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
    if not valid_contours:
        return {"error": "Các khối mây tìm thấy quá nhỏ, không đạt diện tích hình thành bão!"}
        
    all_results = []
    
    # Lặp qua từng khối mây tìm được
    for idx, contour in enumerate(valid_contours):
        # Xác định tâm khối mây bằng Moments
        M = cv2.moments(contour)
        if M["m00"] == 0: 
            continue
            
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        print(f"\n--- BÃO ĐỐI TƯỢNG {idx+1} ---")
        print(f"-> Tự động xác định tâm bão tại (X={center_x}, Y={center_y})")
        
        # 2. Cắt vùng ảnh bão (Crop ROI - Region of Interest)
        half_box = box_size // 2
        y1, y2 = max(0, center_y - half_box), min(h, center_y + half_box)
        x1, x2 = max(0, center_x - half_box), min(w, center_x + half_box)
        
        roi_temp = temp_c_data[y1:y2, x1:x2]
        
        # Bỏ qua nếu bị cắt sát viền ảnh làm khung quá nhỏ (nhỏ hơn 30x30 pixels)
        roi_h, roi_w = roi_temp.shape
        if roi_h < 30 or roi_w < 30:
            print("-> [Bỏ qua] Tâm bão nằm sát rìa ảnh, không đủ diện tích box.")
            continue
        
        # 3. Tính toán ADT Đơn giản hóa
        center_region = roi_temp[roi_h//2-15 : roi_h//2+15, roi_w//2-15 : roi_w//2+15]
        
        eye_temp = np.max(center_region) 
        eyewall_temp = np.min(roi_temp)
        delta_t = eye_temp - eyewall_temp
        
        t_number = 1.0 + (delta_t / 80.0) * 7.0
        t_number = round(float(np.clip(t_number, 1.0, 8.0)), 1)
        
        wind, pressure = map_t_number_to_intensity(t_number)
        
        if plot:
            # Đặt tên ảnh output riêng cho từng cơn bão
            base_name, ext = os.path.splitext(output_img)
            current_out_img = f"{base_name}_{idx+1}{ext}"

            plt.figure(figsize=(6, 6))
            im = plt.imshow(roi_temp, cmap='jet', vmin=-80, vmax=20)
            plt.colorbar(im, label='Nhiệt độ đỉnh mây (°C)')
            plt.title(f"Bão {idx+1} ({center_x}, {center_y}) - T-Num: {t_number}")
            plt.scatter([roi_w//2], [roi_h//2], color='red', marker='x', s=100, label='Center')
            plt.legend()
            plt.savefig(current_out_img, bbox_inches='tight')
            plt.close()

        print(f"-> Phân tích: Mắt bão={eye_temp:.1f}°C | Thành bão={eyewall_temp:.1f}°C | ΔT={delta_t:.1f}°C")
        
        all_results.append({
            "Storm_ID": idx + 1,
            "Detected_Center": (center_x, center_y),
            "Eye_Temp_C": round(float(eye_temp), 2),
            "Eyewall_Temp_C": round(float(eyewall_temp), 2),
            "Delta_T": round(float(delta_t), 2),
            "T-number": t_number,
            "Wind_Speed_knots": wind,
            "Pressure_hPa": pressure,
            "Image_Crop": f"{os.path.splitext(output_img)[0]}_{idx+1}{os.path.splitext(output_img)[1]}" if plot else None
        })
    
    return all_results

def map_t_number_to_intensity(t_num):
    """Bảng tra cứu Dvorak chuẩn: T-number -> Gió (Knots) & Áp suất (hPa)"""
    if t_num < 1.5: return 25, 1009
    elif t_num < 2.5: return 30, 1000
    elif t_num < 3.5: return 45, 991
    elif t_num < 4.5: return 65, 976
    elif t_num < 5.5: return 90, 954
    elif t_num < 6.5: return 115, 927
    elif t_num < 7.5: return 140, 898
    else: return 170, 858

# --- CHẠY TỪ LỆNH COMMAND LINE ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Áp dụng ADT lên ảnh vệ tinh GK-2A (J2K + CBOR)")
    parser.add_argument("--j2k", type=str, required=True, help="Đường dẫn tới file ảnh mây đuôi .j2k")
    parser.add_argument("--cbor", type=str, required=True, help="Đường dẫn tới file calibration đuôi .cbor")
    parser.add_argument("--box", type=int, default=240, help="Kích thước bounding box muốn cắt (mặc định 240)")
    parser.add_argument("--out_img", type=str, default="cropped_storm_temp.png", help="Tên file ảnh xuất ra")
    parser.add_argument("--out_json", type=str, default="storm_labels.json", help="Tên file JSON lưu nhãn để training")
    
    args = parser.parse_args()
    
    try:
        os.makedirs(os.path.basename(args.j2k), exist_ok=True)
        args.out_img = os.path.join(os.path.basename(args.j2k), args.out_img)
        args.out_json = os.path.join(os.path.basename(args.j2k), args.out_json)

        # Bước 1: Ánh xạ ảnh từ DN sang Nhiệt độ (°C)
        print("Đang đọc và ánh xạ nhiệt độ từ J2K & CBOR...")
        temp_c = load_temperature_data_from_j2k(args.j2k, args.cbor)
        
        # Bước 2: Tự động phân tích ADT trên mảng nhiệt độ cấu trúc vật lý
        print("\nĐang thực thi ADT Analysis...")
        results = auto_detect_storm_and_apply_adt(temp_c, box_size=args.box, output_img=args.out_img, plot=True)
        
        # Bước 3: Lưu kết quả ra file JSON định dạng gọn gàng
        import json
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        print("\n=== KẾT QUẢ ADT ĐÃ LƯU ===")
        print(f"-> Dữ liệu nhãn đã được lưu vào: {args.out_json}")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"LỖI: {e}")