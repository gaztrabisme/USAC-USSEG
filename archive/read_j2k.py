import rasterio
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cbor2
import os

def read_j2k_real_data(j2k_path, cbor_path=None):
    """
    Đọc file J2K và áp dụng bảng biến đổi (Look-Up Table) từ file CBOR để lấy nhiệt độ.
    """
    # 1. Đọc dữ liệu ảnh gốc (DN: 0-255)
    with rasterio.open(j2k_path) as src:
        data_dn = src.read(1)
        
    print(f"--- DỮ LIỆU THÔ (Digital Number) ---")
    print(f"Kích thước: {data_dn.shape}, Giá trị: [{np.min(data_dn)}, {np.max(data_dn)}]")

    # 2. Đọc file CBOR để lấy bảng nhiệt độ
    if cbor_path and os.path.exists(cbor_path):
        with open(cbor_path, 'rb') as f:
            cbor_data = cbor2.load(f)
            
        # Trích xuất bảng tra cứu (Look-Up Table - LUT) cho IR105
        # cbordata['calibration']['IR105'] là danh sách các cặp [DN, Nhiệt độ K]
        lut_list = cbor_data['calibration']['IR105']
        
        # Tạo mảng numpy để map nhanh: chỉ số mảng chính là giá trị DN (0-255)
        lut_array = np.zeros(256)
        for dn_val, temp_k in lut_list:
            lut_array[int(dn_val)] = temp_k
            
        # Áp dụng map: Thay thế toàn bộ DN bằng Nhiệt độ (Kelvin)
        temperature_k = lut_array[data_dn]
        
        # Chuyển Kelvin sang Celsius
        temperature_c = temperature_k - 273.15
        
        print(f"\n--- DỮ LIỆU SAU KHI CALIBRATE (Độ C) ---")
        print(f"Nhiệt độ thấp nhất (Mây bão): {np.min(temperature_c):.2f} °C")
        print(f"Nhiệt độ cao nhất (Mắt bão/Biển): {np.max(temperature_c):.2f} °C")

        # Trực quan hóa
        plt.figure(figsize=(10, 8))
        im = plt.imshow(temperature_c, cmap='jet')
        plt.colorbar(im, label='Nhiệt độ (°C)')
        plt.title("Nhiệt độ đỉnh mây vệ tinh GK-2A")
        plt.show()
        
        return temperature_c
    else:
        print("\n[!] Không tìm thấy file CBOR. Chỉ hiển thị dữ liệu cường độ sáng (0-255).")
        plt.figure(figsize=(10, 8))
        plt.imshow(data_dn, cmap='gray')
        plt.colorbar(label='Digital Number (0-255)')
        plt.title("Dữ liệu chưa qua xử lý nhiệt độ")
        plt.show()
        
        return data_dn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đọc file J2K và lấy nhiệt độ từ file CBOR")
    parser.add_argument("file_path", help="Đường dẫn tới file .j2k")
    parser.add_argument("--cbor_path", default=None, help="Đường dẫn tới file product.cbor")
    args = parser.parse_args()
    
    result = read_j2k_real_data(args.file_path, args.cbor_path)