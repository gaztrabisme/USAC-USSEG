import cbor2
import json
import argparse
import sys

# Ép console dùng UTF-8 trên Windows để không bị lỗi ký tự
sys.stdout.reconfigure(encoding='utf-8')

def convert_to_serializable(obj):
    """Hàm đệ quy để convert các kiểu dữ liệu dị (bytes) sang dạng đọc được bởi json"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    else:
        return obj

def main():
    parser = argparse.ArgumentParser(description="Đọc file CBOR và hiển thị thông tin")
    parser.add_argument("cbor_file", help="Đường dẫn tới file .cbor")
    args = parser.parse_args()

    print(f"Đang đọc file: {args.cbor_file}")
    print("-" * 50)
    
    try:
        with open(args.cbor_file, 'rb') as f:
            data = cbor2.load(f)
            
        serializable_data = convert_to_serializable(data)
        
        print("=== THÔNG TIN CHUNG ===")
        print(f"Các khóa (keys) trong dữ liệu: {list(serializable_data.keys())}")
        
        if "bit_depth" in serializable_data:
            print(f"- Bit depth: {serializable_data['bit_depth']} bits")
            
        if "calibration" in serializable_data:
            bands = list(serializable_data['calibration'].keys())
            print(f"- Các dải quang phổ (bands) chứa Calibration: {bands}")
            
            # Hiển thị số điểm dữ liệu của 1 band ví dụ
            sample_band = bands[0]
            num_points = len(serializable_data['calibration'][sample_band])
            print(f"- Bảng tra cứu mẫu cho {sample_band} chứa {num_points} giá trị (DN -> Temperature)")

        #     print("\n=== CHI TIẾT TẤT CẢ CÁC KEYS ===")
        # for key, value in serializable_data.items():
        #     print(f"\n--- KEY: '{key}' ---")
        #     if key == "calibration":
        #         print("(Đã ẩn nội dung chi tiết vì quá dài, xem thông tin ở phần trên)")
        #     elif isinstance(value, (dict, list)):
        #         # Tránh in mảng quá dài làm trôi màn hình
        #         if isinstance(value, list) and len(value) > 20:
        #             summary = value[:5]
        #             summary.append(f"... (có tổng cộng {len(value)} phần tử) ...")
        #             summary.extend(value[-5:])
        #             print(json.dumps(summary, indent=2, ensure_ascii=False))
        #         else:
        #             print(json.dumps(value, indent=2, ensure_ascii=False))
        #     else:
        #         print(value)

        print("\n=== Ý NGHĨA VÀ NỘI DUNG CỦA TỪNG KEY TRONG CBOR ===")
        
        # Từ điển giải thích các keys phổ biến trong GK2A/CBOR
        key_descriptions = {
            "bit_depth": "Độ sâu bit của ảnh (vd: 8, 10, 16). Quyết định phạm vi giá trị của pixel thô (DN).",
            "calibration": "Bảng hệ số chuẩn hóa (Look-up Table - LUT) để quy đổi giá trị pixel thô (DN) sang nhiệt độ hoặc bức xạ.",
            "has_timestamps": "Cờ (Boolean) cho biết dữ liệu có chứa thông tin thời gian (timestamp) quét của từng khối/dòng ảnh hay không.",
            "images": "Danh sách các file ảnh phụ thuộc (hoặc ma trận con), kèm thông tin về kích thước và cách sắp xếp của chúng.",
            "instrument": "Tên thiết bị cảm biến trên vệ tinh đã chụp dữ liệu này (Ví dụ: AMI của vệ tinh GK-2A).",
            "needs_correlation": "Cờ (Boolean) hoặc thông số chỉ ra xem dữ liệu này có cần bù trừ tương quan gì thêm nội bộ không.",
            "product_source": "Đơn vị hoặc hệ thống đã tạo ra/phân phối file này (VD: KMA, NMSC).",
            "product_timestamp": "Mốc thời gian tổng thể của toàn bộ file dữ liệu (thường là thời gian bắt đầu chụp).",
            "projection_cfg": "Cấu hình phép chiếu bản đồ (Georeference). RẤT QUAN TRỌNG để ánh xạ tọa độ pixel (X, Y) sang Kinh độ/Vĩ độ (Lon, Lat).",
            "timestamps": "Mảng lưu trữ chi tiết các mốc thời gian quét điểm/dòng của thiết bị cảm biến vệ tinh.",
            "timestamps_type": "Quy ước và định dạng của các mốc thời gian (VD: mili-giây, định dạng datetime...).",
            "type": "Mô tả định dạng chuẩn / kiểu cấu trúc của file dữ liệu."
        }

        for key, value in serializable_data.items():
            print(f"\n[ Key: {key} ]")
            desc = key_descriptions.get(key, "Thông tin mở rộng chưa được định nghĩa rõ.")
            print(f"  ? Ý nghĩa: {desc}")
            
            # Hiển thị độ lớn đối với các keys chứa cụm dữ liệu quá dài
            if key == "calibration":
                bands = list(value.keys())
                print(f"  --- CHI TIẾT BÊN TRONG 'calibration' ---")
                
                calib_desc = {
                    "IR105": "Bảng tra cứu (LUT) mapping trực tiếp từ trị số Digital Number (0-255) sang Nhiệt độ Kelvin (K) của Band hồng ngoại 10.5 µm.",
                    "bits_for_calib": "Độ phân giải bit được dùng khi thực hiện tạo bảng tính chuẩn hóa.",
                    "calibrator": "Tên thuật toán/class hoặc tiêu chuẩn thực hiện việc quy đổi (ví dụ: biến đổi độ sáng sang nhiệt độ).",
                    "default_range": "Phạm vi hiển thị hữu ích tối thiểu và tối đa mặc định của vệ tinh cho dải đo này.",
                    "type": "Kiểu dữ liệu hay định dạng chuẩn hóa.",
                    "wavenumbers": "Trị số nghịch đảo của bước sóng (Wavenumbers - cm⁻¹) - thông số vật lý đặc trưng tương ứng thiết bị cảm biến dải IR."
                }
                for sub_key, sub_val in value.items():
                    sub_desc = calib_desc.get(sub_key, "Không rõ")
                    print(f"    * {sub_key}: {sub_desc}")
                    
                    if sub_key == "IR105":
                        print(f"      -> Kích thước: {len(sub_val)} điểm dữ liệu.")
                    elif isinstance(sub_val, (dict, list)) and len(sub_val) > 10:
                        print(f"      -> Giá trị: [Mảng/Từ điển lớn chứa {len(sub_val)} phần tử]")
                    else:
                        print(f"      -> Giá trị: {sub_val}")
                        
            elif key == "timestamps":
                print(f"  -> Giá trị: [Mảng dữ liệu dài] gồm {len(value)} mốc thời gian.")
            elif key == "projection_cfg":
                print(f"  -> Giá trị (Cấu hình phép chiếu):")
                print("     " + json.dumps(value, ensure_ascii=False).replace('\n', '\n     '))
            elif key == "images":
                print(f"  -> Giá trị (Thông tin cấu trúc ảnh):")
                print("     " + json.dumps(value, ensure_ascii=False).replace('\n', '\n     '))
            else:
                print(f"  -> Giá trị: {value}")
            
    except ImportError:
        print("Lỗi: Bạn chưa cài đặt thư viện cbor2. Hãy chạy lệnh 'pip install cbor2'")
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")

if __name__ == "__main__":
    main()