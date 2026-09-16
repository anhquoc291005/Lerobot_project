import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Script hỗ trợ xem và kiểm tra nhanh dữ liệu file CSV.")
    parser.add_argument("csv_path", type=str, help="Đường dẫn tới file CSV cần xem")
    parser.add_argument("--rows", type=int, default=10, help="Số lượng dòng muốn hiển thị (mặc định: 10)")
    args = parser.parse_args()

    try:
        print(f"\nĐang tải dữ liệu từ: {args.csv_path}...\n")
        df = pd.read_csv(args.csv_path)
        
        print("=== THÔNG TIN CHUNG ===")
        df.info()
        
        print(f"\n=== XEM TRƯỚC DỮ LIỆU ({args.rows} dòng đầu) ===")
        try:
            # Sử dụng tabulate để kẻ bảng lưới (grid) cực kỳ trực quan
            import tabulate
            print(df.head(args.rows).to_markdown(tablefmt="grid"))
        except ImportError:
            pd.set_option('display.max_columns', None) # Hiển thị tất cả các cột
            pd.set_option('display.width', 1000)       # Tránh bị xuống dòng quá sớm
            print(df.head(args.rows).to_string())
        
        print("\n=== THỐNG KÊ CƠ BẢN ===")
        try:
            print(df.describe().to_markdown(tablefmt="grid"))
        except ImportError:
            print(df.describe().to_string())
        
    except Exception as e:
        print(f"Đã xảy ra lỗi khi đọc file CSV: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()