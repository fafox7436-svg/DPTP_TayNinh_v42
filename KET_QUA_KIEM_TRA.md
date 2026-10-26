# Kết quả kiểm tra bộ mã khôi phục

Ngày hoàn tất: 26/10/2026 (Việt Nam). Phiên bản: 2.1.0-loi-goc.

## Kết quả

- Bộ kiểm tra: 17 đạt, 0 thất bại, 0 lỗi.
- Hai hàm lõi khớp chuỗi token với đoạn trích mã gốc do người dùng cung cấp: `tao_dac_trung` 293 token; `chay_mo_hinh_goc` 534 token. Bỏ qua định dạng, chú thích và phần cache ở giao diện.
- Hai ca đối chiếu độc lập giữa toàn bộ luồng nhập/dự báo với gọi trực tiếp lõi gốc: bốn chuỗi NN, RF, XGBoost, xu hướng đều bằng nhau tuyệt đối trong cùng môi trường; độ lệch tuyệt đối lớn nhất bằng 0 trên các giá trị đã kiểm tra.
- Ca 1: seed 42, hệ số mặc định, kWh. Ca 2: seed 17, hệ số khác mặc định, triệu kWh, thiếu độ ẩm ở bảng dự báo. Cả hai giữ thứ tự đầu vào chưa sắp xếp, chỉ số dòng tùy ý, tháng lặp và tháng trong tập lịch sử.
- Công thức chốt giữ kết quả gốc; kiểm tra phân bổ 7 nhóm ngày, giữ phần nguyên số ngày cắt điện, cột Số ngày khác lịch, lưu/mở JSON và xuất Excel.
- Streamlit AppTest chạy từ `chay_web.py`: nạp dữ liệu minh họa, chạy 6 kỳ kiểm chứng và 6 tháng dự báo, mặc định XGBoost, tạo nút xuất hồ sơ, chặn kết quả cũ sau thay đổi hệ số, khôi phục kết quả khi trả cấu hình cũ và chạy một seed trong Find Seed.
- Kiểm tra AI dùng phản hồi giả lập; không gọi OpenAI/Gemini thật.
- Các tệp Python biên dịch cú pháp thành công.

## Môi trường kiểm tra

Python 3.12.14; Linux x86_64.

| Thư viện | Phiên bản |
|---|---|
| NumPy | 2.3.5 |
| pandas | 2.2.3 |
| SciPy | 1.17.0 |
| scikit-learn | 1.8.0 |
| XGBoost | 3.4.1 |
| joblib | 1.5.3 |
| threadpoolctl | 3.6.0 |
| Streamlit | 1.65.0 |
| Plotly | 6.9.0 |
| openpyxl | 3.1.5 |
| xlrd | 2.0.1 |
| Pydantic | 2.13.5 |
| pytest | 8.3.5 |

## Giới hạn

Các ca kiểm tra sử dụng số liệu tổng hợp. Chưa có tệp sản lượng thực tế, kết quả đã lưu và phiên bản môi trường chạy bản gốc của người dùng để xác nhận khớp với lần chạy thực tế tại đơn vị. Chưa kiểm tra trực tiếp trên Windows hoặc máy chủ triển khai của người dùng.

Kết quả trên chứng minh các ca khôi phục đã kiểm tra khớp mã nguồn gốc trong cùng môi trường. Không chứng minh độ chính xác dự báo thực tế, không khẳng định mô hình tốt hơn bản gốc và không bảo đảm trùng từng bit khi thay thư viện, phần cứng hoặc đầu vào.

## Dấu vết mã nguồn

SHA-256 của văn bản gốc được dùng đối chiếu:

`39b0f1332e9a5c95f0b6e9909fb4a59aa2971ca0bb68fd27233def63b1dc9bd5`

SHA-256 tệp `loi_du_bao_goc.py` trong gói phát hành:

`659e04a9ec07ca365e249040572508411617640fd96c51e510e4834f80ef2201`

Đoạn trích hai hàm tham chiếu có trong `tests/reference_core.json`; mã kiểm tra nằm trong `tests/`.

## Điều chỉnh gói triển khai GitHub / Streamlit Community Cloud

Ngày 26/10/2026: đưa cấu trúc ZIP về thư mục gốc; bỏ các tệp khởi động Windows và tiện ích cài cục bộ khỏi gói Cloud; thêm hai SDK AI vào requirements.txt; thay cấu hình bind 127.0.0.1 bằng chế độ headless; viết lại hướng dẫn triển khai GitHub. Mã Python của giao diện, xử lý dữ liệu, dự báo, chốt phương án và AI không thay đổi. Các phiên bản NumPy, pandas, SciPy, scikit-learn, XGBoost, joblib và threadpoolctl giữ nguyên.

Chưa cập nhật repository hoặc triển khai vào tài khoản Streamlit của người dùng.

Kiểm tra gói Cloud sau chỉnh sửa: cấu hình TOML hợp lệ; không cố định địa chỉ/cổng máy chủ; các tệp Python giữ nguyên SHA-256; các phiên bản thư viện tính toán không đổi. Kiểm tra giao diện từ cấu trúc thư mục gốc đã đạt (1 kiểm tra AppTest, gồm chạy dự báo, xuất hồ sơ, trạng thái kết quả cũ và Find Seed). Đây là kiểm tra trong môi trường Linux, chưa phải xác nhận triển khai thành công trên tài khoản Streamlit của người dùng.
