# Cập nhật phiên bản v42 trên GitHub và Streamlit

Bản này được tích hợp vào chính bộ `DPTP_TayNinh_v42-main.zip` đã gửi. Giữ các tệp sẵn có, bổ sung thư mục `forecast/`, cập nhật `app.py`, `chay_web.py` và `requirements.txt`.

## Cách cập nhật qua trình duyệt

1. Tải và giải nén file ZIP nâng cấp. Mở thư mục bên trong có các tệp `app.py`, `requirements.txt` và thư mục `forecast`.
2. Vào đúng repository GitHub và đúng nhánh đang kết nối với ứng dụng Streamlit của bạn. Mở **thư mục gốc chứa app.py**.
3. Chọn **Add file → Upload files**. Tải các mục sau từ thư mục đã giải nén, giữ nguyên tên và cấu trúc:
   - `app.py`
   - `chay_web.py`
   - `requirements.txt`
   - toàn bộ thư mục `forecast/` cùng các file Python bên trong.
4. Kiểm tra trong GitHub: `app.py` và `requirements.txt` nằm đúng chỗ cũ; bên cạnh có thư mục `forecast`. Tránh tải cả thư mục cha `DPTP_TayNinh_v42-main` thành một thư mục con mới.
5. Ghi nội dung thay đổi và bấm **Commit changes** trên đúng nhánh. Không xóa các file còn lại của repository.
6. Nếu đang dùng Streamlit Community Cloud kết nối nhánh đó, ứng dụng sẽ cập nhật. Vì `requirements.txt` thay đổi, quá trình cài lại thư viện có thể lâu hơn bình thường. Nếu bị lỗi cài đặt, mở log Streamlit và lưu thông báo lỗi.

Lưu ý: Gói ZIP bạn cung cấp có `app.py` gọi `forecast` nhưng không chứa thư mục đó; cần đưa **đầy đủ** `forecast/` lên GitHub. Cả hai điểm khởi động `app.py` và `chay_web.py` đều đã được chuẩn bị; bạn có thể giữ nguyên đường dẫn file chính đang cấu hình ở Streamlit.

Nếu repository có thay đổi mới hơn gói v42 đã gửi, hãy đối chiếu những thay đổi đó trước khi ghi đè ba file trên. Mã trong ZIP không tự cập nhật GitHub hoặc triển khai ứng dụng.

## Cách sử dụng tính năng mới

1. Mở ứng dụng đã cập nhật và vào **Dữ liệu & lịch** để chỉnh ngày nghỉ.
2. Ở **Lịch trực quan**, chọn tháng/năm, đánh dấu ngày Lễ/Tết/cắt điện, ghi chú và bấm **Áp dụng đánh dấu**. Lịch mẫu cần được đơn vị tự kiểm tra.
3. Mở tab **🧮 Tính hệ số k**, tại **Phân tích hệ số k từ điện nhận – điện giao ngày** chọn nhiều file Excel theo ngày. File có sheet `Chiều nhận`, `Chiều giao`, cột `Mã điểm đo`, `Tên điểm đo` và các cột ngày 1–31 như file bạn đã gửi.
4. Xác nhận năm/tháng, chọn phạm vi điểm đo và năm phân tích. Chọn tháng cần dự báo và ngày chốt; chương trình chỉ sử dụng các ngày trước ngày chốt. Chọn nhận trừ giao, chỉ nhận hoặc chỉ giao; xem các cảnh báo và số mẫu.
5. Nếu muốn, chọn các hệ số có dữ liệu để áp dụng riêng cho tháng dự báo. Các số bên thanh cấu hình là hệ số chung mặc định; hệ số đã chọn riêng được báo ở bên dưới và có hiệu lực cho tháng tương ứng. Trở lại **Dữ liệu & lịch** và tải sao lưu JSON sau khi chỉnh sửa lịch/hệ số để dùng lại khi ứng dụng khởi động phiên mới. File điện được tải lại từ máy khi cần.

Điện nhận trừ điện giao là dòng điện ở phạm vi đo, chưa tự động bằng điện thương phẩm của 14 huyện. Các kỳ có thể thay đổi điểm đo; dữ liệu còn thiếu và số lượng năm quan sát ảnh hưởng độ tin cậy. Các giá trị k là thăm dò cần được đối chiếu và kiểm định, chưa phải chứng minh cải thiện dự báo.

## Kiểm tra và giới hạn

Lõi `forecast/models.py` được giữ nguyên từ gói dự báo đã được cung cấp trước đó; các thuật toán MLP, Random Forest và XGBoost không được thay đổi trong bản nâng cấp này. Tính ngày tương đương có thể thay đổi khi người dùng chọn lịch theo ngày hoặc hệ số riêng tháng. Kiểm định tích hợp hiện là kiểm định một bước, chưa tự tái dựng số ước ngày 10 hoặc tính lại k tại từng mốc lịch sử.

Phần phân tích dữ liệu, lịch và sao lưu có thể kiểm tra bằng `python -m unittest discover -s tests -p "test_daily_k.py" -v`. Trong môi trường tạo gói đã kiểm tra các phép tính và luồng giao diện mô phỏng; chưa khởi chạy Streamlit/XGBoost thực và chưa kiểm thử cài thư viện trên máy chủ Streamlit của người dùng.
