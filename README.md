# Dự báo điện thương phẩm — giao diện nâng cấp, lõi mô hình gốc

Phiên bản `2.1.0-loi-goc`. Bộ mã đầy đủ phát triển từ mã nguồn Lê Minh Trí đã cung cấp và giao diện v2. Mục tiêu của bản này là khôi phục phép tính gốc, không thay đổi mô hình để tạo kết quả khác.

## 1. Triển khai từ GitHub lên Streamlit Community Cloud

Bộ mã này dành cho Streamlit Community Cloud lấy mã nguồn từ GitHub. Không cần cài Python, chạy lệnh pip hoặc dùng tệp `.bat` trên máy cá nhân.

### Cập nhật ứng dụng đang dùng

1. Giải nén ZIP. Các tệp `chay_web.py`, `app.py`, `requirements.txt` và các mô-đun hỗ trợ nằm trực tiếp ở cấp đầu tiên của gói, không bị lồng trong thư mục cha.
2. Đưa toàn bộ các tệp đã giải nén vào thư mục gốc của repository GitHub, đúng nhánh mà ứng dụng Streamlit đang sử dụng. Cập nhật đồng bộ các tệp cùng tên; không chỉ thay riêng `chay_web.py` và không tải mỗi tệp ZIP lên repo.
3. Giữ cấu trúc `.streamlit/config.toml` tại gốc repository. Nếu dùng giao diện tải tệp của GitHub mà thư mục này chưa được đưa lên, tạo đúng đường dẫn đó và chép nội dung cấu hình kèm theo.
4. Commit thay đổi. Streamlit sẽ nhận cập nhật từ GitHub; khi `requirements.txt` thay đổi, dịch vụ tự cài lại phụ thuộc.
5. Mở ứng dụng, kiểm tra log tại Manage app. Nếu chưa nhận bản mới, dùng Reboot app. Sau khi mở được giao diện, thử dữ liệu minh họa rồi nạp đúng dữ liệu thực tế của đơn vị.

Tệp khởi động phải là `chay_web.py`. Nếu app đang trỏ tới đường dẫn này thì giữ nguyên. Tệp này nhập các mô-đun cùng thư mục, vì vậy cần tải đủ bộ mã.

### Nếu tạo một ứng dụng mới

Tại [Streamlit Community Cloud](https://share.streamlit.io), chọn Create app, dùng repository GitHub của anh, rồi khai báo:

| Trường | Giá trị |
|---|---|
| Repository | Repository chứa bộ mã đã giải nén |
| Branch | Đúng nhánh đã commit, ví dụ `main` nếu repo dùng nhánh này |
| Main file path | `chay_web.py` |
| Advanced settings → Python version | `3.12`, khớp phiên bản chính/phụ đã kiểm tra gói này |
| Secrets | Chỉ cần nếu sử dụng trợ lý OpenAI hoặc Gemini |

Sau đó chọn Deploy. `requirements.txt` đã gồm thư viện dự báo, Excel, biểu đồ và hai SDK AI để dịch vụ tự cài. Không cần `packages.txt` cho các phụ thuộc đang khai báo. Không cố định địa chỉ `127.0.0.1` hoặc cổng máy chủ trong cấu hình kèm theo; dịch vụ triển khai quản lý phần đó.

Với ứng dụng đã triển khai, không mặc định rằng có thể đổi Python bằng cách sửa một tệp trong repo. Theo tài liệu Streamlit hiện tại, thay phiên bản Python cần triển khai lại ứng dụng. Gói này chưa thực hiện thay đổi trên tài khoản GitHub/Streamlit của người dùng.

### Giữ kết quả của lần chạy cũ

Lõi NN, RF và XGBoost không đổi so với gói khôi phục đã gửi. Các phiên bản tính toán trong `requirements.txt` cũng được giữ nguyên; lần chỉnh này chỉ thêm SDK AI vào danh sách cài tự động và sửa phần triển khai.

Tuy nhiên, các phiên bản thư viện trong gói là môi trường đã kiểm tra, chưa phải bản sao đã xác minh của ứng dụng gốc trên Streamlit của anh. Để xác nhận trùng kết quả của lần chạy cũ, cần đối chiếu `requirements.txt`/lockfile, log cài đặt, phiên bản Python và cùng dữ liệu, lịch, hệ số, seed. Không tự coi việc khớp mã nguồn là đủ để bảo đảm trùng từng số khi môi trường khác.

### Nếu có lỗi khi triển khai

- `No module named ...`: kiểm tra đã commit `requirements.txt` cùng bộ mã, đúng nhánh và đường dẫn app. Tránh để một tệp quản lý phụ thuộc cũ như `uv.lock` hoặc `Pipfile` khiến Cloud chọn môi trường khác; xem log trước khi thay tệp quản lý phụ thuộc.
- Log vẫn hiện `import matplotlib.pyplot as plt` ở dòng 4 của `chay_web.py`: kiểm tra app còn đang chạy tệp/nhánh cũ hay không. `chay_web.py` trong gói này không nhập Matplotlib; biểu đồ dùng Plotly.
- Không thấy `app.py`, `forecasting.py` hoặc `loi_du_bao_goc.py`: tải đủ các mô-đun cùng cấp với `chay_web.py`.
- Nếu lỗi phụ thuộc còn xảy ra, xem đầy đủ thông báo cài đặt trong Manage app, không chỉ dòng `import` cuối cùng.

Nguồn hướng dẫn triển khai: [Deploy your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [Manage your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app), [Upgrade Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python). Đối chiếu ngày 26/10/2026.

## 2. Những phép tính đã giữ từ bản gốc

Hai hàm `tao_dac_trung` và `chay_mo_hinh_goc` nằm trong `loi_du_bao_goc.py`. Giao diện gọi trực tiếp các hàm này qua `forecasting.py`.

| Thành phần | Cấu hình gốc được khôi phục |
|---|---|
| NN | Ba lớp ẩn `(10, 15, 10)`; `relu`; `lbfgs`; `alpha=0.1`; `max_iter=5000`; `random_state=seed` |
| Chuẩn hóa NN | `StandardScaler` cho đầu vào; không thêm bước chuẩn hóa biến đích |
| Random Forest | `n_estimators=200`; `random_state=seed`; học phần dư sau xu hướng tuyến tính |
| XGBoost | `n_estimators=100`; `learning_rate=0.05`; `max_depth=4`; `reg_alpha=0.1`; `random_state=seed`; học phần dư sau xu hướng |
| Đặc trưng, đúng thứ tự | Tháng, Năm, Nhiệt độ TB, Độ ẩm, Mua_Nong, Mua_Mua, Bien_Ngoai_Sinh; chỉ dùng cột có ở cả hai bảng |
| Mùa | Mùa nóng: tháng 3–5; mùa mưa: tháng 6–11; Bien_Ngoai_Sinh bằng 0 |
| Biến đích | `log1p(Tổng thương phẩm / Ngày tương đương)` |
| Xu hướng | Hồi quy tuyến tính theo `(Năm - năm đầu tập học) * 12 + Tháng` |
| Quy đổi dự báo | NN dùng `expm1(dự báo NN)`; RF/XGB dùng `expm1(xu hướng + phần dư)`; nhân số ngày tương đương của tháng cần dự báo |
| Hệ số mặc định | Thứ Hai 0,95; thứ Bảy 0,94; Chủ nhật 0,79; lễ 0,56; Tết 0,40; cắt điện 0,50; ngày thường 1 |
| Lịch nghỉ | Chép nguyên 23 dòng mặc định năm 2023–2027 vào `lich_nghi_goc.py`; cột Năm, Tháng, Tết Âm, Lễ Nhỏ |
| Chốt phương án | Giữ công thức `Q_mô_hình / D_cũ × D_mới`; giữ phân bổ riêng cho 7 nhóm ngày, gồm nhóm cắt điện |
| Mô hình làm cơ sở | Mặc định XGBoost; không tự chuyển sang mô hình có WAPE thấp nhất |

Những tham số không ghi trong mã gốc tiếp tục dùng mặc định của thư viện; không bổ sung siêu tham số để thay đổi quá trình học.

Quy tắc lịch gốc được giữ nguyên: đếm toàn bộ thứ Hai, thứ Bảy, Chủ nhật; trừ tiếp số ngày lễ, Tết và cắt điện khỏi ngày thường; ngày thường âm thì lấy 0. Không tự chuyển sang lịch loại trừ ngày trùng. Số ngày cắt điện được lấy phần nguyên bằng `int()` như bản gốc. Đây là bảo toàn công thức, không phải khẳng định các giả định lịch đã được xác nhận bằng số liệu thực tế.

`Số ngày` đã nhập được giữ nguyên: phần mô hình lấy số ngày lịch; công thức chốt phương án lấy cột `Số ngày`, đúng hành vi gốc. Nếu hai số khác nhau, ứng dụng hiện lưu ý. Tháng lễ/Tết không khai báo có số lượng bằng 0.

Giữ cùng đơn vị số liệu khi đối chiếu. Công thức gốc dùng `log1p`, nên đổi từ kWh sang triệu kWh rồi huấn luyện lại có thể thay đổi kết quả sau khi quy đổi ngược. Chọn đơn vị chỉ gắn nhãn cho dữ liệu người dùng; chương trình không tự đổi số trong Excel.

## 3. Nhập dữ liệu và thao tác

1. Bấm “Tải tệp Excel mẫu”, điền `Lich_su` và `Du_bao`. Có thể nạp cùng một tệp vào hai ô, sau đó chọn đúng sheet.
2. Lịch sử cần `Năm`, `Tháng`, `Tổng thương phẩm`. Dự báo cần `Năm`, `Tháng`. Giữ các cột nhiệt độ, độ ẩm đã dùng ở lần chạy gốc. Nếu một đặc trưng chỉ xuất hiện ở một bảng, quy tắc gốc sẽ không dùng đặc trưng đó.
3. Cột `Số ngày` có thể bỏ để tự lấy theo lịch. Cột `Số ngày cắt điện` có thể bỏ để mặc định 0. Nếu giữ cột thì điền số hợp lệ.
4. Kiểm tra lịch nghỉ, hệ số và seed đúng với lần chạy cần đối chiếu. Bảng mặc định có thể sửa trực tiếp hoặc nhập từ sheet `Lich_nghi`/CSV bốn cột.
5. Bấm “Kiểm chứng và chạy dự báo”. Xem các dự báo riêng NN, RF, XGBoost và hai mô hình đối chiếu.
6. Mở “Chốt phương án”, chọn mô hình làm cơ sở, nhập hệ số/ngày cắt điện và lý do nếu điều chỉnh. “Điều chỉnh thêm (%)” mặc định 0; chỉ thay đổi khi người dùng chủ động nhập hoặc chọn áp dụng tham khảo AI.
7. Tải hồ sơ Excel và JSON phương án. Thay dữ liệu hoặc cấu hình sẽ yêu cầu chạy lại trước khi xuất kết quả, tránh dùng nhầm kết quả cũ.

Ứng dụng giữ thứ tự dòng, không tự sắp xếp lịch sử, nội suy tháng thiếu hay loại tháng lặp. Các giá trị không hợp lệ được báo để người dùng xử lý. Dữ liệu minh họa là số liệu tổng hợp giả lập để thử thao tác, không phải dữ liệu của Công ty Điện lực Tây Ninh.

## 4. Các tiện ích của giao diện

- Kiểm chứng theo thời gian với MAE, RMSE, MAPE và WAPE. Mỗi kỳ học ít nhất 24 tháng trước mốc dự báo. Kết quả kiểm chứng không thay thế kết quả mô hình đã học trên toàn bộ lịch sử. Bảng có tháng lặp vẫn chạy dự báo gốc, nhưng bỏ qua kiểm chứng cuốn chiếu.
- Kiểm chứng hiện sử dụng nhiệt độ/độ ẩm đã biết của tháng kiểm chứng nếu có; chưa đo thêm sai số dự báo thời tiết. WAPE thấp hơn chỉ áp dụng cho các kỳ, chân trời và thông tin đầu vào đã kiểm tra.
- Find Seed chạy lại đúng ba mô hình gốc với từng seed, lưu kết quả theo lô và xuất Excel. “Đạt khoảng mục tiêu” là gần số người dùng nhập, không phải bằng chứng chính xác hơn. Không tự áp dụng seed tìm được vào kết quả chính.
- Giải trình xu hướng nền và chênh lệch giữa dự báo với nền. Phép trừ này không xác định tác động nhân quả riêng của thời tiết hoặc ngày nghỉ.
- Excel xuất kèm đầu vào, lịch, hệ số, seed, phiên bản thư viện, dự báo, sai số, kịch bản và phân bổ ngày. JSON chỉ mở lại với dữ liệu/cấu hình khớp.

Các tiện ích nêu trên không tự sửa cấu hình NN, RF hoặc XGBoost.

## 5. Trợ lý AI tùy chọn

NN, RF, XGBoost và Find Seed chạy được mà không cần API key. SDK OpenAI/Gemini đã có trong `requirements.txt`; Streamlit tự cài khi triển khai.

Nếu dùng trợ lý AI, mở Settings → Secrets của ứng dụng Streamlit và điền `OPENAI_API_KEY`/`GEMINI_API_KEY` theo mẫu `.streamlit/secrets.toml.example`. Cũng có thể nhập API key dùng trong phiên ở giao diện. Không commit khóa thật lên GitHub. Mã mô hình AI được chọn trong giao diện hoặc cấu hình bằng `OPENAI_MODEL`/`GEMINI_MODEL`.

Phần AI giữ luồng của v2: người dùng dán nội dung để phân tích; không tự tải bài báo từ URL như phần đọc web trong mã cũ. Kết quả AI chỉ được áp dụng vào phương án khi bấm nút tương ứng. AI không đổi tham số huấn luyện hoặc dự báo gốc. Kiểm tra AI trong gói này dùng phản hồi giả lập, chưa gọi dịch vụ thực với tài khoản của người dùng.

## 6. Đối chiếu và kiểm tra

Chạy bộ kiểm tra:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Các lệnh trên chỉ dành cho lập trình viên kiểm tra mã trong môi trường riêng, không phải thao tác bắt buộc để triển khai trên Streamlit Community Cloud.

Các kiểm tra gồm:

- So sánh chuỗi token của hai hàm lõi với đoạn trích từ mã gốc đã cung cấp, bỏ qua khoảng trắng, thụt lề và chú thích. Không tạo kỳ vọng bằng cách sao lại kết quả của hàm đang kiểm tra.
- So sánh giá trị đầu ra toàn bộ luồng nhập → dự báo với gọi trực tiếp lõi gốc: NN, RF, XGBoost và xu hướng; yêu cầu bằng nhau tuyệt đối trong cùng môi trường thử nghiệm.
- Kiểm tra hai bộ seed/hệ số/đơn vị, dữ liệu không theo thứ tự thời gian, chỉ số dòng tùy ý, tháng lặp, dự báo lại tháng lịch sử, cắt điện có phần thập phân và thiếu một cột thời tiết ở bảng dự báo.
- Kiểm tra công thức chốt gốc, nhập Excel, lưu/mở JSON, xuất Excel, dữ liệu kiểm chứng chỉ lấy từ trước mốc dự báo, thao tác giao diện, mặc định XGBoost, trạng thái kết quả cũ và Find Seed.

Xem `KET_QUA_KIEM_TRA.md` để biết kết quả và giới hạn kiểm tra. Muốn xác nhận trùng với kết quả đã chạy tại đơn vị, vẫn cần cùng tệp dữ liệu, sheet, thứ tự dòng, đơn vị, lịch nghỉ, hệ số, seed và môi trường thư viện. Chưa có dữ liệu sản lượng thực tế và thông tin đầy đủ về môi trường chạy bản gốc của người dùng để xác nhận trường hợp đó.

## 7. Các tệp chính

| Tệp | Vai trò |
|---|---|
| `chay_web.py` | Điểm khởi động Streamlit |
| `app.py` | Giao diện và thao tác người dùng |
| `loi_du_bao_goc.py` | Hai hàm tính toán gốc được giữ nguyên token |
| `forecasting.py` | Nối giao diện với lõi gốc, kiểm chứng và chốt phương án |
| `lich_nghi_goc.py` | Bảng lịch nghỉ mặc định gốc |
| `data_io.py` | Đọc và kiểm tra đầu vào |
| `reporting.py` | Tệp mẫu, Excel và JSON |
| `ai_assistant.py` | Trợ lý AI tùy chọn từ v2 |
| `requirements.txt` | Toàn bộ phụ thuộc Cloud cài tự động |
| `requirements-dev.txt` | Phụ thuộc để chạy bộ kiểm tra riêng |
| `tests/` | Mã kiểm tra và đoạn trích tham chiếu từ nguồn gốc |

Không tự sửa `loi_du_bao_goc.py` để “tối ưu”. Bất kỳ thử nghiệm thay mô hình nào sau này cần làm thành phiên bản riêng, đối chiếu trên cùng dữ liệu ngoài mẫu và được người dùng đồng ý trước khi thay bản đang dùng.
