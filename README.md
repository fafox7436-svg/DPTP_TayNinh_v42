# Dự báo điện thương phẩm theo tháng — phiên bản 2.0

Bản nâng cấp từ mã Streamlit trong tệp người dùng cung cấp. Giữ ba mô hình Neural Network, Random Forest và XGBoost, cùng cơ chế quy đổi ngày tương đương. Bổ sung kiểm chứng theo thời gian, lập phương án nhiều tháng, xuất hồ sơ và kết nối OpenAI/Gemini.

Đại lượng được dự báo là điện năng theo tháng, không phải công suất phụ tải đỉnh. Chương trình chưa được hiệu chỉnh hoặc nghiệm thu bằng dữ liệu vận hành của đơn vị.

## 1. Cài đặt và mở chương trình

### Windows

1. Cài Python 3.12 bản 64-bit, có Python Launcher. Máy cần có mạng khi cài thư viện.
2. Giải nén toàn bộ bộ mã vào một thư mục, ví dụ `D:\DuBaoDien`.
3. Chạy `cai_dat_windows.bat` một lần. Nếu cài chưa thành công, đọc thông báo trong cửa sổ; không chuyển sang bước tiếp theo.
4. Chạy `chay_windows.bat`. Giữ cửa sổ này mở khi sử dụng.
5. Mở địa chỉ `http://127.0.0.1:8501` nếu trình duyệt chưa tự mở.

Hai tệp `.bat` sử dụng chữ không dấu để tránh lỗi bảng mã của cửa sổ lệnh. Nội dung ứng dụng và tài liệu sử dụng tiếng Việt Unicode.

### Cài bằng lệnh

Trong thư mục đã giải nén, tạo môi trường riêng:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ai.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

Trên Linux/macOS dùng `python3 -m venv .venv` và `.venv/bin/python` thay cho đường dẫn Python của Windows. Bản đã kiểm tra chạy trên Linux/Python 3.12; chưa chạy thực tế các tệp `.bat` trên Windows.

Các thư viện trực tiếp được cố định ở phiên bản đã kiểm tra. Thư viện phụ thuộc gián tiếp do pip giải quyết. Khi nâng phiên bản, chạy lại bộ kiểm tra trước khi sử dụng dữ liệu chính thức.

## 2. Thử nhanh, không cần API key

1. Bật **Dùng dữ liệu minh họa** ở thanh bên.
2. Chọn bước **2. Kiểm chứng và dự báo**, bấm **Kiểm chứng và chạy dự báo**.
3. Xem bảng MAE, RMSE, MAPE, WAPE và so sánh kết quả các mô hình.
4. Chọn bước **3. Chốt phương án** để sửa hệ số, ngày cắt điện tương đương hoặc tỷ lệ điều chỉnh thêm. Ghi lý do khi thay đổi.
5. Tải hồ sơ Excel hoặc tệp JSON phương án.

Dữ liệu minh họa là số liệu giả lập, không phải số liệu của EVN hay Tây Ninh. Phần dự báo số liệu chạy độc lập với AI đọc tin tức; sau khi cài thư viện, không cần mạng để huấn luyện và tính dự báo.

## 3. Chuẩn bị dữ liệu thật

Tắt chế độ minh họa, tải **Tệp Excel mẫu**, rồi điền dữ liệu. Có thể dùng chung một workbook với hai sheet riêng và tải cùng tệp vào cả hai ô; chọn đúng bảng lịch sử và dự báo.

| Cột | Lịch sử | Đầu vào dự báo | Quy tắc |
|---|---|---|---|
| Năm, Tháng | Bắt buộc | Bắt buộc | Một dòng/tháng; không trùng tháng |
| Tổng thương phẩm | Bắt buộc | Không sử dụng | Giá trị lịch sử > 0, cùng đơn vị đã chọn |
| Nhiệt độ TB | Tùy chọn | Tùy chọn | °C; phải có ở cả hai bảng hoặc xóa ở cả hai |
| Độ ẩm | Tùy chọn | Tùy chọn | % trong khoảng 0–100; cùng quy tắc hai bảng |
| Số ngày | Tùy chọn | Tùy chọn | Nếu có phải khớp lịch; nếu không có sẽ tự tính |
| Số ngày cắt điện | Tùy chọn | Tùy chọn | Ngày tương đương, cho phép số lẻ; thiếu cột thì mặc định 0 |

- Lịch sử: tối thiểu 24 tháng liên tục; nên có ít nhất 36 tháng. Để kiểm chứng một chân trời h tháng trên n kỳ, cần tối thiểu `24 + h + n − 1` tháng lịch sử.
- Dự báo: trong phạm vi 12 tháng sau tháng lịch sử cuối cùng. Muốn đánh giá các tháng đã có thực tế, dùng kiểm chứng; không đưa chúng vào tệp dự báo để so sánh ngay trên tập huấn luyện.
- Không để cột tùy chọn tồn tại nhưng trống. Nếu chưa có dữ liệu, xóa hẳn cột ở cả hai bảng thay vì điền số 0 giả định.
- Chọn đúng kWh hoặc triệu kWh. Ứng dụng hiểu số đầu vào theo đơn vị này, không tự đoán đơn vị.
- Chọn dấu thập phân cho các ô lưu dạng chữ. Ô số Excel không bị thay đổi. Với chuỗi `1.234,56`, chọn dấu phẩy; với `1,234.56`, chọn dấu chấm.
- Quét tối đa 30 dòng đầu để tìm tiêu đề Năm/Tháng, cho phép chọn sheet. Bảng tháng tối đa 1.200 dòng, tệp tối đa 20 MB; khi vượt giới hạn ứng dụng báo lỗi, không cắt bớt âm thầm.

## 4. Lịch nghỉ và cắt điện

### Lịch nghỉ

Khai báo ngày cụ thể, loại **Lễ** hoặc **Tết**, gồm ngày nghỉ bù áp dụng cho phạm vi đang phân tích. Có thể nhập trực tiếp hoặc tải CSV/Excel; Excel cần sheet `Lich_nghi`, cột `Ngày`, `Loại`. Dùng ngày dạng ô Date hoặc `YYYY-MM-DD`.

Một ngày lễ rơi vào Chủ nhật chỉ thuộc nhóm Lễ, không đồng thời thuộc nhóm Chủ nhật. Phần mềm chặn ngày khai báo trùng và kiểm tra tổng số ngày đúng lịch, kể cả năm nhuận.

Không tự chuyển bảng số ngày lễ/Tết của mã cũ thành ngày cụ thể vì không có đủ thông tin xác định ngày trùng. Không cung cấp một lịch nghỉ mặc định được coi là chính thức. Người sử dụng phải kiểm tra lịch cho cả lịch sử và dự báo. Nếu chấp nhận chưa xét lễ/Tết, tất cả ngày được phân loại theo thứ trong tuần và giả định này được ghi trong Excel.

### Hệ số

Giá trị khởi đầu được giữ từ mã cũ: thứ Hai 0,95; thứ Bảy 0,94; Chủ nhật 0,79; lễ 0,56; Tết 0,40; cắt điện 0,50. Ngày thường thứ Ba–thứ Sáu có hệ số 1. Đây là giả định do người sử dụng đặt, chưa phải hệ số ước lượng từ số liệu thực tế.

### Cắt điện

Không dùng số vụ cắt điện làm số ngày cắt điện. Ngày cắt điện tương đương phải xét phạm vi chịu ảnh hưởng và thời lượng trên toàn bộ vùng dự báo. Nếu chưa có căn cứ quy đổi, cần người lập phương án xác định lại đầu vào.

Bản mới dùng giả định ảnh hưởng cắt điện phân bố đều theo cơ cấu ngày của tháng:

```text
D_lịch = tổng của (số ngày mỗi loại × hệ số loại ngày)
a = 1 − (ngày cắt điện tương đương / số ngày trong tháng) × (1 − k_cắt_điện)
D_tương_đương = D_lịch × a
```

Cách này không tự trừ mọi ngày cắt điện vào nhóm thứ Ba–thứ Sáu. Đây vẫn là mô hình tổng hợp theo tháng, chưa tính từng khoảng ngừng cấp điện hoặc điện năng không cung cấp theo xuất tuyến. Công thức khác bản cũ nên kết quả có thể thay đổi dù nhập cùng số ngày cắt điện.

## 5. Mô hình và cách đánh giá

| Thành phần | Cách hoạt động |
|---|---|
| Chuẩn hóa mục tiêu | Điện thương phẩm tháng chia ngày tương đương; lấy log của sản lượng ngày cơ sở dương |
| Xu hướng | Hồi quy tuyến tính theo thời gian trên thang log |
| Cùng kỳ | Sản lượng ngày cơ sở của tháng cùng kỳ năm trước nhân ngày tương đương kỳ dự báo |
| Random Forest, XGBoost | Học phần dư trên thang log sau khi tách xu hướng |
| Neural Network | Chuẩn hóa đặc trưng và mục tiêu; dự báo log sản lượng ngày cơ sở |
| Đầu vào mô hình | Chỉ số thời gian, sin/cos của tháng, nhiệt độ và độ ẩm nếu có |
| Kiểm chứng | Từng mốc chỉ dùng các tháng trước đó để huấn luyện; dự báo ở chân trời đã chọn |

Chuyển từ log1p/expm1 sang log/exp để phép đổi đơn vị kWh ↔ triệu kWh không làm thay đổi bản chất mục tiêu. Điện thương phẩm lịch sử bằng 0, âm hoặc không hữu hạn bị chặn trước khi huấn luyện.

Tất cả mô hình được đánh giá trên cùng tập tháng. Chỉ mô hình có đầy đủ kết quả và ít nhất 3 kỳ hợp lệ mới được gợi ý theo WAPE thấp nhất. Mô hình lỗi không được thay bằng dự báo 0. Nếu chưa đủ kỳ, ứng dụng vẫn cho xem kết quả nhưng không gợi ý mô hình tự động.

- MAE: trung bình độ lớn sai số, cùng đơn vị với sản lượng.
- RMSE: căn trung bình bình phương sai số, nhạy hơn với sai số lớn.
- MAPE: trung bình tỷ lệ sai số tuyệt đối của từng kỳ.
- WAPE: tổng sai số tuyệt đối chia tổng sản lượng thực tế, biểu diễn bằng %.

Thời tiết của tháng kiểm chứng là dữ liệu đã biết trong bảng lịch sử. Vì vậy đây là kiểm chứng có điều kiện theo thời tiết; chưa tái hiện sai số của dự báo thời tiết tại thời điểm lập kế hoạch. Số kỳ còn ít, thay đổi cơ cấu khách hàng, sản xuất công nghiệp, phạm vi quản lý và dữ liệu chưa đồng nhất đều có thể ảnh hưởng kết quả.

Chức năng **Find Seed** được thay bằng kiểm chứng theo thời gian. Seed chỉ phục vụ tái lập; không lựa chọn seed để ép dự báo gần sản lượng mong muốn. Mức mong muốn, nếu có căn cứ nghiệp vụ, được thể hiện bằng một phương án điều chỉnh có lý do.

Chưa xây dựng khoảng tin cậy thống kê, tối ưu tham số tự động hay tự hiệu chỉnh hệ số ngày. Không gọi chênh lệch giữa các mô hình là độ tin cậy hoặc nguyên nhân do thời tiết.

## 6. Chốt phương án và lưu hồ sơ

Phương án lưu đồng thời các tháng. Thay mô hình làm cơ sở sẽ dùng cùng bộ hệ số đang nhập để tính lại, không làm mất hệ số của tháng khác.

```text
Q_sau_cơ_cấu = Q_mô_hình × D_mới / D_gốc_đã_lưu
Q_chốt = Q_sau_cơ_cấu × (1 + điều_chỉnh_thêm / 100)
```

Mẫu số D_gốc được giữ trong kết quả của chính lần chạy, không tính lại từ thanh cấu hình đã bị sửa. Dữ liệu, lịch nghỉ, hệ số huấn luyện, seed, chân trời kiểm chứng, đơn vị, khu vực, danh sách mô hình và phiên bản thư viện đều tham gia mã nhận diện lần chạy. Khi khác cấu hình hiện tại, kết quả cũ bị đánh dấu; chặn chốt và xuất Excel cho đến khi chạy lại.

- **Tải hồ sơ Excel**: phương án, phân bổ nhóm ngày, dự báo các mô hình, sai số, chi tiết kiểm chứng, cấu hình, lịch nghỉ, dữ liệu đầu vào và nhật ký ghi nhận trong phiên.
- **Tải phương án JSON**: hệ số, lý do và mô hình đang chọn. Không chứa API key. Muốn mở lại phải tải đúng dữ liệu, giữ đúng cấu hình, chạy dự báo rồi mở JSON ở bước 3.
- **Ghi nhận phiên bản phương án**: ghi thêm một mục nhật ký trong phiên; tải Excel để lưu cùng hồ sơ. Đây không phải chữ ký số hay quy trình phê duyệt văn bản.

Tổng nhóm ngày được tính bằng số ngày nhân sản lượng một ngày sau điều chỉnh; tổng các nhóm khớp Q_chốt. Đây là phân bổ bình quân theo giả định, không phải dự báo từng ngày cụ thể.

Phiên Streamlit không phải nơi lưu lâu dài. Tải xuống trước khi đóng hoặc tải lại phiên. Nếu cần lưu tập trung nhiều người dùng, cần xây dựng tiếp phần tài khoản, cơ sở dữ liệu, phân quyền và quản lý phiên bản trên máy chủ của đơn vị.

## 7. Kết nối OpenAI và Gemini

### OpenAI qua API key

Mã gốc đã có nhánh OpenAI nhưng dùng danh sách mô hình ghi cố định. Bản này dùng `OpenAI().responses.parse(...)`, dữ liệu có cấu trúc qua Pydantic và `store=False`. Có thể nhập mã mô hình hoặc lấy danh sách từ tài khoản. Mô hình phải hỗ trợ Responses API và Structured Outputs; quyền truy cập thực tế tùy tài khoản.

1. Vào bước **4. Trợ lý AI**, chọn **OpenAI**.
2. Nhập OpenAI API key trong ô mật khẩu, hoặc cấu hình biến môi trường `OPENAI_API_KEY` trên máy chạy ứng dụng.
3. Mã mặc định là `gpt-4.1-mini`, có thể thay bằng mã phù hợp. Đây không phải cam kết mô hình mới nhất hoặc lựa chọn tối ưu cho mọi tài khoản.
4. Chọn tháng, dán nội dung nguồn và bấm **Phân tích nội dung đã chọn**.

Khóa Gemini không dùng thay cho khóa OpenAI. Luồng này dùng hạn mức API của tài khoản cấu hình; chương trình chưa tích hợp đăng nhập bằng tài khoản ChatGPT. OpenAI có luồng Sign in with ChatGPT riêng cho ứng dụng đủ điều kiện; không đồng nhất luồng đó với cách dùng API key trong gói mã này.

### Gemini

Chọn **Google Gemini**, nhập API key và mã Gemini Flash đang dùng. Để trống mã mặc định nhằm tránh tự đổi phiên bản Flash của người sử dụng. Có thể bấm **Lấy danh sách mô hình** rồi sao chép mã phù hợp.

Sử dụng SDK `google-genai`, không còn dùng đối tượng cấu hình toàn cục của `google.generativeai`. Nhờ đó khóa của các phiên không được ghi đè vào một cấu hình SDK dùng chung.

### Cách AI tham gia phương án

- Chỉ gửi đoạn tin tức đã dán, khu vực và kỳ phân tích. Không gửi hai tệp Excel hay tự gửi toàn bộ dữ liệu khách hàng.
- URL chỉ để ghi nguồn; bản này không tự tải bài báo. Người dùng dán phần nội dung cần phân tích để biết chính xác nội dung gửi ra ngoài.
- AI được yêu cầu trả `null` nếu không đủ căn cứ định lượng điện năng tháng cùng kỳ. Không tự quy đổi tỷ lệ GDP, IIP hoặc nhiệt độ thành tỷ lệ điện thương phẩm.
- Tỷ lệ 0% được xử lý đúng, khác với thiếu dữ liệu. Trích đoạn làm căn cứ phải có trong nội dung đã dán; người dùng vẫn phải kiểm tra phạm vi và kỳ số liệu.
- Nếu có cùng kỳ, chương trình tính mức sản lượng tham khảo. Chỉ khi bấm **Dùng mức tham khảo cho phương án tháng này** mới chuyển mức đó thành điều chỉnh thêm so với phương án hiện có; không cộng trực tiếp phần trăm cùng kỳ vào dự báo.
- Lỗi mạng, hạn mức, mô hình không phù hợp hoặc phản hồi bị từ chối không làm mất dự báo số liệu. Không hiển thị nguyên văn lỗi nhà cung cấp có thể chứa dữ liệu nhạy cảm.

Không lưu API key vào JSON hay Excel. Có thể đặt khóa trong `.streamlit/secrets.toml` trên máy chạy chương trình, dựa theo tệp `.example`; không gửi tệp chứa khóa thật cùng mã nguồn. `store=False` không phải cam kết dữ liệu không rời máy hoặc cam kết về toàn bộ chính sách lưu dữ liệu của nhà cung cấp.

## 8. Cấu trúc mã và kiểm tra

| Tệp | Vai trò |
|---|---|
| app.py | Giao diện và trạng thái phiên |
| data_io.py | Đọc Excel, chuẩn hóa và kiểm tra dữ liệu |
| forecasting.py | Lịch ngày, mô hình, kiểm chứng, tính phương án |
| ai_assistant.py | Tích hợp OpenAI/Gemini, kiểm tra phản hồi có cấu trúc |
| reporting.py | Excel, tệp mẫu, lưu/mở phương án JSON |
| tests/ | Các tình huống kiểm tra logic và luồng giao diện |

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

Đã kiểm tra 19 trường hợp trên môi trường Linux/Python 3.12: ngày lễ trùng Chủ nhật, năm nhuận, tháng trùng/thiếu, số không hữu hạn, sai số ngày, định dạng số, tìm tiêu đề, ngăn trùng tập huấn luyện/dự báo, tính theo thời gian, đổi đơn vị, cân bằng tổng phương án, cắt điện không hợp lệ, JSON không khớp, công thức trong ghi chú Excel, phản hồi OpenAI/Gemini giả lập, 0%/null, từ chối AI, bảo vệ thông báo lỗi và luồng giao diện dữ liệu minh họa.

Chưa kiểm tra bằng API key thật; chưa thử quyền truy cập mô hình, chi phí, hạn mức hoặc chất lượng phản hồi trực tiếp của tài khoản người dùng. Chưa có Excel thực tế của đơn vị để đánh giá độ chính xác dự báo hay khẳng định bản mới chính xác hơn bản cũ.

## 9. Phạm vi chạy

Cấu hình mặc định chỉ lắng nghe tại `127.0.0.1`; phù hợp chạy trên máy người sử dụng. Ứng dụng chưa cung cấp đăng nhập nhân viên, máy chủ HTTPS hay truy cập điện thoại từ Internet. Không nên coi bộ mã này là một hệ thống nhiều người dùng đã triển khai tại đơn vị.

## 10. Tài liệu tham chiếu

Đối chiếu ngày 04/10/2026:

- [OpenAI — Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI — Developer quickstart](https://developers.openai.com/api/docs/quickstart)
- [OpenAI — GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)
- [OpenAI — Sign in with ChatGPT](https://developers.openai.com/siwc/quickstart)
- [Google — Structured outputs với Generate Content](https://ai.google.dev/gemini-api/docs/generate-content/structured-output)
- [scikit-learn — TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Streamlit — AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest)
