"""Phân tích định tính tùy chọn Gemini/OpenAI; độc lập mô hình dự báo định lượng."""
from __future__ import annotations

import re
import requests
from bs4 import BeautifulSoup


def lay_noi_dung_tu_link(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join(p.get_text() for p in soup.find_all('p'))
            return (text, '✅ Đã đọc link!') if len(text) > 50 else (None, '⚠️ Lỗi link.')
        return None, '⚠️ Lỗi link.'
    except requests.RequestException:
        return None, '⚠️ Lỗi đọc web.'


def trich_xuat_so(text):
    try:
        matches = re.findall(r'-?\d+(?:\.\d+)?', str(text))
        if matches:
            return max(min(float(matches[0]), 50.0), -50.0)
    except (TypeError, ValueError):
        pass
    return 0.0


def danh_sach_model_gemini(api_key):
    """Sử dụng SDK google-generativeai đúng bản gốc, không tự đổi nhà cung cấp."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    result = []
    for model in genai.list_models():
        if ('generateContent' in model.supported_generation_methods
                and 'gemini' in model.name):
            result.append(model.name)
    result.sort(key=lambda m: 100 if 'pro' in m.lower()
                else (50 if '1.5' in m else 10), reverse=True)
    return result


def xu_ly_du_lieu_dinh_tinh(api_key, input_data, target_model_name, provider):
    if not api_key:
        return 0.0, '⚠️ Chưa nhập API Key.', 'Thủ công'
    text_data, status = input_data, ''
    if input_data.strip().startswith('http'):
        extracted, msg = lay_noi_dung_tu_link(input_data)
        if extracted:
            text_data = extracted
            status = msg + '\n'
        else:
            return 0.0, msg, ''
    prompt = (
        f"Đọc thông tin sau: '{text_data[:3000]}'. "
        'Hãy đánh giá xem phụ tải điện tháng này sẽ TĂNG hay GIẢM bao nhiêu % '
        'so với CÙNG KỲ NĂM TRƯỚC. Chỉ đưa ra con số ước lượng dựa trên tác động '
        '(thời tiết, kinh tế...). Trả về định dạng: SỐ | LÝ DO NGẮN GỌN. '
        'Ví dụ: +5.5 | Nắng nóng hơn năm ngoái.'
    )
    try:
        if provider == 'Google Gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(target_model_name)
            response = model.generate_content(prompt)
            res = response.text.strip()
        elif provider == 'OpenAI ChatGPT':
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=target_model_name,
                messages=[{'role': 'system', 'content': 'Bạn là chuyên gia dự báo phụ tải điện.'},
                          {'role': 'user', 'content': prompt}],
                temperature=0.5)
            res = response.choices[0].message.content.strip()
        else:
            return 0.0, '⚠️ Nhà cung cấp AI không được hỗ trợ.', ''
        if '|' in res:
            parts = res.split('|', 1)
            val = trich_xuat_so(parts[0])
            return val, f'{status}✅ AI Đã xong ({provider} - {target_model_name})', parts[1].strip()
        val = trich_xuat_so(res)
        if val != 0.0:
            return val, f'{status}✅ AI Đã xong (Tự bắt số)!', res
        return 0.0, '⚠️ AI không tìm thấy số.', res
    except Exception as exc:
        return 0.0, f'❌ Lỗi AI: {str(exc)[:100]}...', 'Lỗi'
