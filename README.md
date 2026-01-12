# DE-Gambling-Detector

ระบบ AI สำหรับตรวจจับรูปภาพเว็บพนัน พร้อมทั้ง Streamlit UI และ FastAPI API

## โครงสร้างโปรเจกต์
- `app/streamlit_app.py` — UI สำหรับอัปโหลดรูปและดูผลลัพธ์
- `app/api.py` — API สำหรับเรียกใช้งานภายนอก
- `src/predict.py` — ฟังก์ชันหลักสำหรับเตรียมภาพและทำนาย
- `weight/` — โมเดลที่เทรนแล้ว

## การติดตั้ง
```bash
pip install -r requirements.txt
```

## วิธีรัน Streamlit UI
```bash
streamlit run app/streamlit_app.py
```

## วิธีรัน FastAPI
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## ตัวอย่างการเรียก API
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@/path/to/your/image.jpg"
```

### Response
```json
{
  "label": "gambling",
  "confidence": 0.92
}
```
