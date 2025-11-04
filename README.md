# Stock Price Forecasting using Transformer

**Course:** การเรียนรู้เชิงลึก (Deep Learning)  
**Course Code:** 01204466  
**Instructor:** อาจารย์ภารุจ รัตนวรพันธุ์  
**Faculty:** วิศวกรรมศาสตร์ สาขาวิศวกรรมคอมพิวเตอร์  
**University:** มหาวิทยาลัยเกษตรศาสตร์  
**Semester:** ภาคต้น ปีการศึกษา 2568  

---

## Project Title
**การพยากรณ์ราคาหุ้นด้วยข้อมูลเชิงเวลาแบบหลายตัวแปรโดยใช้สถาปัตยกรรม Transformer**  
(Multivariate Time-Series Stock Price Forecasting using Transformer Architecture)

---

## 1. บทนำ
การเคลื่อนไหวของราคาหุ้นเป็นกระบวนการที่ซับซ้อนและยากต่อการคาดการณ์ เนื่องจากได้รับอิทธิพลจากหลายปัจจัย ทั้งปัจจัยทางเศรษฐกิจ ภาวะตลาด และพฤติกรรมนักลงทุน  
โครงงานนี้มีวัตถุประสงค์เพื่อพัฒนาแบบจำลองเชิงลึก (Deep Learning Model) ที่สามารถเรียนรู้รูปแบบการเปลี่ยนแปลงของข้อมูลราคาหุ้นในอดีต เพื่อใช้ในการพยากรณ์แนวโน้มในอนาคต  
โดยใช้สถาปัตยกรรม Transformer ซึ่งสามารถเรียนรู้ความสัมพันธ์ระยะยาวของข้อมูลเชิงเวลาได้อย่างมีประสิทธิภาพ

---

## 2. เหตุผลและความน่าสนใจ
หัวข้อการพยากรณ์ราคาหุ้นได้รับความสนใจอย่างแพร่หลายในวงการวิศวกรรมข้อมูลและการเงิน  
เนื่องจากสามารถนำไปประยุกต์ใช้ในระบบแนะนำการลงทุน การบริหารความเสี่ยง และระบบเทรดอัตโนมัติ  
โครงงานนี้จึงเน้นการใช้แนวคิด Transformer-based Sequential Modeling กับข้อมูลทางการเงินที่เป็นลำดับเวลา (Time-series data) หลายมิติ เพื่อให้โมเดลเข้าใจพฤติกรรมตลาดได้ดีขึ้น

---

## 3. ทำไมต้องใช้ Deep Learning
### 3.1 วิธีดั้งเดิม
วิธีแบบดั้งเดิม เช่น ARIMA, Linear Regression, GARCH เหมาะกับข้อมูลเชิงเส้น แต่ไม่สามารถจัดการกับความซับซ้อนและความผันผวนของข้อมูลราคาหุ้นได้ดีนัก  

### 3.2 วิธี Deep Learning
สถาปัตยกรรม Transformer ใช้กลไก Self-Attention ในการเรียนรู้ความสัมพันธ์ระยะยาวของข้อมูลโดยไม่ต้องพึ่ง RNN  
- ข้อดี:  
  - เรียนรู้ความสัมพันธ์ที่ไม่เชิงเส้นได้ดี  
  - รองรับข้อมูลขนาดใหญ่และหลายมิติ  
  - สามารถพยากรณ์หุ้นหลายตัวพร้อมกันได้  
- ข้อจำกัด:  
  - ใช้ทรัพยากรคอมพิวเตอร์สูง  
  - ต้องมีการเตรียมข้อมูลที่รอบคอบ  

---

## 4. สถาปัตยกรรมของโมเดล
### 4.1 สรุปองค์ประกอบ

| ส่วนประกอบ | รายละเอียด |
|:--|:--|
| Input Layer | ข้อมูล 37 ฟีเจอร์ย้อนหลัง 90 วัน |
| Embedding Layer | Linear projection: 37 → 128 |
| Positional Encoding | เพิ่มข้อมูลลำดับเวลาให้โมเดลรับรู้ตำแหน่งของ timestep |
| Transformer Encoder (3 Layers) | Multi-Head Attention (8 heads) + Feedforward (ReLU) + LayerNorm + Dropout |
| Output Layer | คาดการณ์ log-return 5 วันข้างหน้า |

### 4.2 โครงสร้างโดยรวม
```
Input (Batch, 90, 37)
↓
Linear Projection (→128)
↓
Positional Encoding
↓
Transformer Encoder × 3
↓
Feedforward (ReLU)
↓
Linear Output (→5)
↓
Predicted 5-day log-return
```

---

## 5. การอธิบายโค้ดและระบบ

### 5.1 Data Preprocessing
- ดึงข้อมูลจาก Yahoo Finance API (yfinance)  
- สร้าง Technical Indicators ด้วย pandas-ta เช่น SMA, EMA, RSI, MACD, Bollinger Bands, ATR  
- เพิ่ม Momentum features และ Cyclical time features (sin/cos day-of-week)  
- ลบคอลัมน์ที่ไม่จำเป็น และใช้ StandardScaler ทำ normalization  
- สร้าง sequence windows (lookback = 90, horizon = 5) และแบ่งข้อมูลเป็น Train/Val/Test  

### 5.2 Model Definition
โมเดลถูกนิยามเป็นคลาส nn.Module  
ใช้สถาปัตยกรรม Transformer Encoder ประกอบด้วย  
Multi-Head Attention, Feedforward Layer, LayerNorm, Dropout, Residual Connection, Positional Encoding  
ใช้ ReLU เป็น activation function และ Linear head สำหรับคาดการณ์ 5 วันล่วงหน้า  

### 5.3 Model Training
ใช้ Adam optimizer และ ReduceLROnPlateau scheduler  
พร้อม MSELoss (Mean Squared Error) เป็น loss function  
ฝึกโมเดลเป็นเวลา 40 epochs โดยวัดค่า Train/Validation loss ทุก epoch  
รันบน GPU (Google Colab T4) เพื่อเพิ่มความเร็วในการคำนวณ  

---

## 6. Dataset & Hyperparameters
### 6.1 Dataset
- Source: Yahoo Finance API  
- Period: 2018 – ปัจจุบัน  
- Stocks Used: 50 บริษัทในดัชนีหลักของสหรัฐฯ (AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN ฯลฯ)  
- Features: ราคา (Open, High, Low, Close, Volume) + Technical Indicators (รวม 37 ตัวแปร)  
- Lookback Window: 90 วัน  
- Forecast Horizon: 5 วัน  

### 6.2 Hyperparameters
| พารามิเตอร์ | ค่า |
|:--|:--|
| Optimizer | Adam |
| Learning Rate / Scheduler | ReduceLROnPlateau (mode='min', factor=0.5, patience=10) |
| Initial LR | 1e-4 |
| Batch Size | 128 |
| Epochs | 40 |
| Loss Function | MSELoss |
| Device | GPU (Google Colab T4) |

---

## 7. Evaluation Results
- Train Loss: 0.7139  
- Validation Loss: 0.7706  
- สังเกตได้ว่า Train Loss ลดลงในขณะที่ Val Loss เพิ่มขึ้นเล็กน้อย แสดงถึง Overfitting เล็กน้อย  
- โมเดลสามารถจับแนวโน้มราคาหุ้นได้ถูกต้อง แม้จะมีความคลาดเคลื่อนในจุดที่ผันผวนมาก  

---

## 8. Conclusion
โมเดล Transformer สามารถเรียนรู้แนวโน้มของราคาหุ้นได้อย่างมีเสถียรภาพและแม่นยำระดับหนึ่ง  
ผลลัพธ์แสดงให้เห็นถึงศักยภาพของ Deep Learning ในการประมวลผลข้อมูลเชิงเวลาและสามารถนำไปต่อยอดในระบบเทรดอัตโนมัติหรือระบบแนะนำการลงทุนในอนาคตได้

---

## 9. References
1. Vaswani, A., et al. (2017). Attention is all you need. NeurIPS 2017, 6000–6010.  
2. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654–669.  
3. Zhang, X., Li, Y., & Pan, L. (2020). Stock movement prediction with spatial-temporal graph transformer. CIKM 2020, 2825–2832.  
4. Yahoo Inc. (n.d.). Yahoo Finance API documentation. Retrieved from https://finance.yahoo.com  
5. Pandas-TA. (n.d.). Technical Analysis Library for Python. Retrieved from https://github.com/twopirllc/pandas-ta  

---

## 10. Team Contribution

| ลำดับ | ชื่อผู้จัดทำ | หน้าที่รับผิดชอบ | สัดส่วนของงาน |
|:--:|:--|:--|:--:|
| 1 | นายก้องภพ ไพเราะ | - จัดเตรียมและประมวลผลข้อมูล (Preprocessing)<br>- สร้างตัวชี้วัดทางเทคนิค (Indicators)<br>- จัดรูปแบบข้อมูลให้เหมาะสมต่อการนำเข้าโมเดล (Formatting Data) | 50% |
| 2 | นายปฐมพงศ์ บวรเจริญพันธุ์ | - ออกแบบและพัฒนาโมเดล Transformer<br>- เขียนส่วนการฝึกสอน (Training Loop)<br>- พัฒนาและทดสอบส่วนการอนุมานผล (Inference/Evaluation) | 50% |

ทั้งสองคนร่วมกันออกแบบแนวทางการทดลอง วิเคราะห์ผลลัพธ์ และจัดทำรายงานฉบับสมบูรณ์

---

## Repository Structure
```
├── data/                  # ข้อมูลดิบและข้อมูลหลัง preprocessing
├── models/                # โมเดล Transformer และเวอร์ชันเทรนแล้ว
├── notebooks/             # Jupyter notebooks สำหรับการทดลอง
├── utils/                 # ฟังก์ชันช่วย preprocess, indicator, evaluation
├── train.py               # สคริปต์หลักสำหรับฝึกโมเดล
├── inference.py           # สคริปต์สำหรับพยากรณ์ผลลัพธ์
└── README.md              # ไฟล์อธิบายโปรเจกต์
```

---

© 2025 Kasetsart University – Department of Computer Engineering
