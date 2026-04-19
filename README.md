#  Steel Surface Defect Detection System


---

## 📖 Project Overview

This project presents an **intelligent vision-based system** for detecting surface defects in steel using deep learning and machine learning techniques.

The system integrates **object detection, classification, explainable AI, and production analytics** into a single dashboard for real-time industrial monitoring.

---

## 🚀 Key Features

* 🔍 Defect detection using **YOLOv8**
* 🧠 Classification using **ResNet50 + Random Forest**
* 🔬 Explainable AI using **Grad-CAM**
* 📊 Real-time analytics dashboard (Streamlit)
* ⚙️ OEE (Overall Equipment Effectiveness) calculation
* 🚨 Automated alert system for high defect rates
* 📋 Downloadable inspection logs

---

## 🧠 System Architecture

| Component          | Model / Technology | Purpose                   |
| ------------------ | ------------------ | ------------------------- |
| Object Detection   | YOLOv8s            | Detect + localize defects |
| Feature Extraction | ResNet50           | Extract deep features     |
| Classification     | Random Forest      | Classify defect type      |
| Explainability     | Grad-CAM           | Visualize model focus     |
| Dashboard          | Streamlit          | Real-time monitoring UI   |

---

## 📊 Dataset

**NEU Surface Defect Dataset**

| Class           | Train | Val | Test |
| --------------- | ----- | --- | ---- |
| Crazing         | 205   | 60  | 36   |
| Inclusion       | 214   | 60  | 38   |
| Patches         | 247   | 62  | 44   |
| Pitted Surface  | 306   | 58  | 54   |
| Rolled-in Scale | 204   | 60  | 36   |
| Scratches       | 196   | 60  | 35   |

---

## 📈 Results

| Model                 | Task                      | Metric              | Score       |
| --------------------- | ------------------------- | ------------------- | ----------- |
| YOLOv8s               | Detection + Localization  | mAP@50              | **72.1%**   |
| ResNet50 + RF         | Classification            | Accuracy            | **97.1%**   |
| ResNet50 (Fine-tuned) | Classification (Grad-CAM) | Validation Accuracy | **~96–98%** |

---

## ❓ Why is YOLO mAP lower than RF accuracy?

This is expected because:

* **YOLOv8** performs both:

  * Object detection (bounding box localization)
  * Classification

* **Random Forest** performs only:

  * Classification (no localization)

👉 YOLO must satisfy:

* Correct class ✔️
* Correct bounding box (IoU ≥ 50%) ✔️

👉 Therefore:

* mAP is naturally lower than classification accuracy
* But YOLO is **more useful in real-world production**

---

## ⚙️ OEE Calculation

OEE (Overall Equipment Effectiveness) is calculated as:

OEE = Availability × Performance × Quality

Where:

* Quality = (Total Inspected − Defective) / Total Inspected

👉 OEE ≥ 85% is considered **world-class manufacturing performance**

---

## 🖥️ How to Run the Project

### 🔹 Option 1: Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place these files in the same folder:

* `yolov8_best.pt`
* `rf_model.pkl`
* `val_images.csv`
* `resnet50_finetuned.pth`

---

### 🔹 Option 2: Run in Google Colab

* Run all notebook cells
* Use ngrok (Cell 11)
* Open generated public URL

---

## 📊 Dashboard Features

| Tab               | Description                           |
| ----------------- | ------------------------------------- |
| 🎯 Live Detection | Real-time defect detection simulation |
| 📊 Analytics      | Charts, defect distribution, OEE      |
| 🔬 Grad-CAM       | Explainable AI heatmaps               |
| 📋 Inspection Log | Downloadable inspection data          |
| ℹ️ About          | Project summary                       |

---

## 🎯 Demo Instructions

1. Run the Streamlit app
2. Go to **Live Detection → Start Inspection**
3. Observe real-time detection
4. Check **Analytics tab** for OEE and charts
5. Use **Grad-CAM tab** for explainability
6. Download logs from **Inspection Log**

---




## 🏭 Applications

* Industrial quality inspection
* Automated defect detection
* Smart manufacturing systems
* Production line monitoring

---

## 🧾 Conclusion

This project demonstrates how **deep learning + machine learning + explainable AI + analytics** can be combined to build a **real-world industrial monitoring system**.

---

**Author:** V.S. Sre Nanthini (810022104026)

Department of Computer Science and Engineering

Anna University, BIT Campus, Trichy - 620024

Under the Guidance of **Dr. K. Latha** , Assistant Professor.
