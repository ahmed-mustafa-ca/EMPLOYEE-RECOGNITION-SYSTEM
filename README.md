# 🧑‍💼 Employee Recognition System

This project is an **AI-powered real-time employee recognition and attendance system** that uses face recognition to automatically detect employees and log their attendance. Built with **Python**, **Deep Learning**, and **Streamlit**, it provides a contactless and intelligent solution for workplace monitoring.

---

## 🚀 Features

- 🎯 **Real-Time Face Recognition** using deep learning models  
- 🧠 **Facenet512 Embeddings** for high-accuracy detection  
- 👥 **Multi-Face Detection** (supports multiple people in one frame)  
- ⏱️ **Automatic Attendance Logging** with timestamp  
- 🚫 **Duplicate Prevention** using cooldown system  
- 📊 **Streamlit Dashboard** for live monitoring and analytics  
- 🗂️ **Employee Management System** (Add, View, Delete employees)  
- 💾 **SQLite Database** for persistent storage  

---

## 🧩 Tech Stack

- Python  
- **OpenCV**  
- **DeepFace / Facenet512**  
- **NumPy**  
- **Pandas**  
- **Streamlit**  
- **SQLite**  

---

## ⚙️ How It Works

1. **Employee Registration**  
   Users upload multiple images of an employee. The system generates facial embeddings and stores them.

2. **Face Detection & Recognition**  
   The webcam captures live video → faces are detected → embeddings are generated → matched with stored data.

3. **Attendance Logging**  
   If a match is found, the system logs:
   - Employee Name  
   - Date  
   - Time  
   - Status (On-time / Late)

4. **Dashboard (Streamlit)**  
   Displays:
   - Live recognition feed  
   - Attendance records  
   - Analytics & stats  

---

## 🧠 Example Workflow

| Step | Action |
|------|--------|
| 1 | Register employee with 3–5 images |
| 2 | Start webcam |
| 3 | System detects face |
| 4 | Matches with database |
| 5 | Logs attendance automatically |

---

## 📊 Example Output

Name: Ahmed Mustafa

Confidence: 0.87

Status: Present

Time: 09:12 AM

---

## 📈 Future Improvements

📱 Mobile App (iOS + Android)

☁️ Cloud Deployment (AWS / GCP)

🔔 Notifications (Email / WhatsApp)

🧾 Payroll Integration

🧠 Face Anti-Spoofing (security upgrade)

📊 Advanced Analytics Dashboard

---
