# 📧 EMAIL CLASSIFIER MODEL

A simple and effective **Email Spam Classifier** project that identifies whether an email is **Spam** or **Ham** using a machine-learning model.  
Is repository me training notebook, pre-trained model (`emailspam_model.pkl`) aur ek simple Flask app (`app.py`) included hai.

---

## 🚀 Features
- Email text ko spam/ham me classify karta hai  
- Preprocessing + ML model training Jupyter Notebook me available  
- Pre-trained model included  
- Flask app ke through prediction interface  
- Easy to run, easy to retrain

---

## 📂 Repository Structure
├── app.py # Flask app to make predictions
├── emailclassificationmodel.ipynb # Notebook for training & evaluation
├── emailspam_model.pkl # Pre-trained ML model
└── templates/ # HTML files for web UI (if any)



---

## 🔧 Installation & Setup

### 1️⃣ Clone the Repository
bash
git clone https://github.com/jainish09/EMAIL-CLASSIFER-MODEL.git
cd EMAIL-CLASSIFER-MODEL 
 
python -m venv venv
venv\Scripts\activate
pip install flask scikit-learn pandas numpy joblib
pip install -r requirements.txt
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"email_text": "Congratulations! You have won a free iPhone!"}'


{
  "prediction": "spam",
  "confidence": 0.92
}
