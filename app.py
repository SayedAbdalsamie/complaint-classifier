# ====================================
# ✅ Streamlit App for Complaint Classifier (no training data required)
# ====================================

import streamlit as st

# ✅ يجب أن يكون هذا أول أمر Streamlit
st.set_page_config(page_title="Complaint Classifier", page_icon="📂", layout="centered")

import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib
import pandas as pd


# =============================
# ✅ تحميل الموديلات والتوكن
# =============================
@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizerFast.from_pretrained("tokenizer_distilbert")
    model_cat = DistilBertForSequenceClassification.from_pretrained("category_model")
    model_sub = DistilBertForSequenceClassification.from_pretrained("subcategory_model")
    return tokenizer, model_cat.eval(), model_sub.eval()


tokenizer, model_cat, model_sub = load_models()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cat.to(device)
model_sub.to(device)


# =============================
# ✅ تحميل الـ LabelEncoders من ملفات .pkl
# =============================
@st.cache_resource
def load_encoders():
    le_cat = joblib.load("category_encoder.pkl")
    le_sub = joblib.load("subcategory_encoder.pkl")
    return le_cat, le_sub


le_cat, le_sub = load_encoders()


# =============================
# ✅ دالة التنبؤ
# =============================
def predict(complaint):
    inputs = tokenizer(
        complaint, return_tensors="pt", truncation=True, padding=True, max_length=128
    ).to(device)

    with torch.no_grad():
        logits_cat = model_cat(**inputs).logits
        logits_sub = model_sub(**inputs).logits

    pred_cat = torch.argmax(logits_cat, dim=1).item()
    pred_sub = torch.argmax(logits_sub, dim=1).item()

    category = le_cat.inverse_transform([pred_cat])[0]
    subcategory = le_sub.inverse_transform([pred_sub])[0]

    return category, subcategory


# =============================
# ✅ واجهة المستخدم - Streamlit UI
# =============================
st.title("📂 Complaint Classifier")
st.markdown("أدخل شكوى، وسيتوقع النموذج التصنيفات التالية:")

complaint_input = st.text_area("✍️ اكتب الشكوى هنا", height=150)

if st.button("🔍 Classify and Generate Report"):
    if complaint_input.strip() == "":
        st.warning("⚠️ Please enter a complaint first.")
    else:
        with st.spinner("⏳ Classifying the complaint and preparing report..."):
            cat, sub = predict(complaint_input)

        st.success("✅ Classification Complete!")
        st.markdown(f"**📁 Category:** `{cat}`")
        st.markdown(f"**📂 Subcategory:** `{sub}`")

        # ✅ Generate English incident report
        st.markdown("---")
        st.subheader("📄 Generated Incident Report (English)")

        st.markdown(
            f"""
**To:**  
Municipal Environmental Affairs Department  

**From:**  
Smart Complaint Classification System  

**Date:**  
{pd.Timestamp.today().strftime('%B %d, %Y')}

**Subject:**  
📢 Public Complaint - {cat} Detected

---

### 📝 Overview of the Complaint
Our AI-powered classification system has automatically detected and classified a public complaint as a case of **{cat}**.

---

### 💬 Original Complaint:
> _"{complaint_input}"_

---

### 📂 Detected Category:
- **Main Category:** {cat}
- **Subcategory:** {sub}

---

### 🚨 Severity Assessment:
The system has flagged this issue based on the content of the complaint. It is recommended to further review this case due to its potential public or environmental impact.

---

### 🧭 Recommended Actions:
- **Inspection:** Immediate assessment of the incident location.
- **Monitoring:** Deploy necessary equipment or teams.
- **Advisory (if needed):** Notify nearby residents if health or safety risks are confirmed.

---

### 🖋️ Prepared By:
Smart Complaint Analysis System  
AI-Powered Monitoring Team  
"""
        )


st.markdown("---")
st.caption("🔧 يعمل بواسطة DistilBERT + Streamlit")
