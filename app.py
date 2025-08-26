<<<<<<< HEAD
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ------------------------------
# Load preprocessors
# ------------------------------
entity_encoder, scaler = joblib.load("preprocessors.pkl")

# Target column (same as training)
target = "Depressive disorders (share of population) - Sex: Both - Age: Age-standardized"

# Load dataset just to know feature names
df = pd.read_csv("1- mental-illnesses-prevalence.csv")
X_num = df.select_dtypes(include="number").drop(columns=[target, "Year"], errors="ignore")
numeric_features = X_num.columns.tolist()

# ------------------------------
# Model Definition (same as training)
# ------------------------------
class DepressionModel(nn.Module):
    def __init__(self, n_entities, n_num_features):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, 8)  # must be 8
        self.fc1 = nn.Linear(8 + n_num_features, 64)  # 8 + 4 = 12
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, entity, num_features):
        entity_vec = self.entity_emb(entity)
        x = torch.cat([entity_vec, num_features], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ------------------------------
# Load trained model
# ------------------------------
n_entities = len(entity_encoder.classes_)
model = DepressionModel(n_entities, len(numeric_features))
model.load_state_dict(torch.load("depression_model.pt", map_location="cpu"))
model.eval()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Depression Prevalence Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Depression Prevalence Prediction")
st.markdown("Enter country and mental health prevalence indicators to predict the **share of depressive disorders** in the population.")

# Country (Entity)
country = st.selectbox("🌍 Select Country", entity_encoder.classes_)

# Numeric inputs
inputs = {}
st.subheader("📊 Mental Illness Prevalence Features")
for col in numeric_features:
    val = st.number_input(f"{col}", value=float(df[col].mean()))
    inputs[col] = val

# Predict button
if st.button("🔮 Predict"):
    # Prepare input
    entity_idx = torch.tensor([entity_encoder.transform([country])[0]], dtype=torch.long)

    num_values = [[inputs[col] for col in numeric_features]]
    num_scaled = scaler.transform(num_values)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        pred = model(entity_idx, num_tensor).item()

    st.success(f"📌 Predicted prevalence of depressive disorders: **{pred:.2f}% of population**")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & PyTorch")
=======
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import joblib

# ------------------------------
# Load preprocessors
# ------------------------------
entity_encoder, scaler = joblib.load("preprocessors.pkl")

# Target column (same as training)
target = "Depressive disorders (share of population) - Sex: Both - Age: Age-standardized"

# Load dataset just to know feature names
df = pd.read_csv("1- mental-illnesses-prevalence.csv")
X_num = df.select_dtypes(include="number").drop(columns=[target, "Year"], errors="ignore")
numeric_features = X_num.columns.tolist()

# ------------------------------
# Model Definition (same as training)
# ------------------------------
class DepressionModel(nn.Module):
    def __init__(self, n_entities, n_num_features):
        super().__init__()
        self.entity_emb = nn.Embedding(n_entities, 8)  # must be 8
        self.fc1 = nn.Linear(8 + n_num_features, 64)  # 8 + 4 = 12
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, entity, num_features):
        entity_vec = self.entity_emb(entity)
        x = torch.cat([entity_vec, num_features], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# ------------------------------
# Load trained model
# ------------------------------
n_entities = len(entity_encoder.classes_)
model = DepressionModel(n_entities, len(numeric_features))
model.load_state_dict(torch.load("depression_model.pt", map_location="cpu"))
model.eval()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Depression Prevalence Predictor", page_icon="🧠", layout="centered")

st.title("🧠 Depression Prevalence Prediction")
st.markdown("Enter country and mental health prevalence indicators to predict the **share of depressive disorders** in the population.")

# Country (Entity)
country = st.selectbox("🌍 Select Country", entity_encoder.classes_)

# Numeric inputs
inputs = {}
st.subheader("📊 Mental Illness Prevalence Features")
for col in numeric_features:
    val = st.number_input(f"{col}", value=float(df[col].mean()))
    inputs[col] = val

# Predict button
if st.button("🔮 Predict"):
    # Prepare input
    entity_idx = torch.tensor([entity_encoder.transform([country])[0]], dtype=torch.long)

    num_values = [[inputs[col] for col in numeric_features]]
    num_scaled = scaler.transform(num_values)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        pred = model(entity_idx, num_tensor).item()

    st.success(f"📌 Predicted prevalence of depressive disorders: **{pred:.2f}% of population**")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & PyTorch")
>>>>>>> 6052a46 (Initial commit with requirements.txt)
