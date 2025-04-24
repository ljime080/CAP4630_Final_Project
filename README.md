# 📈 CAP4630_Final_Project – Stock Forecasting Ensemble

This project is a Streamlit-based web application for **30-day stock price forecasting** using an ensemble of deep learning models and statistical methods.

Each team member contributed a model architecture, enabling the app to compare performance across various forecasting techniques. Users can input a stock ticker, and the app will train or load models, visualize predictions, and allow CSV download of forecasted prices.

---

## 🧠 Team Model Contributions

| Member | Model Architecture                          |
| ------ | ------------------------------------------- |
| Juan   | LSTM Seq2Seq                                |
| Omar   | LSTM Bidirectional Seq2Seq                  |
| Jayvee | LSTM Bidirectional Regressor                |
| Tito   | GRU Seq2Seq                                 |
| Luis   | ARIMA (Statistical) & Transformer Regressor |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/CAP4630_Final_Project.git
cd CAP4630_Final_Project
git checkout dev
```

### 2️⃣ Create a virtual environment using `pip`

#### On Windows:

```bash
python -m venv project_env
project_env\Scripts\activate
```

#### On macOS/Linux:

```bash
python3 -m venv project_env
source project_env/bin/activate
```

---

### 3️⃣ Install the required Python packages

Make sure you're inside the root folder where `requirements.txt` is located:

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Streamlit app

```bash
streamlit run app.py
```

Then visit [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖼️ App Features

- ✅ Input any stock ticker (e.g. `TSLA`, `AAPL`, `GOOG`)
- 📊 Train models or load pre-trained ones
- 🧠 Uses ARIMA, LSTM, GRU, and Transformer architectures
- 📈 Visualize forecasts in a multi-line Plotly chart
- 💾 Download predictions as CSV
- 🚀 Efficient training with session caching and model persistence

---

## 📁 Project Structure

```
CAP4630_Final_Project/
│
├── app.py                    # Streamlit UI
├── ensemble_forecasts.py     # Model training, forecasting, ARIMA, and utilities
├── requirements.txt          # Python dependencies
├── models/                   # Saved model files organized by ticker
├── plots/                    # Saved visualizations (optional)
└── README.md                 # This file
```

---

## 🧩 Dependencies

Make sure your environment includes:

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `tensorflow`
- `scikit-learn`
- `statsmodels`
- `yfinance`
- `stqdm` _(optional, for progress bars)_

You don’t need to install these manually — they are all in `requirements.txt`.

---

## 🧠 Technologies Used

- **TensorFlow / Keras** – for building LSTM, GRU, and Transformer models
- **Statsmodels** – for classical ARIMA forecasting
- **Plotly** – for interactive line charts
- **Streamlit** – for building the user interface
- **scikit-learn** – for metrics and preprocessing
- **yfinance** – to fetch historical stock prices

---

## 🎓 About the Course

This project was created for **CAP4630: Introduction to Artificial Intelligence** at **Florida International University**.

---

## 👥 Authors

- Juan – LSTM Seq2Seq
- Omar – LSTM Bidirectional Seq2Seq
- Jayvee – LSTM Bidirectional Regressor
- Tito – GRU Seq2Seq
- Luis – ARIMA & Transformer Forecasting

---

## 📌 Notes

- Models are saved inside `/models/<TICKER>/` as `.keras` files.
- The app will **reuse existing models** if available, speeding up repeat forecasts.
- Forecasts are always for the **next 30 days** from the last date in the dataset.

---

Enjoy exploring your forecasts! 📊📈💡
