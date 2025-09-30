Predicts laptop price based on technical specifications (CPU, RAM, GPU, etc.).

Uses a machine learning pipeline: data processing, encoding, and then a RandomForest model.

Web interface built using Streamlit — users can select specs and see predicted price in real time.

Automatically detects feature columns (numeric/categorical) from your dataset — minimal hardcoding.

Includes a debug section to confirm environment, model, and sample prediction.

Supports flexible usage and future upgrades (you can swap the model, add features).

Safe reading of dataset (CSV) instead of version-sensitive pickles.

How To Run
## 🚀 How to Run the Project Locally

**Prerequisites**  
- Python 3.10+ (or 3.x) installed  
- Git installed  

**Steps**

```bash
# 1. Clone your repo
git clone https://github.com/vamanvs/laptop_price_analysis.git
cd laptop_price_analysis

# 2. Create & activate virtual environment
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate
# macOS/Linux
# source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. (If not already done) Train the model
python train_model.py

# 5. Run the Streamlit app
streamlit run app.py

# 6. Open in browser
# Navigate to http://localhost:8501
