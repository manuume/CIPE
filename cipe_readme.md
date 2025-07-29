# Customer Influence & Persuasion Engine (CIPE)

**Live Demo:** (https://cipe-demo.streamlit.app/)

## Project Overview

This project moves beyond traditional conversion prediction to answer a more critical business question: 

### Moving Beyond "Who Will Buy?" to "Who Can We Influence?"

Standard models identify customers likely to convert, but often waste marketing budget on two groups:

- **Sure Things:** Customers who would have converted anyway.
- **Lost Causes:** Customers who will not convert, regardless of an offer.

The Customer Influence & Persuasion Engine (CIPE) is an end-to-end uplift modeling system that identifies the **"Persuadables"**—the crucial segment of customers who make a purchase only because they received a marketing offer. By focusing on this group, businesses can dramatically increase campaign profitability and maximize Return on Investment (ROI).

## Key Features

- **Uplift Score Prediction:** Quantifies the net impact of a marketing offer on an individual customer's likelihood to convert.

- **Customer Persona Identification:** Automatically segments customers into four key personas: Persuadables, Sure Things, Lost Causes, and Do Not Disturb.

- **Actionable Recommendations:** Provides clear, strategic advice for each customer persona, enabling data-driven marketing decisions.

- **Interactive Dashboard:** A user-friendly Streamlit interface for non-technical users to analyze customer profiles in real-time.

## Tech Stack & Methodology

**Core Libraries:** Python, Pandas, Scikit-learn, XGBoost, Streamlit

**Modeling Approach:** A Two-Model (T-Learner) approach was implemented using XGBoost for its high performance and scalability. Two separate models were trained: one on the "treatment" group (received offer) and one on the "control" group (no offer). The difference in their predictions yields the uplift score.

**Model Explainability:** SHAP was used to interpret the model's predictions, providing deep insights into the key features that drive customer persuadability.

## Key Results & Insights

The model's performance was evaluated using a Qini Curve, which measures the incremental conversions gained by targeting customers based on their uplift score.

The curve demonstrates that the CIPE model vastly outperforms random targeting, confirming its effectiveness in identifying high-impact customers. Targeting just the top 25% of customers ranked by uplift score captures the vast majority of potential incremental revenue.

**Key Insight:** The model revealed that the most "Persuadable" customers are typically new users who show high engagement (e.g., many pages viewed) but have not yet made a significant purchase. This insight allows for the creation of highly targeted and efficient marketing campaigns.

## Project Structure

```
CIPE_Project/
|
|- app/
|  |- app.py
|  |- requirements.txt
|
|- data/
|  |- (Data should be placed here)
|
|- models/
|  |- model_ctrl.pkl
|  |- model_treat.pkl
|
|- notebooks/
|  |- 1_data_analysis_and_modeling.ipynb
|
|
|- README.md
```

## Setup and Usage

To run this project locally, please follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/CIPE_Project.git
cd CIPE_Project
```

### 2. Download the dataset:

Download the dataset from [this link]((https://drive.google.com/file/d/1MNG6bPPZVqPJqltVnLdF049ASfLndrp0/view?usp=drive_link)) and place `CIPE_data.csv` inside the `data/` folder.

### 3. Set up a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install the required dependencies:

Navigate to the app directory and install the packages.

```bash
cd app
pip install -r requirements.txt
```

### 5. Run the Streamlit application:

```bash
streamlit run app.py
```

The application should now be running in your web browser.
