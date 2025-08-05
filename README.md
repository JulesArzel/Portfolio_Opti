# Portfolio Optimization & Reinforcement Learning

## Project Overview

📂 RL_Portfolio_Project/
├── data/
│   ├── data.py
├── features/
│   ├── features_builder.py
│   ├── markowitz.py
│   └── regime.py
├── env/
│   └── portfolio_env.py
├── models/
│   └── rl_agent.py
├── main.ipynb
└── backtests/
📂 Labs/
├── Factor_Model.ipynb
├── HMM_Ruptures_ML.ipynb
└── Markowitz_BlackLitterman.ipynb

This repository is a comprehensive exploration of **quantitative portfolio optimization**, blending classic financial theory with modern machine learning and reinforcement learning (RL). It follows a structured pipeline:

1. **Markowitz Mean–Variance & Black–Litterman**  
2. **Fama–French Three-Factor Analysis**  
3. **Market Regime Detection using HMM & ML**  
4. **Dynamic RL-based Portfolio Allocation with PPO**

Each part builds on the previous one, culminating in a dynamic portfolio agent that learns to allocate capital intelligently over time.

---

## Lab Notebooks & Modules

### 1. Mean–Variance Optimization & Black–Litterman
- **Markowitz**: Explores Mean-Variance theory (Min Variance Portfolio, CML, ...) Implements the tangent portfolio, allowing for unconstrained leverage/shorting.
- **Black–Litterman**: Integrates investor views to adjust the prior equilibrium returns—though optional in this pipeline

**Files/Modules**:  
- `Labs/Markowitz_BlackLitterman.ipynb`  
- `features/markowitz.py`

---

### 2. Fama–French Three-Factor Model
- Explores return drivers via market, size, and value factors.
- Regresses asset returns on Fama–French factors to assess explanatory power and factor betas.

**Files/Modules**:  
- `Labs/Factor_Model.ipynb`  


---

### 3. Regime Detection with HMM & ML
- **Hidden Markov Model (HMM)**: Unsupervised detection of latent market regimes from asset returns.
- **ML Regime Prediction**: Trains an XGBoost classifier on HMM-derived regimes using technical indicators to forecast upcoming regimes.

**Files/Modules**:  
- `Labs/HMM_Ruptures_ML.ipynb`  
- `features/regime.py`

---

### 4. Reinforcement Learning with PPO
- Builds a custom `PortfolioEnv` (Gymnasium) that simulates portfolio allocation over time.
- Feeds the RL agent with state features:  
  - One-hot encoded HMM regimes  
  - Predicted regimes from ML  
  - Markowitz tangent weights  
  - Technical indicators (e.g., volatility, momentum)  
- Trains a PPO agent to maximize a **risk-adjusted reward** \( r_t - \lambda \cdot \sigma_t \)

**Files/Modules**:  
- `main.ipynb`
- `features/feature_builder.py`  
- `env/portfolio_env.py`  
- `models/rl_agent.py`
- `data/data.py`

---

## Theory Highlights

- **Mean–Variance Optimization**: A foundational framework (Markowitz, 1952) balancing return and risk; the tangent portfolio maximizes Sharpe ratio.
- **Black–Litterman**: Combines equilibrium market expectations with investor views for improved posterior estimates.
- **Fama–French 3-Factor Model**: Explains security returns through size (SMB), value (HML), and market risk.
- **Hidden Markov Model (HMM)**: Identifies latent “regimes” (e.g., bull vs. bear) that govern asset behavior.
- **Regime Prediction with XGBoost**: Emulates HMM states based on observable lagged features—allows forward-looking signal.
- **PPO (Proximal Policy Optimization)**: A stable and efficient modern RL algorithm; used here for dynamic allocation decisions.


---





