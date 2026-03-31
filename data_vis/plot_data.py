import pandas as pd
import matplotlib.pyplot as plt

# Load your local logs
lstm_df = pd.read_csv("LSTM_Tuner_training_log.csv")
mamba_df = pd.read_csv("Mamba_Tuner_training_log.csv")

# Plot them together
plt.plot(lstm_df['Epoch'], lstm_df['Val_MSE'], label="LSTM Validation")
plt.plot(mamba_df['Epoch'], mamba_df['Val_MSE'], label="Mamba Validation")
plt.legend()
plt.show()