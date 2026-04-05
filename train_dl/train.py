import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from mamba_ssm import Mamba
import pandas as pd  # --- Added Pandas for local logging ---

from dataset import DroneSequenceDataset 

# --- 1. Baseline LSTM Architecture ---
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step_output = lstm_out[:, -1, :] 
        return self.fc(last_step_output)

# --- 2. Mamba Architecture ---
class MambaPIDTuner(nn.Module):
    def __init__(self, input_dim=4, d_model=256, output_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba = Mamba(
            d_model=d_model, 
            d_state=16,      
            d_conv=4,        
            expand=2,        
        )
        self.output_proj = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)  
        x = self.mamba(x)       
        last_step_output = x[:, -1, :] 
        return self.output_proj(last_step_output)

# --- 3. The Training Engine ---
def train_model(model, train_loader, val_loader, model_name, epochs, device='cuda'):
    print(f"\n--- Starting Training: {model_name} ---")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    training_log = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:02d}/{epochs} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}")

        # --- LOCAL LOGGER UPDATE & SAVE ---
        # Append this epoch's data to our list
        training_log.append({
            "Epoch": epoch + 1,
            "Train_MSE": float(avg_train),
            "Val_MSE": float(avg_val)
        })
        
        # Save to CSV immediately. If the script crashes at epoch 45, 
        # you still have 45 epochs of safely saved data!
        df_log = pd.DataFrame(training_log)
        df_log.to_csv(f"{model_name}_training_log.csv", index=False)
        
        # Early Stopping Logic
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), f"{model_name}_weights.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= 20:
            print(f"Early stopping triggered! No improvement for 20 epochs. Best Val MSE: {best_val_loss:.4f}")
            break
            
    print(f"Finished training {model_name}. Log saved to {model_name}_training_log.csv.")

# --- 4. Execution ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Accelerator: {device}")
    
    dataset_path = "data/oracle_trajectories_v2.csv" 
    dataset = DroneSequenceDataset(dataset_path, sequence_length=50)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=True)
    
    lstm_model = BaselineLSTM()
    mamba_model = MambaPIDTuner()
    
    # train_model(lstm_model, train_loader, val_loader, "LSTM_Tuner", epochs=200, device=device)
    train_model(mamba_model, train_loader, val_loader, "Mamba_Tuner", epochs=120, device=device)