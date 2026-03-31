import torch
import time
from train import MambaPIDTuner

def benchmark_model(d_model_size, runs=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking Mamba (d_model={d_model_size}) on {device}...")
    
    # 1. Initialize the model
    model = MambaPIDTuner(input_dim=4, d_model=d_model_size, output_dim=3).to(device)
    model.eval()
    
    # 2. Create "Fake" Telemetry (Batch=1, SeqLen=50, Features=4)
    # This perfectly mimics the shape of data from your flight controller
    dummy_input = torch.randn(1, 50, 4).to(device)
    
    # 3. GPU WARM-UP
    # GPUs have a "wake-up" time. If you don't run a few passes first, 
    # your timing will be artificially slow.
    for _ in range(100):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # 4. THE BENCHMARK
    # CRITICAL: You must synchronize the GPU before starting the clock!
    torch.cuda.synchronize() 
    start_time = time.perf_counter()
    
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # CRITICAL: You must synchronize the GPU again before stopping the clock!
    torch.cuda.synchronize() 
    end_time = time.perf_counter()
    
    # Calculate Metrics
    total_time = end_time - start_time
    avg_time_ms = (total_time / runs) * 1000
    
    # Calculate Total Trainable Parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {params:,}")
    print(f"Average Latency:  {avg_time_ms:.4f} ms per step\n")

if __name__ == "__main__":
    # Test your current baseline
    benchmark_model(d_model_size=64)
    
    # Test what would happen if you made the model 4x smarter
    benchmark_model(d_model_size=256)