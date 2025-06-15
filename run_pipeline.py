import os
import torch
import pandas as pd
from train import main as train_main
from inference import main as inference_main

def run_complete_pipeline():
    print("Starting model training...")
    train_main()
    
    print("\nStarting inference on test set...")
    inference_main()
    
    print("\nComplete pipeline executed successfully!")
    print(f"Check {OUTPUT_FILE} for lane cutting detection results")

if __name__ == "__main__":
    run_complete_pipeline()