import torch
import glob
import os
from dataset import detect_paths

def verify():
    _, output_dir = detect_paths()
    print(f"🔍 VERIFYING DATA ALIGNMENT IN: {output_dir}")
    
    files = glob.glob(os.path.join(output_dir, "**", "*.pt"), recursive=True)
    if not files:
        print("❌ No files found! Run train.py first to generate data.")
        return

    print(f"   Found {len(files)} total tensor files.")
    
    # Check random sample
    sample = torch.load(files[0], weights_only=False)
    data = sample['data']
    label = sample['label']
    
    print("\n✅ SHAPE CHECK:")
    print(f"   Tensor Shape: {data.shape} (Expected: [750, 9])")
    print(f"   Label Type:   {type(label)} (Expected: int)")
    
    if data.shape == (750, 9):
        print("\n🎉 ALIGNMENT SUCCESS: Data is 500Hz, 1.5s, 9 Channels.")
    else:
        print("\n⚠️ ALIGNMENT FAILURE: Dimensions mismatch!")

if __name__ == "__main__":
    verify()
