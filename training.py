import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
import os

# Import our custom modules
from dataset import MaestroDataset, MaestroPreprocessor, detect_paths
from model import MAESTRO

def main():
    # 1. Setup Environment
    data_root, output_dir = detect_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 RUNNING ON: {device}")
    
    # 2. Run Preprocessing (if needed)
    print("🔄 Checking Data...")
    processor = MaestroPreprocessor(data_root, output_dir)
    processor.run()

    # 3. Train per Task
    tasks = ['FLANKER', 'PVT', 'NBACK', 'MATB']
    
    for task in tasks:
        task_dir = os.path.join(output_dir, task)
        if not os.path.exists(task_dir): continue
        
        print(f"\n" + "="*40 + f"\n🧠 TRAINING TASK: {task}\n" + "="*40)
        
        # Load Data
        full_ds = MaestroDataset(task_dir)
        if len(full_ds) < 10: 
            print("⚠️ Not enough data.")
            continue
            
        train_size = int(0.8 * len(full_ds))
        train_ds, test_ds = random_split(full_ds, [train_size, len(full_ds)-train_size])
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader  = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        # Initialize Model
        model = MAESTRO().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        # Class Weights
        labels = [d[1] for d in train_ds]
        c0, c1 = labels.count(0), labels.count(1)
        weights = torch.tensor([(c0+c1)/(2*c0), (c0+c1)/(2*c1)]).to(device) if c0*c1 > 0 else None
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Training Loop
        for epoch in range(30):
            model.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"   Epoch {epoch+1}/30 | Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                targets.extend(y.cpu().numpy())
                
        print(classification_report(targets, preds, target_names=['Low', 'High'], zero_division=0))
        torch.save(model.state_dict(), os.path.join(output_dir, f"maestro_{task.lower()}.pth"))

if __name__ == "__main__":
    main()
