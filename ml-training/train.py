import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from dataset import get_dataloaders
from model import TrafficSignCNN

def evaluate_model(model, test_loader, device, save_report_path=None):
    """
    Evaluates the model on test dataset and prints/saves classification metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    total_samples = len(all_targets)
    correct_samples = sum(p == t for p, t in zip(all_preds, all_targets))
    test_acc = correct_samples / total_samples
    
    print(f"\n=========================================", flush=True)
    print(f"FINAL TEST ACCURACY: {test_acc*100:.2f}% ({correct_samples}/{total_samples})", flush=True)
    print(f"=========================================\n", flush=True)
    
    report_dict = classification_report(all_targets, all_preds, digits=4, output_dict=True)
    report_text = classification_report(all_targets, all_preds, digits=4)
    print("Classification Report:\n", report_text, flush=True)
    
    if save_report_path:
        summary = {
            "test_accuracy": float(test_acc),
            "total_samples": int(total_samples),
            "correct_samples": int(correct_samples),
            "metrics": report_dict
        }
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Evaluation report saved to: {save_report_path}", flush=True)
        
    return test_acc

def train_model(epochs=15, batch_size=64, lr=0.001, evaluate_only=False):
    if not torch.cuda.is_available():
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        print(f"PyTorch using {num_cores} CPU threads for execution.", flush=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    # Load dataset
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    
    model = TrafficSignCNN(num_classes=43).to(device)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    report_path = os.path.join(model_dir, 'evaluation_report.json')
    
    if evaluate_only:
        print(f"\n--- Running Evaluation Only using {best_model_path} ---", flush=True)
        if not os.path.exists(best_model_path):
            print(f"Error: Model checkpoint not found at {best_model_path}")
            sys.exit(1)
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
        return evaluate_model(model, test_loader, device, save_report_path=report_path)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_val_acc = 0.0
    print("\n--- Starting Training Loop ---", flush=True)
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            total_train += images.size(0)
            
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{len(train_loader)} | "
                      f"Current Batch Loss: {loss.item():.4f}", flush=True)
            
        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_correct / total_train
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                total_val += images.size(0)
                
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = val_correct / total_val
        elapsed = time.time() - start_time
        
        scheduler.step(epoch_val_acc)
        
        print(f"==> Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%", flush=True)
        
        # Save best model checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  [Checkpoint] Saved new best model to {best_model_path} (Val Acc: {best_val_acc*100:.2f}%)", flush=True)
            
    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc*100:.2f}%", flush=True)
    
    # Evaluation Phase on Test Set
    print("\n--- Evaluating Best Model on Test Set ---", flush=True)
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    return evaluate_model(model, test_loader, device, save_report_path=report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate TrafficSignCNN on GTSRB dataset.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--evaluate-only", action="store_true", help="Only run evaluation on test set using best_model.pth")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, evaluate_only=args.evaluate_only)
