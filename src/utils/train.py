# path: ~/Develop/rotating-machinery-fault-analysis/src/utils/train.py
import torch
import numpy as np
from pathlib import Path
from rich.progress import track

def train_model(model, criterion, optimizer, num_epochs, train_dataloader, 
                valid_dataloader, save_path: Path, console=None):
    """모델을 학습시킵니다."""
    loss_values = []
    loss_values_v = []
    check = 0
    accuracy_past = 0
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in track(range(1, num_epochs + 1), description="Training"):
        # 학습
        model.train()
        batch_number = 0
        running_loss = 0.0
        
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            y_hat = model(x_train)
            loss = criterion(y_hat, y_train.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_number += 1
            
        loss_values.append(running_loss / batch_number)
        
        # 검증
        model.eval()
        accuracy = 0.0
        total = 0.0
        
        with torch.no_grad():
            for x_valid, y_valid in valid_dataloader:
                v_hat = model(x_valid)
                v_loss = criterion(v_hat, y_valid.long())
                _, predicted = torch.max(v_hat.data, 1)
                total += y_valid.size(0)
                accuracy += (predicted == y_valid).sum().item()
                
            loss_values_v.append(v_loss.item())
            accuracy = accuracy / total
            
        if epoch % 10 == 0 and console:
            console.print(f'Epoch [{epoch}/{num_epochs}] '
                        f'Train Loss: {loss.item():.6f}, '
                        f'Valid Loss: {v_loss.item():.6f}, '
                        f'Accuracy: {accuracy:.6f}')
        
        # Early stopping 체크
        if accuracy_past > accuracy:
            check += 1
        else:
            check = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
            
        accuracy_past = accuracy
        
        if check > 50:
            console.print('Early stopping triggered!')
            break
            
    return loss_values, loss_values_v

def evaluate_model(model, test_dataloader):
    """모델을 평가합니다."""
    model.eval()
    total = 0.0
    accuracy = 0.0
    
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            total += y_test.size(0)
            accuracy += (predicted == y_test).sum().item()
            
    return accuracy / total