import torch
import numpy as np
from pathlib import Path
from rich.progress import track

def train_model(model, criterion, optimizer, num_epochs, train_dataloader, 
                valid_dataloader, save_path: Path, console=None):
    """모델을 학습시킵니다."""
    loss_values = []
    loss_values_v = []
    train_acc_values = []
    valid_acc_values = []

    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_train_acc = 0
    best_valid_acc = 0

    check = 0
    accuracy_past = 0
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    result_path = save_path / 'training_result.txt'

    with open(result_path, 'w') as f:
        f.write('=== Training Results ===\n\n')
        f.write('Epoch\tTrain Loss\tValid Loss\tTrain Acc\tValid Acc\n')
        f.write('------------------------------------------------\n')

    for epoch in track(range(1, num_epochs + 1), description="Training"):
        # 학습
        model.train()
        batch_number = 0
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            y_hat = model(x_train)
            loss = criterion(y_hat, y_train.long())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_number += 1
            
            # 학습 정확도 계산
            _, predicted = torch.max(y_hat.data, 1)
            train_total += y_train.size(0)
            train_correct += (predicted == y_train).sum().item()
            
        train_loss = running_loss / batch_number
        train_accuracy = train_correct / train_total
        loss_values.append(train_loss)
        train_acc_values.append(train_accuracy)
        
        # 검증
        model.eval()
        valid_correct = 0
        valid_total = 0
        valid_loss = 0.0
        
        with torch.no_grad():
            for x_valid, y_valid in valid_dataloader:
                v_hat = model(x_valid)
                v_loss = criterion(v_hat, y_valid.long())
                _, predicted = torch.max(v_hat.data, 1)
                valid_total += y_valid.size(0)
                valid_correct += (predicted == y_valid).sum().item()
                valid_loss += v_loss.item()
                
            valid_accuracy = valid_correct / valid_total
            valid_loss = valid_loss / len(valid_dataloader)
            loss_values_v.append(valid_loss)
            valid_acc_values.append(valid_accuracy)
            
            # 결과 저장
            with open(result_path, 'a') as f:
                f.write(f'[{epoch:3d}]\t{train_loss:.4f}\t\t{valid_loss:.4f}\t\t{train_accuracy:.4f}\t\t{valid_accuracy:.4f}\n')
        
        # 최고 성능 업데이트
        if train_loss < best_train_loss:
            best_train_loss = train_loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
        if train_accuracy > best_train_acc:
            best_train_acc = train_accuracy
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            
        if epoch % 10 == 0 and console:
            console.print(f'Epoch [{epoch}/{num_epochs}] '
                        f'Train Loss: {train_loss:.6f}, '
                        f'Valid Loss: {valid_loss:.6f}, '
                        f'Train Acc: {train_accuracy:.6f}, '
                        f'Valid Acc: {valid_accuracy:.6f}')
        
        # Early stopping 체크
        if accuracy_past > valid_accuracy:
            check += 1
        else:
            check = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
            
        accuracy_past = valid_accuracy
        
        if check > 50:
            console.print('Early stopping triggered!')
            break
    
    # 학습 완료 후 최종 결과 추가
    with open(result_path, 'a') as f:
        f.write('\n\n=== Final Best Results ===\n\n')
        f.write(f'Best Train Loss: {best_train_loss:.6f}\n')
        f.write(f'Best Valid Loss: {best_valid_loss:.6f}\n')
        f.write(f'Best Train Accuracy: {best_train_acc:.6f}\n')
        f.write(f'Best Valid Accuracy: {best_valid_acc:.6f}\n')
            
    return loss_values, loss_values_v, train_acc_values, valid_acc_values

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