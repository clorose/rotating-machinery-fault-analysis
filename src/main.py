# path: ~/Develop/rotating-machinery-fault-analysis/src/main.py
import argparse
import torch
from rich.console import Console
from rich.progress import track
from pathlib import Path
from datetime import datetime
from models.dnn import KAMP_DNN
from models.cnn import KAMP_CNN
from models.rnn import KAMP_RNN
from utils.data import load_and_preprocess_data
from utils.train import train_model, evaluate_model
from utils.viz import plot_confusion_matrix, plot_loss_graph
from config import config

console = Console()

def main(args):
    console.print("[bold blue]Starting rotary machine fault analysis...[/bold blue]")
    
    # 모델 실행 시간에 따른 저장 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.runs_dir / f"run_{timestamp}"
    models_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    
    # 필요한 디렉토리 생성
    for dir_path in [models_dir, plots_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드 및 전처리
    with console.status("[bold green]Loading and preprocessing data...") as status:
        train_dataloader, valid_dataloader, test_dataloader, x_test, y_test = load_and_preprocess_data(
            data_dir=config.raw_data_dir,
            processed_dir=config.processed_data_dir,
            batch_size=config.batch_size
        )
    
    models = {
        'dnn': (KAMP_DNN(), models_dir / 'dnn'),
        'cnn': (KAMP_CNN(), models_dir / 'cnn'),
        'rnn': (KAMP_RNN(), models_dir / 'rnn')
    }
    
    # 모델 선택
    selected_models = list(models.keys()) if args.models == 'all' else [args.models]
    
    for name in selected_models:
        model, save_path = models[name]
        console.print(f"\n[bold cyan]Training {name.upper()} model...[/bold cyan]")
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=0.01 # L2 regularization
        )
        
        # 모델 학습
        loss_values, loss_values_v, train_acc_values, valid_acc_values = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.epochs,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            save_path=save_path,
            console=console
        )
        
        # 모델 평가
        accuracy = evaluate_model(model, test_dataloader)
        console.print(f"[bold green]{name.upper()} Test Accuracy: {accuracy:.4f}[/bold green]")
        
        # 결과 시각화
        plot_confusion_matrix(
            model=model,
            x_test=x_test,
            y_test=y_test,
            save_path=plots_dir / f"{name}_confusion_matrix.png"
        )
        
        plot_loss_graph(
            loss_values=loss_values,
            loss_values_v=loss_values_v,
            title=f"{name.upper()} Training & Validation Loss",
            save_path=plots_dir / f"{name}_loss.png"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, default='all',
                    help='Model to train: dnn, cnn, rnn, or all')
    parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train')
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        console.print("\n[bold red]Training interrupted by user[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")