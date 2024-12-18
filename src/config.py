# path: ~/Develop/rotating-machinery-fault-analysis/src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # 데이터 관련
    data_dir: Path = Path("data")
    raw_data_dir: Path = data_dir / "raw"
    processed_data_dir: Path = data_dir / "processed"
    
    # 모델 저장 관련
    runs_dir: Path = Path("runs")
    models_dir: Path = runs_dir / "models"
    plots_dir: Path = runs_dir / "plots"
    
    # 학습 파라미터
    batch_size: int = 5000
    learning_rate: float = 0.001
    early_stopping_patience: int = 50
    
    # 데이터 전처리 파라미터
    window_size: int = 15  # 이동평균 필터 윈도우 크기
    
    # 모델 파라미터
    hidden_size: int = 100
    dropout_rate: float = 0.2
    
    def __post_init__(self):
        # 필요한 디렉토리 생성
        for dir_path in [self.processed_data_dir, self.models_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

config = Config()