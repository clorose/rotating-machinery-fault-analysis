# path: ~/Develop/rotating-machinery-fault-analysis/src/utils/data.py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime

def save_state(data, filename: Path, stage: str):
    """현재 데이터 상태를 파일로 저장합니다."""
    if isinstance(data, list):
        # 여러 DataFrame이 있는 경우
        for i, df in enumerate(data):
            df.to_csv(filename.parent / f"{stage}_sensor{i+1}.csv", index=False)
    elif isinstance(data, pd.DataFrame):
        # 단일 DataFrame인 경우
        data.to_csv(filename, index=False)
    elif isinstance(data, np.ndarray):
        # Numpy 배열인 경우
        np.savetxt(filename, data, delimiter=',')
    
    # 데이터 형태 정보도 함께 저장
    with open(filename.parent / f"{stage}_info.txt", 'w') as f:
        f.write(f"Data shape: {data[0].shape if isinstance(data, list) else data.shape}\n")
        f.write(f"Data type: {type(data)}\n")
        if isinstance(data, (np.ndarray, pd.DataFrame)):
            f.write(f"Data description:\n{pd.DataFrame(data).describe().to_string()}")

def load_raw_data(save_dir: Path, data_dir: Path):
    """센서 데이터를 로드합니다."""
    sensors = []
    for i in range(1, 5):
        sensor = pd.read_csv(
            data_dir / f'g1_sensor{i}.csv',
            names=['time', 'normal', 'type1', 'type2', 'type3']
        )
        sensors.append(sensor)
    
    # 원본 데이터 상태 저장
    save_state(sensors, save_dir / "1_raw_data.csv", "raw")
    return sensors

def interpolate_data(save_dir: Path, sensors, sample_rate=0.001):
    """센서 데이터를 선형보간합니다."""
    x_new = np.arange(0, 140, sample_rate)
    interpolated = []
    
    for sensor in sensors:
        y_new = []
        for item in ['normal', 'type1', 'type2', 'type3']:
            f_linear = interpolate.interp1d(sensor['time'], sensor[item], kind='linear')
            y_new.append(f_linear(x_new))
        interpolated.append(pd.DataFrame(
            np.array(y_new).T,
            columns=['normal', 'type1', 'type2', 'type3']
        ))
    
    # 보간된 데이터 상태 저장
    save_state(interpolated, save_dir / "2_interpolated_data.csv", "interpolated")
    return interpolated

def apply_moving_average(data, window_size=15):
    """이동평균 필터를 적용합니다."""
    filtered = np.convolve(data, np.ones(window_size), 'valid') / window_size
    return filtered.reshape(len(filtered), 1)

def prepare_data(save_dir: Path, sensors, window_size=15):
    """데이터를 전처리하고 학습에 사용할 수 있는 형태로 준비합니다."""
    # 이동평균 적용
    normal = np.concatenate([
        apply_moving_average(sensor['normal'], window_size) 
        for sensor in sensors
    ], axis=1)
    
    type1 = np.concatenate([
        apply_moving_average(sensor['type1'], window_size)
        for sensor in sensors
    ], axis=1)
    
    type2 = np.concatenate([
        apply_moving_average(sensor['type2'], window_size)
        for sensor in sensors
    ], axis=1)
    
    type3 = np.concatenate([
        apply_moving_average(sensor['type3'], window_size)
        for sensor in sensors
    ], axis=1)
    
    # 이동평균 적용 후 데이터 저장
    moving_avg_data = np.concatenate([normal, type1, type2, type3], axis=1)
    save_state(moving_avg_data, save_dir / "3_moving_average_data.csv", "moving_average")
    
    # 정규화
    scaler = MinMaxScaler()
    scaler.fit(normal)
    
    normal = scaler.transform(normal)
    type1 = scaler.transform(type1)
    type2 = scaler.transform(type2)
    type3 = scaler.transform(type3)
    
    # 정규화 후 데이터 저장
    normalized_data = np.concatenate([normal, type1, type2, type3], axis=1)
    save_state(normalized_data, save_dir / "4_normalized_data.csv", "normalized")
    
    # 데이터 크기 조정
    normal = normal[30000:130000][:]
    type1 = type1[30000:130000][:]
    type2 = type2[30000:130000][:]
    type3 = type3[30000:130000][:]
    
    # 최종 전처리된 데이터 저장
    final_data = np.concatenate([normal, type1, type2, type3], axis=1)
    save_state(final_data, save_dir / "5_final_preprocessed_data.csv", "final")
    
    return normal, type1, type2, type3

def create_dataloaders(save_dir: Path, normal, type1, type2, type3, batch_size=5000):
    """데이터로더를 생성합니다."""
    # 데이터 분할
    train_data = np.concatenate((
        normal[80000:], type1[80000:],
        type2[80000:], type3[80000:]
    ))
    
    valid_data = np.concatenate((
        normal[60000:80000], type1[60000:80000],
        type2[60000:80000], type3[60000:80000]
    ))
    
    test_data = np.concatenate((
        normal[:60000], type1[:60000],
        type2[:60000], type3[:60000]
    ))
    
    # 분할된 데이터 저장
    split_data = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
    }
    for name, data in split_data.items():
        save_state(data, save_dir / f"6_{name}_split_data.csv", f"split_{name}")
    
    # 레이블 생성
    train_label = np.concatenate((
        np.full((20000,1), 0), np.full((20000,1), 1),
        np.full((20000,1), 2), np.full((20000,1), 3)
    ))
    
    valid_label = np.concatenate((
        np.full((20000,1), 0), np.full((20000,1), 1),
        np.full((20000,1), 2), np.full((20000,1), 3)
    ))
    
    test_label = np.concatenate((
        np.full((60000,1), 0), np.full((60000,1), 1),
        np.full((60000,1), 2), np.full((60000,1), 3)
    ))
    
    # 텐서 변환
    x_train = torch.from_numpy(train_data).float()
    y_train = torch.from_numpy(train_label).float().T[0]
    x_valid = torch.from_numpy(valid_data).float()
    y_valid = torch.from_numpy(valid_label).float().T[0]
    x_test = torch.from_numpy(test_data).float()
    y_test = torch.from_numpy(test_label).float().T[0]
    
    # 데이터로더 생성
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = TensorDataset(x_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(x_valid), shuffle=False)
    
    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)
    
    return train_dataloader, valid_dataloader, test_dataloader, x_test, y_test

def load_and_preprocess_data(data_dir: Path, processed_dir: Path, batch_size=5000):
    """전체 데이터 전처리 파이프라인을 실행합니다."""
    # 타임스탬프가 포함된 처리 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_dir = processed_dir / f"processed_{timestamp}"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    sensors = load_raw_data(processed_dir, data_dir)
    interpolated = interpolate_data(processed_dir, sensors)
    normal, type1, type2, type3 = prepare_data(processed_dir, interpolated)
    return create_dataloaders(processed_dir, normal, type1, type2, type3, batch_size)