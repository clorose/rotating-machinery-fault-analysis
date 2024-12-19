import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy import signal
from typing import Optional, Tuple, Dict
import json

class DnnPreprocessor:
    def __init__(self):
        # Scalers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.robust_scaler = RobustScaler()
        
        # Dimensionality reduction
        self.pca = None
        self.pca_n_components = None
        
        # Signal processing
        self.bandpass_params = None
        self.smoothing_window = None
        
        # Outlier detection
        self.outlier_threshold = None
        self.feature_means = None
        self.feature_stds = None
        
        # Feature selection
        self.selected_features = None
        
    def remove_outliers(self, x: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Z-score 기반 이상치 제거"""
        if self.feature_means is None:
            self.feature_means = np.mean(x, axis=0)
            self.feature_stds = np.std(x, axis=0)
            self.outlier_threshold = threshold
            
        z_scores = np.abs((x - self.feature_means) / self.feature_stds)
        mask = np.all(z_scores < threshold, axis=1)
        return x[mask]
    
    def apply_bandpass_filter(self, x: np.ndarray, 
                            lowcut: float = 0.1, 
                            highcut: float = 0.4, 
                            fs: float = 1.0,
                            order: int = 5) -> np.ndarray:
        """주파수 대역 필터링"""
        self.bandpass_params = {
            'lowcut': lowcut,
            'highcut': highcut,
            'fs': fs,
            'order': order
        }
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, x, axis=0)
    
    def smooth_signals(self, x: np.ndarray, window_size: int = 5) -> np.ndarray:
        """이동 평균을 사용한 신호 스무딩"""
        self.smoothing_window = window_size
        return np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='valid'),
            axis=0, arr=x
        )
    
    def apply_pca(self, x: np.ndarray, n_components: float = 0.95) -> np.ndarray:
        """PCA를 사용한 차원 축소"""
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            self.pca_n_components = n_components
            return self.pca.fit_transform(x)
        return self.pca.transform(x)
    
    def select_features(self, x: np.ndarray, feature_indices: list) -> np.ndarray:
        """특정 피처만 선택"""
        self.selected_features = feature_indices
        return x[:, feature_indices]
    
    def normalize(self, x: np.ndarray, method: str = 'standard') -> np.ndarray:
        """데이터 정규화/표준화"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
            
        if method == 'standard':
            return self.standard_scaler.fit_transform(x)
        elif method == 'minmax':
            return self.minmax_scaler.fit_transform(x)
        elif method == 'robust':
            return self.robust_scaler.fit_transform(x)
        else:
            raise ValueError("Unknown normalization method")
    
    def transform(self, x: np.ndarray) -> np.ndarray:
        """저장된 파라미터로 변환"""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
            
        # 저장된 전처리 단계들 순차적으로 적용
        if self.selected_features is not None:
            x = self.select_features(x, self.selected_features)
            
        if self.bandpass_params is not None:
            x = self.apply_bandpass_filter(x, **self.bandpass_params)
            
        if self.smoothing_window is not None:
            x = self.smooth_signals(x, self.smoothing_window)
            
        if self.outlier_threshold is not None:
            x = self.remove_outliers(x, self.outlier_threshold)
            
        if self.pca is not None:
            x = self.pca.transform(x)
            
        return torch.FloatTensor(x)
    
    def save_params(self, path: str):
        """전처리 파라미터 저장"""
        params = {
            'pca_n_components': self.pca_n_components,
            'bandpass_params': self.bandpass_params,
            'smoothing_window': self.smoothing_window,
            'outlier_threshold': self.outlier_threshold,
            'selected_features': self.selected_features
        }
        
        np.save(f'{path}/standard_scaler_mean.npy', self.standard_scaler.mean_)
        np.save(f'{path}/standard_scaler_scale.npy', self.standard_scaler.scale_)
        
        if self.pca is not None:
            np.save(f'{path}/pca_components.npy', self.pca.components_)
            np.save(f'{path}/pca_mean.npy', self.pca.mean_)
            
        with open(f'{path}/preprocessing_params.json', 'w') as f:
            json.dump(params, f)
    
    def load_params(self, path: str):
        """전처리 파라미터 로드"""
        self.standard_scaler.mean_ = np.load(f'{path}/standard_scaler_mean.npy')
        self.standard_scaler.scale_ = np.load(f'{path}/standard_scaler_scale.npy')
        
        with open(f'{path}/preprocessing_params.json', 'r') as f:
            params = json.load(f)
            
        self.pca_n_components = params['pca_n_components']
        self.bandpass_params = params['bandpass_params']
        self.smoothing_window = params['smoothing_window']
        self.outlier_threshold = params['outlier_threshold']
        self.selected_features = params['selected_features']
        
        if self.pca_n_components is not None:
            self.pca = PCA(n_components=self.pca_n_components)
            self.pca.components_ = np.load(f'{path}/pca_components.npy')
            self.pca.mean_ = np.load(f'{path}/pca_mean.npy')