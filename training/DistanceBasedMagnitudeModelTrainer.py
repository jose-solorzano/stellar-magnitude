from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from torch import nn, optim

from models.DistanceBasedMagnitudeModel import DistanceBasedMagnitudeModel


class DistanceBasedMagnitudeModelTrainer:
    def __init__(self, num_passes: int, num_splits: int, max_training_items: int, num_epochs: int, lr: float,
                 id_column='source_id', response_column='response', target_column='phot_g_mean_mag'):
        self.num_passes = num_passes
        self.lr = lr
        self.num_epochs = num_epochs
        self.max_training_items = max_training_items
        self.id_column = id_column
        self.response_column = response_column
        self.target_column = target_column
        self.num_splits = num_splits
        self.mag_column_groups = [
            ['phot_bp_mean_mag', 'phot_g_mean_mag', 'phot_rp_mean_mag'],
            ['tmass_j_m', 'tmass_h_m', 'tmass_ks_m'],
            ['gsc23_b_mag', 'gsc23_v_mag'],
        ]
        self.mag_columns = [
            'phot_bp_mean_mag', 'phot_g_mean_mag', 'phot_rp_mean_mag',
            'tmass_j_m', 'tmass_h_m', 'tmass_ks_m',
            'gsc23_b_mag', 'gsc23_v_mag',
        ]

    def get_color_metric_pairs(self) -> List[Tuple[str, str]]:
        result = []
        # for mag_column_group in self.mag_column_groups:
        #     n = len(mag_column_group)
        #     for i in range(n):
        #         col1 = mag_column_group[i]
        #         for j in range(i + 1, n):
        #             col2 = mag_column_group[j]
        #             result.append((col1, col2,))
        n = len(self.mag_columns)
        for i in range(n):
            col1 = self.mag_columns[i]
            for j in range(i + 1, n):
                col2 = self.mag_columns[j]
                result.append((col1, col2,))
        return result

    def get_inputs(self, data_frame: pd.DataFrame):
        distance = 1000.0 / data_frame['parallax'].values
        lat_deg = data_frame['b'].values
        long_deg = data_frame['l'].values
        distance_t = torch.from_numpy(distance).float().unsqueeze(1)
        lat_rad_t = torch.from_numpy(np.deg2rad(lat_deg)).float().unsqueeze(1)
        long_rad_t = torch.from_numpy(np.deg2rad(long_deg)).float().unsqueeze(1)
        color_metric_pairs = self.get_color_metric_pairs()
        var_list = []
        for col1, col2 in color_metric_pairs:
            difference = data_frame[col1] - data_frame[col2]
            var_list.append(difference.values)
        color_metrics_t = torch.from_numpy(np.transpose(var_list)).float()
        return distance_t, color_metrics_t, lat_rad_t, long_rad_t,

    def get_label(self, data_frame: pd.DataFrame):
        target_mag = data_frame[self.target_column].values
        return torch.from_numpy(target_mag).float().unsqueeze(1)

    def train_fold(self, train_frame: pd.DataFrame, linear_params: Tuple[float, float, float]):
        color_metric_pairs = self.get_color_metric_pairs()
        num_color_metrics = len(color_metric_pairs)
        distance_t, color_metrics_t, lat_rad_t, long_rad_t = self.get_inputs(train_frame)
        target_mag_t = self.get_label(train_frame)
        est_d_param, est_log_d_param, est_bias = linear_params
        model = DistanceBasedMagnitudeModel(num_color_metrics, est_d_param, est_log_d_param, est_bias)
        model.train()
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        print('Training split...')
        for epoch in range(self.num_epochs):
            pred_mag = model(distance_t, color_metrics_t, lat_rad_t, long_rad_t)
            loss = loss_fn(pred_mag, target_mag_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: {loss_value:.3f}')
        return model

    def get_response_frame(self, data_frame: pd.DataFrame, model: DistanceBasedMagnitudeModel):
        distance_t, color_metrics_t, lat_rad_t, long_rad_t = self.get_inputs(data_frame)
        print(f'Color metrics mean: {torch.mean(color_metrics_t).item()}')
        print(f'Color metrics std: {torch.std(color_metrics_t).item()}')
        model.eval()
        pred_mag = model(distance_t, color_metrics_t, lat_rad_t, long_rad_t)
        pred_mag_np = pred_mag.squeeze(1).detach().numpy()
        return pd.DataFrame({
            self.id_column: data_frame[self.id_column].values,
            self.response_column: pred_mag_np,
        })

    def train_pass(self, pass_number: int, data_frame: pd.DataFrame, linear_params: Tuple[float, float, float]):
        print(f'Training pass {pass_number}...')
        shuffled_frame = data_frame.sample(frac=1)
        shuffled_frame.reset_index(inplace=True, drop=True)
        kf = KFold(n_splits=self.num_splits)
        valid_rf_list = []
        for train_idx, test_idx in kf.split(shuffled_frame):
            full_train_frame: pd.DataFrame = shuffled_frame.iloc[train_idx]
            num_training_items = min(len(full_train_frame), self.max_training_items)
            train_frame = full_train_frame.sample(n=num_training_items)
            valid_frame = shuffled_frame.iloc[test_idx]
            fold_model = self.train_fold(train_frame, linear_params)
            valid_rf = self.get_response_frame(valid_frame, fold_model)
            valid_rf_list.append(valid_rf)
        response_rf: pd.DataFrame = pd.concat(valid_rf_list, sort=False).reset_index(drop=True)
        response_rf.sort_values(self.id_column, inplace=True)
        return response_rf

    def train(self, data_frame: pd.DataFrame):
        linear_params = self.estimate_linear_params(data_frame)
        print(f'Estimated linear params: {linear_params}')
        response_sum = np.zeros((len(data_frame),))
        first_id = None
        for p in range(self.num_passes):
            print(f'Pass {p}...')
            r_frame = self.train_pass(p, data_frame, linear_params)
            if first_id is None:
                first_id = r_frame[self.id_column].values
            response = r_frame[self.response_column].values
            response_sum += response
        mean_response = response_sum / self.num_passes
        return pd.DataFrame({
            self.id_column: first_id,
            self.response_column: mean_response,
        })

    def estimate_linear_params(self, data_frame: pd.DataFrame):
        distance = 1000.0 / data_frame['parallax'].values
        log_distance = np.log(distance)
        var_data = np.transpose([
            distance, log_distance,
        ])
        label = data_frame[self.target_column].values
        fitter = LinearRegression()
        fitter.fit(var_data, label)
        return tuple(fitter.coef_) + (fitter.intercept_,)

    def get_error_model_vars(self, data_frame: pd.DataFrame):
        flux = data_frame['phot_g_mean_flux'].values
        flux_error = data_frame['phot_g_mean_flux_error'].values
        mag_error = (np.log(flux + flux_error) - np.log(flux)) ** 2
        parallax = data_frame['parallax'].values
        parallax_error = data_frame['parallax_error'].values
        log_distance_error = (np.log(1000.0 / parallax) - np.log(1000.0 / (parallax + parallax_error))) ** 2
        return np.transpose([
            mag_error,
            log_distance_error,
        ])

    def get_error_model_response(self, data_frame: pd.DataFrame, model, response_column: str):
        vars = self.get_error_model_vars(data_frame)
        pred_sq_errors = model.predict(vars)
        return pd.DataFrame({
            self.id_column: data_frame[self.id_column].values,
            response_column: pred_sq_errors,
        })

    def train_error_model_fold(self, train_frame: pd.DataFrame, error_column='mag_model_residual'):
        error_model_vars = self.get_error_model_vars(train_frame)
        error_model_label = train_frame[error_column].values ** 2
        fitter = LinearRegression()
        fitter.fit(error_model_vars, error_model_label)
        return fitter

    def train_error_model_pass(self, pass_number: int, data_frame: pd.DataFrame, response_column: str, num_splits=5):
        print(f'Training pass {pass_number}...')
        shuffled_frame = data_frame.sample(frac=1)
        shuffled_frame.reset_index(inplace=True, drop=True)
        kf = KFold(n_splits=num_splits)
        valid_rf_list = []
        for train_idx, test_idx in kf.split(shuffled_frame):
            train_frame: pd.DataFrame = shuffled_frame.iloc[train_idx]
            valid_frame = shuffled_frame.iloc[test_idx]
            fold_model = self.train_error_model_fold(train_frame)
            valid_rf = self.get_error_model_response(valid_frame, fold_model, response_column)
            valid_rf_list.append(valid_rf)
        response_rf: pd.DataFrame = pd.concat(valid_rf_list, sort=False).reset_index(drop=True)
        response_rf.sort_values(self.id_column, inplace=True)
        return response_rf

    def train_error_model(self, data_frame: pd.DataFrame, response_column='modeled_sq_error'):
        first_id = None
        response_sum = np.zeros((len(data_frame),))
        for p in range(self.num_passes):
            print(f'Pass {p}...')
            r_frame = self.train_error_model_pass(p, data_frame, response_column)
            if first_id is None:
                first_id = r_frame[self.id_column].values
            response = r_frame[response_column].values
            response_sum += response
        mean_response = response_sum / self.num_passes
        return pd.DataFrame({
            self.id_column: first_id,
            response_column: mean_response,
        })
