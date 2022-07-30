import os

import numpy as np

from config import CONFIG
from training.MagnitudeModelTrainer import MagnitudeModelTrainer
import pandas as pd


def run(num_passes: int, num_splits: int, max_training_items, num_epochs: int, lr: float):
    rc = 'modeled_magnitude'
    trainer = MagnitudeModelTrainer(num_passes, num_splits, max_training_items, num_epochs, lr,
                                    response_column=rc)
    in_file = CONFIG['in_file']
    out_dir = CONFIG['out_dir']
    print('Loading data...')
    data_frame = pd.read_csv(in_file)
    print(f'Loaded {len(data_frame)} rows.')
    mag_model_file = os.path.join(out_dir, 'magnitude-model.csv')
    mag_model_frame = pd.read_csv(mag_model_file)
    merged_frame = pd.merge(data_frame, mag_model_frame, on='source_id', how='inner')
    response_frame = trainer.train_error_model(merged_frame)
    merged_frame = pd.merge(mag_model_frame, response_frame, on='source_id', how='inner')
    merged_frame['est_mag_error'] = np.sqrt(merged_frame['modeled_sq_error'])
    merged_frame['mag_anomaly'] = merged_frame['mag_model_residual'] / merged_frame['est_mag_error']
    merged_frame.drop(['modeled_sq_error'], axis=1, inplace=True)
    out_file = os.path.join(out_dir, 'magnitude-model-with-error.csv')
    merged_frame.to_csv(out_file, index=False)
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    params = {
        'num_passes': 3,
        'num_splits': 2,
        'max_training_items': 50000,
        'num_epochs': 2000,
        'lr': 0.003,
    }
    run(**params)

