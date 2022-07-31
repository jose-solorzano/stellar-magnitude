import os

from config import CONFIG
from training.BandBasedMagnitudeModelTrainer import BandBasedMagnitudeModelTrainer
from training.DistanceBasedMagnitudeModelTrainer import DistanceBasedMagnitudeModelTrainer
import pandas as pd


def run(num_passes: int, num_splits: int, max_training_items, num_epochs: int, lr: float):
    rc = 'modeled_magnitude'
    trainer = BandBasedMagnitudeModelTrainer(num_passes, num_splits, max_training_items, num_epochs, lr,
                                             response_column=rc)
    in_file = CONFIG['in_file']
    out_dir = CONFIG['out_dir']
    print('Loading data...')
    data_frame = pd.read_csv(in_file)
    print(f'Loaded {len(data_frame)} rows.')
    source_id_set = set(data_frame['source_id'].values)
    print(f'Number of source_ids is {len(source_id_set)}.')
    data_frame.drop_duplicates('source_id', keep=False, inplace=True)
    data_frame.reset_index(drop=True, inplace=True)
    print(f'Kept {len(data_frame)} after removal of duplicates.')
    response_frame = trainer.train(data_frame)
    orig_frame = data_frame[['source_id', trainer.target_column]]
    merged_frame = pd.merge(orig_frame, response_frame, on='source_id', how='inner')
    merged_frame['mag_model_residual'] = merged_frame[trainer.target_column].values - merged_frame[rc].values
    out_file = os.path.join(out_dir, 'allwise-w3-model.csv')
    merged_frame.to_csv(out_file, index=False)
    print(f'Wrote {out_file}')


if __name__ == '__main__':
    params = {
        'num_passes': 3,
        'num_splits': 2,
        'max_training_items': 50000,
        'num_epochs': 401,
        'lr': 0.001,
    }
    run(**params)
