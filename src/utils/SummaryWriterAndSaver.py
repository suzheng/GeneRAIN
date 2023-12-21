import os
import csv
import json
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def float32_to_float(obj):
    if isinstance(obj, dict):
        return {key: float32_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [float32_to_float(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32):
        return float(obj)
    return obj



class SummaryWriterAndSaver(SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_file_name = self._get_file_writer().event_writer._file_name
        self.data_file_path = os.path.join(os.path.dirname(self.event_file_name), "Data." + os.path.basename(self.event_file_name) + ".data.csv")
        self.histogram_file_path = os.path.join(os.path.dirname(self.event_file_name), "Data." + os.path.basename(self.event_file_name) + ".histogram.json")
        self.header = ['event_file_path', 'method']
        self._create_data_file()

    def _create_data_file(self):
        if not os.path.exists(self.data_file_path):
            with open(self.data_file_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(self.header)

    def _save_data(self, method, **kwargs):
        self._update_header(kwargs)
        with open(self.data_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            row_data = [self._get_file_writer().event_writer._file_name, method] + [''] * (len(self.header) - 2)
            for key, value in kwargs.items():
                row_data[self.header.index(key)] = value
            csv_writer.writerow(row_data)

    def _save_histogram_data(self, data_dict):
        data_dict = float32_to_float(data_dict)
        with open(self.histogram_file_path, 'a') as json_file:
            json.dump(data_dict, json_file)
            json_file.write('\n')

    def _update_header(self, kwargs):
        updated_header = False
        for key in kwargs.keys():
            if key not in self.header:
                self.header.append(key)
                updated_header = True

        if updated_header:
            with open(self.data_file_path, 'w', newline='') as header_file:
                header_writer = csv.writer(header_file)
                header_writer.writerow(self.header)

    def add_scalar(self, *args, **kwargs):
        super().add_scalar(*args, **kwargs)
        data_dict = dict(zip(['tag', 'scalar_value', 'global_step'], args))
        self._save_data('add_scalar', **data_dict)

    def add_hparams(self, *args, **kwargs):
        
        hparam_dict, metric_dict = args[0], args[1]
        # Convert any list values in hparam_dict to strings
        hparam_dict = {k: (str(v).replace(',', '|') if isinstance(v, list) else v) for k, v in hparam_dict.items()}
        super().add_hparams(hparam_dict, metric_dict)
        combined_dict = {**hparam_dict, **metric_dict}
        self._save_data('add_hparams', **combined_dict)

    def add_histogram_raw(self, *args, **kwargs):
        super().add_histogram_raw(*args, **kwargs)
        data_dict = kwargs
        self._save_histogram_data(data_dict)
