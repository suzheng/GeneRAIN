import json
import os

class ParamFinder:
    def __init__(self, json_file):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            self.input_parameters = json.load(f)

    def find(self, key, default_value):
        if key not in self.input_parameters:
            self.input_parameters[key] = default_value
            print(f"Parameter of {key} not found in the input json file, use default value of {default_value}!")
        return self.input_parameters.get(key)

    def save_updated_parameters(self, output_file):
            # Get the directory of the checkpoint file
        dir_path = os.path.dirname(output_file)
        # Create the directory if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(self.input_parameters, f, indent=2)
    def get_param_dict(self):
        return self.input_parameters