import json

class ParamFinder:
    def __init__(self, json_file):
        self.json_file = json_file
        with open(self.json_file, 'r') as f:
            self.input_parameters = json.load(f)

    def find(self, key, default_value):
        if key not in self.input_parameters:
            self.input_parameters[key] = default_value
        return self.input_parameters.get(key)

    def save_updated_parameters(self, output_file):
        with open(output_file, 'w') as f:
            json.dump(self.input_parameters, f, indent=2)