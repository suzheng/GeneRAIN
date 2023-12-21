import json
import os

# key in json file has to end with '_path' in all the relative paths, then config_loader.py will append project_path to it
class Config:
    # def __init__(self, config_path=config_file_path, proj_folder_in_the_obj=proj_folder):
    def __init__(self, config_file_path=None):
        script_dir_path = os.path.dirname(os.path.abspath(__file__))
        parent_dir_path = os.path.dirname(script_dir_path)
        proj_folder = os.path.dirname(parent_dir_path)
        if os.environ.get("SPECIES") == "Mouse":
            proj_folder = os.path.abspath(proj_folder + "/../GeneRAIN_Mouse")
            print("Using mouse project folder: " + proj_folder)
        proj_folder = proj_folder + "/"
        if config_file_path == None:
            config_file_path = proj_folder + "/config.json"
        self.project_path = proj_folder
        self.proj_path = proj_folder
        with open(config_file_path, "r") as f:
            self.config = json.load(f)
            for key, value in self.config.items():
                if key.endswith("_path"):
                    value = os.path.join(self.proj_path, value)
                setattr(self, key, value)

    def get(self, *args):
        value = self.config
        for key in args:
            value = value[key]
            if key.endswith("_path"):
                value = os.path.join(self.proj_path, value)
        return value

