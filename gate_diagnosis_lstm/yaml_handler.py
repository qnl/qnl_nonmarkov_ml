import os
import yaml


class YamlHandler:
    def __init__(self):
        self.yaml_file = None

    def load_settings(self, yaml_path, relative=False):

        if relative:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            self.yaml_file = os.path.join(dir_path, yaml_path)
        else:
            self.yaml_file = yaml_path

        with open(self.yaml_file) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data
