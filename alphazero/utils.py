import yaml


def read_yaml(filename: str) -> dict:
    with open(filename, 'r') as stream:
        try:
            return yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
