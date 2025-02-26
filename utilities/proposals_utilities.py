import json
import subprocess


def load_json(filename):
    with open(filename, 'r') as handle:
        data_json = json.load(handle)
    return data_json


def save_json(filename, entity):
    with open(filename, 'w') as handle:
        json.dump(entity, handle, indent=4)


def with_ffprobe(filename):
    duration, frames = 0, 0
    result = subprocess.check_output(f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"', shell=True).decode()
    fields = json.loads(result)['streams'][0]
    duration = fields['duration']
    return duration
