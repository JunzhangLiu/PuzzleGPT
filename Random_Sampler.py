from __future__ import print_function, division

import argparse
import json
import os
import random
import time
import ast
import clip
import numpy as np
import requests
import torch
from MainReasoner import *
from PIL import Image
from lavis.models import load_model_and_preprocess
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from image_patch import *
from utils import show_single_image

with open('api.key') as f:
    api_key = f.read().strip()

transform_tsr = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize(size=(224, 224), max_size=None, antialias=None),
    transforms.ToTensor()
])
random.seed(138)
CONTINENTS = {
    'NA': 'North America',
    'SA': 'South America',
    'AS': 'Asia',
    'OC': 'Oceania',
    'AF': 'Africa',
    'EU': 'Europe'
}
import datetime, calendar

MONTHS = calendar.month_name
MONTHS2num = {month: index for index, month in enumerate(MONTHS) if month}
image_base_url = f'https://static01.nyt.com/'


def load_jsonl(file_name):
    print(f'loading file {file_name}')
    data = []
    with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f):
            data.append(json.loads(line.strip()))
    return data


def print_and_save_image(cropped_image, save_path: str):
    show_single_image(cropped_image, save_path=save_path)


def get_gold_data(data_folder):
    # train_path = data_folder + 'train.jsonl'
    test_path = data_folder + 'gold_test.jsonl'
    dev_path = data_folder + 'gold_dev.jsonl'
    print(f'loading data from {data_folder}')
    # trains = load_jsonl(train_path)
    tests = load_jsonl(test_path)
    devs = load_jsonl(dev_path)
    return devs, tests


def time_label_2natural(date):
    # '2021-5-21' to format 'May 21, 2020'
    date = date.split('-')
    if len(date) == 3:
        year, month, day = date
        return f'{MONTHS[int(month)]} {day}, {year}'
    if len(date) == 2:
        year, month = date
        return f'{MONTHS[int(month)]}, {year}'
    if len(date) == 1:
        year = date[0]
        return f'{year}'


def to_date(date):
    if date[-2:] == 'th':
        return date[:-2]
    elif date[-2:] == 'st':
        return date[:-2]
    elif date[-2:] == 'nd':
        return date[:-2]
    elif date[-2:] == 'rd':
        return date[:-2]
    else:
        return date


def to_Month(month):
    if month == 'january':
        return 'January'
    elif month == 'february':
        return 'February'
    elif month == 'march':
        return 'March'
    elif month == 'april':
        return 'April'
    elif month == 'may':
        return 'May'
    elif month == 'june':
        return 'June'
    elif month == 'july':
        return 'July'
    elif month == 'august':
        return 'August'
    elif month == 'september':
        return 'September'
    elif month == 'october':
        return 'October'
    elif month == 'november':
        return 'November'
    elif month == 'december':
        return 'December'
    else:
        return month


# month = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November',
#          'December']


def natural2time_label(date):
    # May 21, 2020 to format 2021-5-21
    mon = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
           'November', 'December']
    date = date.replace('a photo taken in ', '')
    date = date.split(', ')
    if len(date) == 1:
        if len(date[0].split()) != 1:
            return '2021-5-21'
        elif date[0].capitalize() in mon:
            return '2021-5-21'
        else:
            year = date[0]
            return f'{year}'
    elif len(date) == 2:
        monthday, year = date
        monthday = monthday.split()
        if len(monthday) == 1:
            month = monthday[0]
            return f'{year}-{MONTHS2num[to_Month(month)]}'
        elif monthday[1].capitalize() in mon:
            if len(monthday) == 2:
                day, month = monthday
                return f'{year}-{MONTHS2num[to_Month(month)]}-{int(to_date(day))}'
        elif monthday[0].capitalize() in mon:
            if len(monthday) == 2:
                month, day = monthday
                return f'{year}-{MONTHS2num[to_Month(month)]}-{int(to_date(day))}'
    else:
        return '2021-5-21'


def nyt_image_url_to_name(url):
    name = url.replace(image_base_url, '').replace('/', '-')
    if not name.endswith('.jpg'):
        name = name + '.jpg'
    return name


class VisionLangDataset(Dataset):
    """
    """

    def __init__(self, data, img_dir, transform=None):
        valid_data = [k for k in data]
        self.valid_time_data = [k for k in data if k['gold_time_suggest'] is not None]
        self.valid_loc_data = [k for k in data if k['gold_location_suggest'] is not None]

        self.valid_data = valid_data
        self.loc_data = valid_data
        self.img_dir = img_dir
        self.transform = transform
        self.time_label_name = 'gold_time_suggest'
        self.loc_label_name = 'gold_location_suggest'
        self.prepare_images()

    def __len__(self):
        return len(self.valid_data)

    def all_len(self):
        return len(self.valid_time_data), len(self.valid_loc_data)

    def __getitem__(self, idx):
        # print(self.time_data[idx]['image_url'], self.loc_data[idx]['image_url'])
        # assert self.valid_data[idx]['image_url'] == self.loc_data[idx]['image_url']
        url = self.valid_data[idx]['image_url']
        web_url = self.valid_data[idx]['web_url']
        img_path = os.path.join(self.img_dir, nyt_image_url_to_name(url))
        # download image if not exist
        if not os.path.exists(img_path):
            response = requests.get(url, stream=True)
            with open(img_path, 'wb') as handle:
                for block in response.iter_content(1024):
                    if not block:
                        break
                    handle.write(block)
        if not os.path.exists(img_path):
            print(f'Cannot download image {img_path}')
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        time_labels = self.valid_data[idx][self.time_label_name]
        loc_labels = self.valid_data[idx][self.loc_label_name]

        if time_labels is not None:
            time_labels = time_label_2natural(time_labels)
            time_labels_text = 'a photo taken in ' + time_labels
        else:
            time_labels = 'empty'
            time_labels_text = 'empty'
        if loc_labels is not None:
            loc_labels_text = 'a photo taken in ' + loc_labels
        else:
            loc_labels = 'empty'
            loc_labels_text = 'empty'

        return image, time_labels, time_labels_text, loc_labels, loc_labels_text, web_url, img_path

    def prepare_images(self):

        for idx in tqdm(range(len(self))):
            url = self.valid_data[idx]['image_url']
            img_path = os.path.join(self.img_dir, nyt_image_url_to_name(url))
            if not os.path.exists(img_path):
                response = requests.get(url, stream=True)
                with open(img_path, 'wb') as handle:
                    for block in response.iter_content(1024):
                        if not block:
                            break
                        handle.write(block)
            if not os.path.exists(img_path):
                print(f'Cannot download image {img_path}')


def get_accuracy(probs, ground_truth, N, type=''):
    _, top_labels1 = probs.topk(1, dim=-1)

    # print(f'Top1 label: {top_labels1}')
    # print(f'Ground Truth: {ground_truth}')
    # N = 400

    hit1 = len([1 for i in range(N) if ground_truth[i] in top_labels1[i]]) / N
    with open('results.txt', 'w') as f:
        f.write(f'Accuracy for {type} is {100 * hit1};')
        f.write('\n')
    print(f'{type} with {probs.shape[1]} unique labels, accuracy is {100 * hit1:.2f}')


def get_f1(probs, ground_truth, N, type, label_name, id2label=None):
    # N = 400

    _, top_labels1 = probs.topk(1, dim=-1)
    y_pred = [id2label[top_labels1[i]] for i in range(N)]
    y_true = ground_truth  # [id2label[ground_truth[i]] for i in range(N)]
    if 'loc' in label_name:
        hier_fun = get_hierarchical_geo_labels
    else:
        y_pred = [natural2time_label(i) for i in y_pred]
        y_true = [natural2time_label(i) for i in y_true]
        hier_fun = get_hierarchical_time_labels
    y_pred_hier = [hier_fun(i) for i in y_pred]
    y_true_hier = [hier_fun(i) for i in y_true]
    with open('results.txt', 'w') as f:
        f.write(f'F1 Score for {type} is {100 * example_f1(y_pred_hier, y_true_hier)};')
        f.write('\n')
    print(
        f'{type} with {probs.shape[1]} unique labels, Example F1 is {100 * example_f1(y_pred_hier, y_true_hier):.2f}\t')


def example_f1(y_pred_hier, y_true_hier, bre=True):
    f1 = 0
    N = len(y_pred_hier)
    if bre:
        # print(N, y_pred_hier[0], y_true_hier[0])
        for i in range(N):
            inter = set([k for k in y_pred_hier[i] if k in y_true_hier[i]])
            # w/o Brevity Penalty
            # f1 += 2 * len(inter) / (len(y_pred_hier[i]) + len(y_true_hier[i]))

            # w/ Brevity Penalty
            f1 += min(1, np.exp(1 - len(y_true_hier[i]) / len(y_pred_hier[i]))) \
                  * (2 * len(inter) / (len(y_pred_hier[i]) + len(y_true_hier[i])))
        f1 = f1 / N
        print('brevitized')
        return f1
    # print(N, y_pred_hier[0], y_true_hier[0])
    else:
        for i in range(N):
            inter = set([k for k in y_pred_hier[i] if k in y_true_hier[i]])
            f1 += 2 * len(inter) / (len(y_pred_hier[i]) + len(y_true_hier[i]))
        f1 = f1 / N
        return f1


def get_decade(year):
    return f'{str(year)[:-1]}0s'


def get_century(year):
    # If year is between 1 to 100 it will come in 1st century
    if year <= 100:
        return "1st century"
    elif year % 100 == 0:
        return f'{year // 100} century'
    else:
        return f'{year // 100 + 1} century'


def get_hierarchical_time_labels(date):
    # print(date)
    if date is None:
        return ['21 century']
    if 'century' in date:
        return [date]
    if date[-1] == 's':
        return [f'{get_century(int(date[:-1]))}', date]
    date = date.split('-')
    date = [MONTHS2num[k] if k in MONTHS2num else k for k in date]
    date = list(map(int, date))
    labels = None
    if len(date) == 3:
        year, month, day = date
        labels = [f'{get_century(year)}', f'{get_decade(year)}', f'{year}', f'{year}-{month}', f'{year}-{month}-{day}']
    if len(date) == 2:
        year, month = date
        labels = [f'{get_century(year)}', f'{get_decade(year)}', f'{year}', f'{year}-{month}']
    if len(date) == 1:
        year = date[0]
        labels = [f'{get_century(year)}', f'{get_decade(year)}', f'{year}']
    return labels


def get_hierarchical_geo_labels(loc):
    locs = loc.split(', ')

    if len(locs) > 3:
        return [', '.join(locs[:-2]).capitalize(), locs[-2].capitalize(), locs[-1].capitalize()]
    else:
        return locs


def run():
    input_data_folder = f'/home/sssak/TARA/data/nyt_preprocessed/input/'
    input_image_folder = f'/home/sssak/TARA/data/nyt_preprocessed/images/'
    print(f'start baseline evaluation: reasoning with clip on nyt')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    devs, tests = get_gold_data(data_folder=input_data_folder)

    clip_model, clip_preprocess = clip.load('ViT-B/32', device=device)
    clip_model.eval()

    train_label = {'gold_location_suggest': 'location', 'gold_time_suggest': 'date', 'location': 'location',
                   'gold_time': 'date', 'date': 'date'}

    # define a transformation
    dataname = 'test'
    if dataname == 'test':
        test_dataset = VisionLangDataset(tests, input_image_folder, transform=transform_tsr)
    else:
        test_dataset = VisionLangDataset(devs, input_image_folder, transform=transform_tsr)
    print(f'This is on {dataname}')

    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)
    N = len(test_dataset)
    N_time, N_loc = test_dataset.all_len()
    print(f'length: Time: {N_time}, Location: {N_loc}')

    filename_loc = '/home/sssak/inter_results/0225/1010.txt'
    filename_date = '/home/sssak/inter_results/0225/1010_time.txt'
    label_loc_path = '/home/sssak/inter_results/0225/label_loc.txt'
    label_time_path = '/home/sssak/inter_results/0225/label_time.txt'

    # filename_loc = '/home/sssak/1126/test_blip/1010.txt'
    # filename_date = '/home/sssak/1126/test_blip/1010_time.txt'
    # label_loc_path = '/home/sssak/1126/test_blip/label_loc.txt'
    # label_time_path = '/home/sssak/1126/test_blip/label_time.txt'
    print(filename_loc)

    with open(filename_loc, 'r') as f:
        l = f.readlines()
        f.close()
    current_idx = int(l[0])
    loc_predicates = ast.literal_eval(l[1])
    if loc_predicates is None:
        loc_predicates = []

    with open(filename_date, 'r') as f:
        l = f.readlines()
        f.close()
    time_predicates = ast.literal_eval(l[0])
    if time_predicates is None:
        time_predicates = []

    with open(label_loc_path, 'r') as f:
        l = f.readlines()
        f.close()
    label_loc_10 = ast.literal_eval(l[0])
    if label_loc_10 is None:
        label_loc_10 = []

    with open(label_time_path, 'r') as f:
        l = f.readlines()
        f.close()
    label_time_10 = ast.literal_eval(l[0])
    if label_time_10 is None:
        label_time_10 = []

    cnt = current_idx + 1

    with torch.no_grad():
        for images, time_labels, time_labels_text, loc_labels, loc_labels_text, web_url, img_path in test_dataloader:
            # Images are batched

            for idx in range(current_idx + 1, images.shape[0]):
                print(f'-----------cnt:{cnt}----------')
                print(f'-----------label: {loc_labels_text[idx]}-----------------')
                print(f'-----------label: {time_labels_text[idx]}-----------------')
                # first try 5
                # # =============
                if idx >= 300:
                    break
                # # =============
                # end of first try five
                patch = ImagePatch(images[idx, :, :, :].squeeze(0))
                # patch.print_image()
                date, loc = Reasoner(patch, img_path, web_url, cnt, api_key, loc_labels[idx])._reasoner()

                date = date.replace('\n', '')
                loc = loc.replace('\n', '')

                if time_labels_text[idx] != 'empty' or time_labels[idx] != 'empty':
                    time_predicates.append(date)
                    label_time_10.append(time_labels_text[idx])
                    with open(filename_date, 'w') as f:
                        f.write(str(time_predicates))
                        f.close()
                    with open(label_time_path, 'w') as f:
                        f.write(str(label_time_10))
                        f.close()
                    # label_time_10.append(time_labels_text[idx])
                if loc_labels_text[idx] != 'empty' or loc_labels[idx] != 'empty':
                    loc_predicates.append(loc)
                    # label_time_10.append(time_labels_text[idx])
                    label_loc_10.append(loc_labels_text[idx])
                    with open(filename_loc, 'w') as f:
                        f.write(f'{cnt}\n')
                        f.write(str(loc_predicates))
                        f.close()
                    with open(label_loc_path, 'w') as f:
                        f.write(str(label_loc_10))
                        f.close()
                cnt += 1

                # Wait for the GPT to reset the currency.
                # time.sleep(5)

            id2time_labels = list(set(time_labels_text))
            id2loc_labels = list(set(loc_labels_text))
            # first try 5
            # ============

            with open(filename_loc, 'r') as f:
                l = f.readlines()
                f.close()
            loc_predicates = ast.literal_eval(l[1])

            with open(filename_date, 'r') as f:
                l = f.readlines()
                f.close()
            time_predicates = ast.literal_eval(l[0])

            with open(label_loc_path, 'r') as f:
                l = f.readlines()
                f.close()
            label_loc_10 = ast.literal_eval(l[0])

            with open(label_time_path, 'r') as f:
                l = f.readlines()
                f.close()
            label_time_10 = ast.literal_eval(l[0])

            id2time_labels = label_time_10
            id2loc_labels = label_loc_10
            time_labels_text = time_labels_text[:N_time]
            loc_labels_text = loc_labels_text[:N_loc]
            time_labels_text = label_time_10
            loc_labels_text = label_loc_10
            # ============
            # end of first try five
            time_labels2id = {k: i for i, k in enumerate(id2time_labels)}
            loc_labels2id = {k: i for i, k in enumerate(id2loc_labels)}

        print(time_predicates)
        print(loc_predicates)
        time_preds_tokens = clip.tokenize(time_predicates, truncate=True).to(device)
        time_labels_tokens = clip.tokenize(time_labels_text, truncate=True).to(device)
        loc_preds_tokens = clip.tokenize(loc_predicates, truncate=True).to(device)
        loc_labels_tokens = clip.tokenize(loc_labels_text, truncate=True).to(device)

        time_preds_features = clip_model.encode_text(time_preds_tokens).float()
        time_labels_features = clip_model.encode_text(time_labels_tokens).float()
        loc_preds_features = clip_model.encode_text(loc_preds_tokens).float()
        loc_labels_features = clip_model.encode_text(loc_labels_tokens).float()
        print(f'clip output shape{time_labels_features.shape}')

        time_labels_features /= time_labels_features.norm(dim=-1, keepdim=True)
        time_preds_features /= time_preds_features.norm(dim=-1, keepdim=True)
        loc_labels_features /= loc_labels_features.norm(dim=-1, keepdim=True)
        loc_preds_features /= loc_preds_features.norm(dim=-1, keepdim=True)

        time_labels_logits = (100.0 * time_preds_features @ time_labels_features.T)
        loc_labels_logits = (100.0 * loc_preds_features @ loc_labels_features.T)

        time_labels_probs = time_labels_logits.softmax(dim=-1).cpu()
        loc_labels_probs = loc_labels_logits.softmax(dim=-1).cpu()

        # GT labels for time and loc
        ground_truth_time_labels = [time_labels2id[k] for k in time_labels_text]
        ground_truth_loc_labels = [loc_labels2id[k] for k in loc_labels_text]

        # time accuracy and f1
        N_time = len(time_predicates)
        if N_time != 0:
            get_accuracy(time_labels_probs, ground_truth_time_labels, N_time, 'Clip Time Reasoning with labels,')
            get_f1(time_labels_probs, time_labels_text, N_time, 'Clip Time Reasoning with labels,', 'time F1',
                   id2time_labels)

        # loc accuracy and f1
        N_loc = len(loc_predicates)
        if N_loc != 0:
            get_accuracy(loc_labels_probs, ground_truth_loc_labels, N_loc, 'Clip Loc Reasoning with labels,')
            get_f1(loc_labels_probs, loc_labels_text, N_loc, 'Clip Loc Reasoning with labels,', 'location F1',
                   id2loc_labels)

        print(f'Location Preds: {loc_predicates}, Location Labels: {label_loc_10}')
        print(f'Time Preds: {time_predicates}, Time Labels: {label_time_10}')

    print('done')


# main running part
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset_name', type=str, default='nyt', help='dataset folder name')
    parser.add_argument('-clip', '--clip_model_name', type=str, default='ViT-B/32', help='model name or path')
    parser.add_argument('-lb', '--label_name', type=str, default='gold_time_suggest', help='')
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args)
    run()
