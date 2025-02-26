from main_simple_lib import *
from image_patch import *
import openai
import requests
import json
import clip
import base64
import requests
from random import sample
from torchvision import transforms
import re
import time
import urllib.request
import io
from socket import timeout
import country_converter as coco
from functools import partial
from geopy.geocoders import Nominatim
import pycountry_convert as pc
import collections
from collections import defaultdict
import geopy
import os

geopy.geocoders.options.default_user_agent = "616448752@qq.com"
geolocator = Nominatim(timeout=10)
geocode = partial(geolocator.geocode, language="en")

with open('api.key') as f:
    openai.api_key = f.read().strip()
    api_key = f.read().strip()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def file_finder(path='/', head=''):
    files = os.listdir(path)
    for file in files:
        if file.endswith('.png') and head + '++' in file:
            return path + '/' + file


month2mm = {
    'january': '01',
    'february': '02',
    'march': '03',
    'april': '04',
    'may': '05',
    'june': '06',
    'july': '07',
    'august': '08',
    'september': '09',
    'october': '10',
    'november': '11',
    'december': '12'
}

transform_url = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize(size=(224, 224), max_size=None, antialias=None),
    transforms.ToTensor()
])


def open_image(url):
    try:
        file = io.BytesIO(urllib.request.urlopen(url, timeout=20).read())
        img = Image.open(file).convert("RGB")
        img = transform_url(img)
    except:
        img = None
        print('No image')
    return img


# continent converter
def country_to_continent(country_name):
    try:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name, cn_name_format="lower")
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    except:
        print(country_name)
        country_continent_name = None
    return country_continent_name


def detailed_name(loc):
    flag = True
    count = 0
    loc_out = None
    while flag and count <= 3:
        try:
            count += 1
            loc_out = geocode(loc)[0]
            flag = False
        except Exception as e:
            time.sleep(1)
            continue

    return loc_out


def brief_name(loc):
    names = [loc]
    loc = coco.convert(names=names, to='name_short')
    return loc


def get_continent(loc):
    brief_n = brief_name(loc)
    continent = country_to_continent(brief_n)
    return continent


def rec_txt(directory, to_store):
    with open(directory, 'a') as f:
        f.write(to_store)
        f.write('\n')
        f.close()


class Reasoner:
    """
    Reason the location
    """

    def __init__(self, image: ImagePatch, img_path: str, web_url: str=None, cnt=0, openai_key='', label='',
                 text='', time_clue='',model_and_tokenizer = None):

        """
        image : ImagePatch, the image_patch instance from which we want to reason the time and location
        web_url: str, the exact url of this image. We want to avoid retrieving this url
        """
        self.label = label
        self.text = text
        self.time_clue = time_clue.replace('\n', '')

        self.confidence_check = []
        self.confidence_target = []

        self.openai_key = openai_key
        self.img_path = img_path
        self.save_img_path = img_path.replace('/', '[')
        if model_and_tokenizer is not None:
            self.model,self.tokenzier = model_and_tokenizer
        else:
            self.model,self.tokenzier = None,None
        # self.pics_dir = f'/home/sssak/inter_results/0402/pics/'
        # self.txt_dir = '/home/sssak/inter_results/0215/text.txt'
        # self.path_dir = '/home/sssak/inter_results/0315/text.txt'
        self.cnt = cnt
        self.web_url = web_url
        self.keywords_threshold = {}
        self.time_threshold = {}
        self.time_range = {}
        self.keywords_prompt = ''
        self.main_patch = image

        self.image_summary = self.main_patch.simple_query('What are the words shown in this image？'
                                                          'Short Answer:')
        event = self.main_patch.simple_query('What is the event showing on this image? Short Answer:')
        self.event = self.gpt_query(f'Can you conclude this text in at most 3 words and removing non-specific '
                                    f'details:\n"{event}"')

        self.initial_guess = self.main_patch.simple_query('What are the 3 most possible cities where this image '
                                                          'was most likely taken in? Separate the answers with commas.')
        print(self.initial_guess)
        self.initial_guess = self.initial_guess.split(', ')
        self.initial_guess = list(set(self.initial_guess))

        self.year_threshold = collections.defaultdict(lambda: 0.0)
        self.month_threshold = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))
        self.date_threshold = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))

        self.continent_threshold = collections.defaultdict(lambda: 0.0)
        self.country_threshold = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))
        self.city_threshold = collections.defaultdict(lambda: collections.defaultdict(lambda: 0.0))

        # """
        # initialize
        self.continent_threshold['continent'] = 0.0
        self.country_threshold['continent']['country'] = 0.0
        self.city_threshold['country']['city'] = 0.0

        self.year_threshold['year'] = 0.0
        self.month_threshold['year']['month'] = 0.0
        self.date_threshold['month']['day'] = 0.0
        # print('Initially Guessed Location: ', self.initial_guess)

        self.entities = {'event': self.image_summary}
        self.api_key = "AIzaSyDXBf2XjY9-20lmNq6S92P3DNkiN2aiA8Y"
        self.cx = "e311a3a6804544e3f"

        # keywords: list of unique keywords; c_keywords is a str containing all keywords, linked with ','
        self.keywords = self.main_patch.simple_query('Keywords of this image? Answer with the keywords only and '
                                                     'separate with commas. Short Answer:').split(', ')
        supplimentary_keywords = self.main_patch.simple_query('What are the keywords of this image? Answer with the '
                                                              'keywords only and separate with commas. '
                                                              'Short Answer:').split(', ')

        for sk in supplimentary_keywords:
            if sk not in self.keywords:
                self.keywords.append(sk)

        self.loc_candidates = {}

        # Decade guess
        date_candidate = self.main_patch.simple_query('In what date was this image taken?').replace('on ', '')
        self.date_can = self.gpt_query(f'Convert the date {date_candidate} to yyyy-mm-dd. Answer the corresponding'
                                       f' yyyy-mm-dd only. Short Answer:')

        # we get a decade
        self.decade = self.get_decade(self.date_can)
        # this is a default time range(n=1)
        try:
            self.n_timerange = 1

            self.years = self.decade2interval(self.decade)
            for y in self.years:
                self.update_time_hie(y)
        except:
            pass
        self.update_time_hie_part(self.date_can)

        # 0401
        for guess_candidate in self.initial_guess:
            if self.check_valid(guess_candidate) and self.check_confident(guess_candidate):
                guess_candidate = self.loc_complete(guess_candidate)
                loc_result = guess_candidate.replace('\n', '')
                self.update_loc_hie(loc_result, n=1)

        self.retrieval = {}
        self.retr_res = {}

        # results
        self.date_results = {}
        self.loc_results = {}


    def get_entity(self, threshold=3):

        # initial definition
        self.entities['celebrities'] = []
        self.entities['keywords'] = []
        self.entities['text'] = []
        self.loc_candidates['celebrities'] = []
        self.loc_candidates['keywords'] = []
        self.loc_candidates['text'] = []

        # keywords
        useful_keywords = []
        reliable_loc = self.initial_guess
        Pass = False
        keyword_clues = []
        keywords_threshold = {f'{self.initial_guess.lower()}': 1}
        for k in self.keywords:
            # print(f'Keywords: {k}')
            key_patch = self.main_patch.find(k)

            if len(key_patch) == 0:
                continue

            for patch in key_patch:
                if self.str_to_bool(patch.simple_query('Does this reflect any location?')):
                    entity = patch.simple_query(f'What is this {k}?')
                    if k not in useful_keywords:
                        useful_keywords.append(k)
                    # step reasoner
                    current_candidate = ' or '.join([*set(keywords_threshold.keys())])
                    if self.str_to_bool(patch.simple_query(f'Are you confident that this is taken '
                                                           f'in {current_candidate}?')):
                        loc = patch.best_text_match(['a photo taken in ' + k for k in
                                                     [*set(keywords_threshold.keys())]]).lower()
                        keywords_threshold[loc.replace('a photo taken in ', '')] += 1
                    else:
                        loc = patch.simple_query(f'Where is this {entity}?').lower()
                        keywords_threshold[loc] = 1
                    clue = f'A {entity} that might indicate {loc}'
                    keyword_clues.append(clue)

                if max(keywords_threshold.values()) >= threshold:
                    reliable_loc = max(keywords_threshold, key=keywords_threshold.get)
                    Pass = True
                    break

            if Pass:
                break

        if not Pass:
            keywords_prompt = f'Given that an event includes {" and ".join(keyword_clues)}, guess where did ' \
                              f'this event take place? Answer with the location only:'
            reliable_loc = self.gpt_query(keywords_prompt)

        self.prompt_items['keywords'] = keyword_clues

        # --------------------------this is for google---------------------------#
        self.entities['keywords'] = [*set(useful_keywords)]
        # keywords_continent = self.gpt_query(f'Given that {keys}, what is the most likely continent is this image?')
        self.loc_candidates['keywords'] = reliable_loc
        # ---------------------------end of keywords------------------------------- #

        # celebrities
        # celebrities does not necessarily need the corresponding location. I am pushing this to the prior knowledge
        # base of BLiP2 and GPT
        if self.str_to_bool(self.main_patch.simple_query('Are there any celebrity or famous faces?')):
            faces = self.main_patch.find('celebrity faces')
            faces_candidate = []
            for face in faces:
                face_name = face.simple_query('Who is this person?')
                face_name = ' '.join([*set(face_name.split(' '))])  # to remove cases having repeat last first names
                if face_name == 'person' or face_name in faces_candidate:  # remove faces that are not important
                    continue
                faces_candidate.append(face_name)

            self.entities['celebrities'] = [*set(faces_candidate)]
        # ---------------------------end of celebrity------------------------------- #

        # text
        # text supports does not necessarily requires corresponding locations. I am pushing this to the prior knowledge
        # base of BLiP and GPT
        if self.str_to_bool(self.main_patch.simple_query('Are there any text?')):
            contexts = []
            languages = []
            # lang_locs = []
            main_text = self.main_patch.read_text()
            for txt in main_text:
                contexts.append(txt)

            lan = self.main_patch.simple_query('What language is shown on this image?')
            languages.append(lan)
            print(f'Show the OCRed text: {",".join(contexts)}')
            cleaned_txt = self.gpt_query(f'Can you give me the useful words form "{",".join(contexts)}"?'
                                         f'Please answer with the filtered words only:')
            langs = ','.join([*set(languages)])  # a string contains all languages
            lan_continents = self.gpt_query(f'In which continent is(are) {langs} used? Answer with North America '
                                            f'or South America or Asia or Oceania or Africa or Europe: ')

            self.entities['text'] = cleaned_txt  # text will not be used in loc reasoning?
            self.entities['languages'] = langs
            self.loc_candidates['languages'] = lan_continents
        # ---------------------------end of celebrity------------------------------- #

        for k in ['keywords', 'languages', 'text', 'celebrities']:
            if k in self.entities.keys():
                print(f'{k} after initialization: {self.entities[k]}')
            else:
                print(f'There is no {k} in this image.')

    def get_century(self, date):
        return None

    def get_decade(self, date):
        year = date.split('-')[0]
        return f'{year[:3]}0s'

    def decade2interval(self, decade):
        start_year = int(decade[:4])
        possible_years = [str(i) for i in range(start_year, start_year + 10)]
        return possible_years

    def range2interval(self, time_range):
        try:
            start, end = time_range.split('-')
            possible_years = [str(i) for i in range(int(start), int(end) + 1)]
            return possible_years
        except:
            return None

    def check_valid(self, loc):

        return self.str_to_bool(
            self.gpt_query(f'Honestly speaking, do you think "{loc}" is a valid city or country or continent? Answer '
                           f'with yes or no only. Short Answer:'))

    def get_img(self):
        return self.main_patch

    def loc_complete(self, loc):
        # print(f'In name: {loc}')
        print(f'input Location: {loc}')
        loc = loc.split('in ')[-1]
        # loc.replace('in the','')
        # loc.replace('in ','')

        if len(loc.split(', ')) >= 3:
            return loc
        try:
            if 'turkey' in loc.split(', ')[-1].lower():
                return ', '.join([loc, 'Asia'])
            elif 'turkey' in loc.split(', ')[-2].lower():
                return loc
        except:
            pass
        # this loc_ is a detailed location
        loc_ = detailed_name(loc)
        # print(f'detailed name result: {loc_}')
        if type(loc_) == list:
            loc_ = loc_[0]
        elif loc_ is None:
            loc_ = loc
        # we want to have the country
        try:
            country = loc_.split(', ')[-1]
            continent = country_to_continent(country)
        except:
            country = loc_.split(', ')[-2:]
            country = ', '.join(country)
            country = brief_name(country)
            continent = country_to_continent(country)
        if continent is None:
            continent = self.gpt_query(f'What is the continent that {country} is located in? '
                                       f'Answer with the continent only and avoid any prompts. Short Answer:')
            continent.replace('.', '')
        # print(f'continent result from country to continent: {continent}')
        # print(f'Loc: {loc}')
        long_string = ', '.join([loc_, continent])
        print(f'Long String from loc completion: {long_string}')
        # long_list = long_string.split(', ')
        # long_list = list(set(long_list))

        # return ', '.join(long_list)
        return long_string

    def update_loc_hie(self, guess_candidate, n=1):
        guess_candidate = guess_candidate.replace('\n', '')
        guess_candidate = guess_candidate.lower()
        if len(guess_candidate.split(', ')) == 1:
            continent = guess_candidate
            self.continent_threshold[continent] += n
            print(f'Updated location:{continent}')
        else:
            *city, country, continent = guess_candidate.split(', ')
            city_ = ', '.join(city)
            self.continent_threshold[continent] += n
            self.country_threshold[continent][', '.join([country, continent])] += n
            if len(city) > 0:
                self.city_threshold[', '.join([country, continent])][', '.join([city_, country, continent])] += n
            else:
                self.city_threshold[', '.join([country, continent])][', '.join([country, continent])] += n
            print(f'Updated location:{city_}, {country}, {continent}')

    def update_time_hie(self, date):
        if self.str_to_bool(self.gpt_query(f'Is "{date}" a valid date? Answer with yes or no. Short Answer:')):
            date = date.replace('\n', '')
            if len(date.split('-')) == 3:
                year, month, day = date.split('-')
                print(f'Date: {day}, Month: {month}, year: {year}')
                self.year_threshold[year] += 1
                self.month_threshold[year]['-'.join([year, month])] += 1
                self.date_threshold['-'.join([year, month])]['-'.join([year, month, day])] += 1

            elif len(date.split('-')) == 2:

                year, month = date.split('-')
                day = self.main_patch.simple_query(f'In which date within {date} was this image most possibly taken? '
                                                   f'Answer the specific day in 2-digit. Short Answer:')
                try:
                    an_int = int(day[-2:])
                    day = day[-2:]
                except:
                    day = '01'
                # self.date = {year: date_candidate}
                self.year_threshold[year] += 1
                self.month_threshold[year]['-'.join([year, month])] += 1
                self.date_threshold[month]['-'.join([year, month, day])] += 1

            elif len(date.split('-')) == 1:
                year = date
                self.year_threshold[year] += 1
        else:
            pass

    def update_time_hie_part(self, date, n=1):
        if self.str_to_bool(self.gpt_query(f'Is "{date}" a valid date? Answer with yes or no. Short Answer:')):
            date = date.replace('\n', '')
            if len(date.split('-')) == 3:
                year, month, day = date.split('-')
                print(f'Date: {day}, Month: {month}, year: {year}')
                self.year_threshold[year] += n
                self.month_threshold[year]['-'.join([year, month])] += n
                self.date_threshold['-'.join([year, month])]['-'.join([year, month, day])] += n

            elif len(date.split('-')) == 2:

                year, month = date.split('-')
                day = self.main_patch.simple_query(f'In which date within {date} was this image most possibly taken? '
                                                   f'Answer the specific day in 2-digit. Short Answer:')
                try:
                    an_int = int(day[-2:])
                    day = day[-2:]
                except:
                    day = '01'
                # self.date = {year: date_candidate}
                self.year_threshold[year] += n
                self.month_threshold[year]['-'.join([year, month])] += n
                self.date_threshold[month]['-'.join([year, month, day])] += n

            elif len(date.split('-')) == 1:
                year = date
                self.year_threshold[year] += n
        else:
            pass

    def gpt_query(self, message, temperature=.8, max_tokens=128, model='gpt-3.5-turbo'):
        flag = True
        messages = [
            {"role": "user", "content":message}
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to('cuda')
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=256,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=0.6,
                top_p=0.9,
            )
        response = output[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        return answer
        # if model == 'gpt-3.5-turbo':
        #     messages = [{"role": "user", "content": message}]
        #     while flag:
        #         try:
        #             response = openai.ChatCompletion.create(
        #                 model=model,
        #                 messages=messages,
        #                 max_tokens=max_tokens,
        #                 temperature=temperature
        #             )
        #             flag = False
        #         except openai.error.OpenAIError as e:
        #             time.sleep(5)
        #             continue

        #     return response['choices'][0]['message']['content'].replace('/n/n', '')
        # elif model == 'text-davinci-003':
        #     while flag:
        #         try:
        #             response = openai.Completion.create(
        #                 model=model,
        #                 prompt=message,
        #                 max_tokens=max_tokens,
        #                 temperature=temperature
        #             )
        #             flag = False
        #         except openai.error.OpenAIError as e:
        #             time.sleep(5)
        #             continue

        #     return response['choices'][0]['text'].replace('/n/n', '')

    def check_included(self, loc1, loc2):

        return self.gpt_query(f'Is {loc1} included in {loc2} or {loc2} included in {loc1}? Answer with yes or no:')

    def check_confident(self, loc, skip=False):
        loc = loc.split(',')
        if len(loc) == 1:
            loc = ''.join(loc)
        else:
            loc = ', '.join(loc[-2:])
        if not skip:
            flag = self.str_to_bool(self.main_patch.simple_query(f'Are you confident this image was taken in {loc}? '
                                                                 f'Short Answer:'))
            print(f'Flag for loc {loc}: {flag}')
            return flag
        else:
            return True

        # return True

    def list2str(self, list_sample):
        return ', '.join(itr for itr in list_sample)

    def str_to_bool(self, string: str):
        return False if 'no' in string.lower() else True

    def _check_threshold(self, threshold):
        continent = max(self.continent_threshold, key=self.continent_threshold.get)
        country = max(self.country_threshold[continent], key=self.country_threshold[continent].get)
        return max(self.city_threshold[country].values()) >= threshold


    def web_retrieve_pic(self, query, gt=None, limit=3, frac=0):
        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cx}&q={query}"
        candidates = {}
        response = requests.get(url)
        data = json.loads(response.text)
        date_info = None
        snippet_list = []
        link2img = None
        date_list = []
        cnt = 0
        if 'items' not in data.keys():
            return None, None

        print(f'Retrieved length: {len(data["items"])}')

        for i in data['items']:
            cnt += 1
            try:
                if gt is not None and i['link'] == gt:
                    continue
                else:
                    date_key = None
                    pagemap = i['pagemap']['metatags']
                    # pagemap is a list of dict, need to go through it to check keys
                    flag = False
                    for I in pagemap:
                        for key in I.keys():
                            if 'og:image' in key:
                                flag_key = key
                                flag = True
                                # for the retrievals with images, get date clues.
                                for date in I.keys():
                                    if 'time' in date or 'date' in date:
                                        date_key = date
                                        break
                                    else:
                                        date_key = None

                                if date_key is None:
                                    date_info = i['snippet']
                                else:
                                    date_info = I[date_key]

                                link2img = I[flag_key]
                                break

                        if flag:
                            break

                if link2img is not None:
                    snippet = i['snippet']
                    try:
                        img = open_image(link2img)
                        img_p = ImagePatch(img)

                        if img is not None:
                            # -------
                            if 0 < frac < 1:
                                score_image = self.main_patch.best_image_match(img)
                                score_image = score_image.item()
                                score_text = self.main_patch.score(snippet)
                                score_text = score_text.item()
                                score = score_image + score_text
                            elif frac >= 1:
                                score = self.main_patch.score(snippet).item()
                            else:
                                score_image = self.main_patch.best_image_match(img)
                                score_image = score_image.item()
                                score = score_image
                            if score > 95:

                                if score not in candidates.keys() and date_info is not None:
                                    candidates['--da-sh--'.join([snippet, date_info])] = score
                                elif score not in candidates.keys():
                                    candidates['--da-sh--'.join([snippet, snippet])] = score
                            # print('Image Saved')
                            # img_p.print_and_save_image(save_path=(self.pics_dir + f'{score}_{query}.png'))


                        else:
                            continue

                    except:
                        # score = 0
                        # if date_info is not None:
                        #     candidates['--da-sh--'.join([snippet, date_info])] = score
                        # else:
                        #     candidates['--da-sh--'.join([snippet, snippet])] = score
                        pass

                # else:
                # snippet = i['snippet']
                # score = 0
                # if date_info is not None:
                #     candidates['--da-sh--'.join([snippet, date_info])] = score
                # else:
                #     candidates['--da-sh--'.join([snippet, snippet])] = score


            except KeyError as e:
                pass

        print(f'Snippet length: {len(candidates.items())}')
        if len(candidates.values()) >= limit:

            sorted_keys = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
            sorted_keys = sorted_keys[:limit]
            print('Sorted keys: ', sorted_keys, 'score:', )
            for i in sorted_keys:
                snippet, date_info = i[0].split('--da-sh--')
                snippet_list.append(snippet)
                date_list.append(date_info)
        elif len(candidates.values()) > 0:
            print('Candidate Items: ', candidates.items())
            for i in candidates.items():
                snippet, date_info = i[0].split('--da-sh--')
                snippet_list.append(snippet)
                date_list.append(date_info)

        else:
            snippet_list = []
            date_list = []
        return snippet_list, date_list


    def web_retrieve_pics(self, query, gt=None, limit=3, location=None, frac=0):

        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cx}&q={query}"

        candidates = {}
        response = requests.get(url)
        data = json.loads(response.text)

        if location:
            url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cx}&q={location}"
            response = requests.get(url)
            data_loc = json.loads(response.text)
            if 'items' not in data_loc.keys():
                pass
            elif 'items' not in data.keys():
                data = data_loc
            else:
                for i in data_loc['items']:
                    data['items'].append(i)

        date_info = None
        snippet_list = []
        link2img = None
        date_list = []
        cnt = 0

        if 'items' in data.keys():
            print(f'Retrieved length: {len(data["items"])}')

            for i in data['items']:
                cnt += 1
                try:
                    if gt is not None and i['link'] == gt:
                        continue
                    else:
                        date_key = None
                        pagemap = i['pagemap']['metatags']
                        # pagemap is a list of dict, need to go through it to check keys
                        flag = False
                        for I in pagemap:
                            for key in I.keys():
                                if 'og:image' in key:
                                    flag_key = key
                                    flag = True
                                    # for the retrievals with images, get date clues.
                                    for date in I.keys():
                                        if 'time' in date or 'date' in date:
                                            date_key = date
                                            break
                                        else:
                                            date_key = None

                                    if date_key is None:
                                        date_info = i['snippet']
                                    else:
                                        date_info = I[date_key]

                                    link2img = I[flag_key]
                                    break

                            if flag:
                                break

                    if link2img is not None:
                        snippet = i['snippet']
                        try:
                            img = open_image(link2img)
                            img_p = ImagePatch(img)

                            if img is not None:
                                # -------

                                if 0 < frac < 1:
                                    score_image = self.main_patch.best_image_match(img)
                                    score_image = score_image.item()
                                    score_text = self.main_patch.score(snippet)
                                    score_text = score_text.item()
                                    score = score_image + score_text
                                elif frac >= 1:
                                    score = self.main_patch.score(snippet).item()
                                else:
                                    score_image = self.main_patch.best_image_match(img)
                                    score_image = score_image.item()
                                    score = score_image
                                if score > 95:
                                    if score not in candidates.keys() and date_info is not None:
                                        candidates['--da-sh--'.join([snippet, date_info])] = score
                                    elif score not in candidates.keys():
                                        candidates['--da-sh--'.join([snippet, snippet])] = score

                                # img_p.print_and_save_image(
                                #     save_path=(self.pics_dir + f'{score}_{query}_{self.api_key}.png'))

                            else:
                                continue

                        except:
                            # score = 0
                            # if date_info is not None:
                            #     candidates['--da-sh--'.join([snippet, date_info])] = score
                            # else:
                            #     candidates['--da-sh--'.join([snippet, snippet])] = score
                            pass

                    # # being stricter
                    # else:
                    #     snippet = i['snippet']
                    #     score = 0
                    #     if date_info is not None:
                    #         candidates['--da-sh--'.join([snippet, date_info])] = score
                    #     else:
                    #         candidates['--da-sh--'.join([snippet, snippet])] = score

                except KeyError as e:
                    pass

        print(f'Snippet length: {len(candidates.items())}')
        if len(candidates.values()) >= limit:

            sorted_keys = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
            sorted_keys = sorted_keys[:limit]
            print('Sorted keys: ', sorted_keys, 'score:', )
            for i in sorted_keys:
                snippet, date_info = i[0].split('--da-sh--')
                snippet_list.append(snippet)
                date_list.append(date_info)
        elif len(candidates.values()) > 0:
            print('Candidate Items: ', candidates.items())
            for i in candidates.items():
                snippet, date_info = i[0].split('--da-sh--')
                snippet_list.append(snippet)
                date_list.append(date_info)

        else:
            snippet_list = []
            date_list = []
        return snippet_list, date_list


    def update_threshold(self, loc):
        skip = False
        if self.check_valid(loc) and self.check_confident(loc):
            for k in self.keywords_threshold.keys():
                if self.check_included(loc, k):
                    self.keywords_threshold[k] += 1
                    skip = True
                    break
            if not skip:
                self.keywords_threshold[loc] = 1

    def get_loc_list(self):
        r = []
        for item in self.continent_threshold:
            for k, v in self.country_threshold[item].items():
                if v > 0:
                    r.append(k)
        return r

    def get_year_list(self):
        return [k for k, v in self.year_threshold.items() if v > 0]

    def _reasoner_hie_separate(self, threshold=5):

        self.get_ready()
        self.date_result = None
        self.location_result = None
        threshold = threshold
        cele_exist = False
        threshold_reached = False
        text_clear = False
        date_result = None
        time_root = False
        ranges = defaultdict(int)

        names = []

        cleaned_txt = None

        reg_chas = []
        contexts = []
        level_confidence = 0
        loc_confidence = 0

        # FIRST PRINCIPLE
        # GROUND KEYWORDS

        # all the keywords I may have
        self.text = set(self.text.split(', '))
        # self.event
        self.c_keywords = ', '.join(self.keywords)
        if self.text:
            concrete_keywords = self.gpt_query(
                f'"{self.event}"\n"{self.text}"\n"{self.c_keywords}"\n Above texts are describing'
                f' an same event. Please conclude the event with no more than 5 words. '
                f'Keep your answer informative, short and concise. Short Answer:')
            LOC = self.gpt_query(
                f'"{self.event}"\n"{self.text}"\n"{self.c_keywords}"\n Does any of the above text contains '
                f'location information such as city, country or continent? If yes, please answer that '
                f'city, country or continent. Otherwise, answer No. Short Answer:')
        else:
            concrete_keywords = self.gpt_query(
                f'"{self.event}"\n"{self.c_keywords}"\n Above texts are describing'
                f' an same event. Please conclude the event with no more than 5 words. '
                f'Keep your answer informative, short and concise. Short Answer:')
            LOC = self.gpt_query(
                f'"{self.event}"\n"{self.c_keywords}"\n Does any of the above text contains '
                f'location information such as city, country or continent? If yes, please answer that '
                f'city, country or continent. Otherwise, answer No. Short Answer:')

        # self.c_keywords = concrete_keywords

        if 'no' not in LOC:
            if self.check_valid(LOC):
                loc = self.loc_complete(LOC)
                loc_confidence += 1

                if self.check_confident(loc):
                    self.location_result = loc
                    loc_confidence += 1
                    self.update_loc_hie(loc)

                    if self._check_threshold(threshold):
                        threshold_reached = True

        if self.location_result:
            print(self.location_result)

        landmarks = self.main_patch.find('famous or special entity')

        for l in landmarks:
            if not threshold_reached:
                if self.str_to_bool(l.simple_query('Are you confident that this point to a specific location such as a'
                                                   'city, country or a continent? '
                                                   'Answer with yes or no. Short Answer:')):
                    loc = l.simple_query('In which city is this image most likely taken?')
                    if 'non' not in loc.lower():
                        if self.check_valid(loc):
                            loc = self.loc_complete(loc)
                            if self.check_confident(loc):
                                self.update_loc_hie(loc)

                if self._check_threshold(threshold):
                    threshold_reached = True

        # ground text
        # rec_txt(self.txt_dir, f'----------first principle----------')
        if not threshold_reached:
            # check time from text
            
            time_candidate = self.gpt_query(f'{self.time_clue}\nGiven above information, please ground a specific year '
                                            f'or month or date from it and answer with it only. If you can not find a '
                                            f'date, answer No. Short Answer:')
            if 'no' not in time_candidate.lower():
                date = self.gpt_query(f'Please conclude the time in f{time_candidate}. Short answer:')
                self.date_result = date.replace('.', '')
                level_confidence += 1

                time_root = True
            else:
                time_root = False

            # deal with
            if self.str_to_bool(
                    self.main_patch.simple_query('Are there any recognizable words on this image? Answer '
                                                 'with yes or no:')):
                txt_exist = True
                main_text = set(list(self.main_patch.read_text()))
                # rec_txt(self.txt_dir, f'Text read from : {main_text}')
                main_text = ', '.join(main_text)

                for text in self.entities['text']:
                    context = text.simple_query('What are the words shown on this image？Short Answer:')
                    context = context.split(', ')
                    if len(context) != 0:
                        for words in context:
                            contexts.append(words)
                    else:
                        continue

                contexts = list(set(contexts))

                # print(f'Show the OCR-ed text: {", ".join(contexts)}')
                # rec_txt(self.txt_dir, f'Show the OCR-ed text: {", ".join(contexts)}')
                # rec_txt(self.txt_dir, f'Show the OCR-ed text: {languages}')
                if len(", ".join(contexts)) >= 3:
                    text_clear = True
                    if self.text:
                        cleaned_txt = self.gpt_query(
                            f'"{self.text}"\n"{", ".join(contexts)}"\n"{main_text}"\nGiven above words and characters, '
                            f'please remove the unrecognizable words and organize the most informative words to a clean '
                            f'text. Please be concise and informative. Short Answer:')
                    else:
                        cleaned_txt = self.gpt_query(
                            f'"{", ".join(contexts)}"\n"{main_text}"\nGiven above words and characters, '
                            f'please remove the unrecognizable words and organize the most informative words to a clean '
                            f'text. Please be concise and informative. Short Answer:')
                    # print('--------------After Clean-------------')
                    # print(f'The cleaned text is: {cleaned_txt}')
                    # rec_txt(self.txt_dir, f'Show the OCR-ed text: {cleaned_txt}')

                    time_candidate = self.gpt_query(f'{cleaned_txt}\nGiven above information, please ground a '
                                                    f'specific year or month or date from it and answer with it '
                                                    f'only. If you can not find a date, answer No.')
                    if 'no' not in time_candidate.lower():

                        # print('--------------Time Clue Contained------------')
                        # rec_txt(self.txt_dir, f'--------------Time Clue Contained------------')
                        if time_root is True:
                            date = self.gpt_query(f'"{time_candidate}"\n"{self.date_result}". Are these dates possible '
                                                  f'refer to the same date? If you think it is possible, please answer '
                                                  f'the date that is more specific. Otherwise, please answer "NO" only. '
                                                  f'Short Answer:')

                            if 'no' not in date.lower():
                                self.date_result = date.replace('.', '')
                                level_confidence += 1
                        else:
                            time_root = True
                            level_confidence += 1
                        # print(f'The time clue is: {date_result}')
                        # rec_txt(self.txt_dir, f'The time clue is: {date_result}')

                    location_candidate = self.gpt_query(f'{cleaned_txt}\nGiven above information, please ground a '
                                                        f'continent or a country or a city from it and answer with it '
                                                        f'only. If you can not find a date, answer No.')
                    # print('--------------Loc Clue Contained------------')
                    if 'no' in location_candidate.lower():
                        loc_clues = self.gpt_query(
                            f'What is the loc clue for the recognized words "{cleaned_txt}"? '
                            f'Short answer:')
                        loc = self.gpt_query(f'Please conclude the loc in f{loc_clues}. Short answer:')
                        # print(f'loc {loc} from loc_clues')
                        # rec_txt(self.txt_dir, f'loc {loc} from loc_clues')
                        if self.check_valid(loc):
                            # rec_txt(self.txt_dir, f'loc {loc} from loc_clues valid')
                            loc_result = loc.replace('.', '')
                            loc_result = self.loc_complete(loc_result)
                            if self.check_confident(loc_result):
                                loc_result = loc_result.replace('\n', '')
                                # print(f'The loc clue is: {loc_result}')
                                # rec_txt(self.txt_dir, f'loc {loc_result} from loc_clues confident and updated')
                                self.update_loc_hie(loc_result)

            if self._check_threshold(threshold):
                threshold_reached = True

        # keywords(EVERY IMAGE WILL HAVE)
        # ground people
        # Restricted the celebrities
        if not threshold_reached:
            # names = []
            self.entities['celebrities'] = self.main_patch.find('celebrities or famous faces')
            if len(self.entities['celebrities']) != 0:  # need to check what is the right thing
                # print('----------There might be one or more celebrities-----------')
                # rec_txt(self.txt_dir, '----------There might be one or more celebrities-----------')
                for cele in self.entities['celebrities']:

                    if self.str_to_bool(
                            cele.simple_query('Are you confident that this person is famous? Answer '
                                              'with yes or no:')):
                        name = cele.simple_query("What is the name of this person?")
                        if 'person' in name.lower() or 'people' in name.lower() or 'celeb' in name.lower():
                            name = cele.simple_query("What is the name of this celebrity that you are most "
                                                     "confident with?")
                            if 'person' in name.lower() or 'people' in name.lower() or 'celeb' in name.lower():
                                continue

                        # rec_txt(self.txt_dir, f'name from cele_{cnt}: {name}')
                        names.append(name)
                        # loc update
                        loc_refresh = self.main_patch.simple_query(
                            f'Given {name} in this image, what is the most possible country where this '
                            f'image was taken? Answer with the country and avoid any prompts. Short Answer:')
                        loc_refresh = self.main_patch.simple_query(
                            f'In which city of {loc_refresh} was this picture most likely taken? Answer the city '
                            f'only and avoid any prompts. Short Answer:'
                        )

                        loc_refresh = loc_refresh.replace('\n', '')

                        if self.check_valid(loc_refresh):
                            loc_refresh = self.loc_complete(loc_refresh)
                            if self.check_confident(loc_refresh):
                                # rec_txt(self.txt_dir, f'loc of {name} from cele_{cnt}: {loc_refresh} accepted and updated')
                                self.update_loc_hie(loc_refresh)

                        # time update
                        time_range = self.gpt_query(f'What is the most possible time range with {name}? Format your'
                                                    f' answer as "start-end" and answer the range only. '
                                                    f'Short Answer:')
                        if self.date_result:
                            if 'yes' in self.gpt_query(f'"{time_range}"\n"{self.date_result}"\n'
                                                       f'Is the time range aligned with the later date? Answer Yes or No only. '
                                                       f'Short Answer:'):
                                level_confidence += 1
                        # rec_txt(self.txt_dir, f'time range of {name} from cele_{cnt}: {time_range}')
                        year_list = self.range2interval(time_range)
                        if year_list is not None:
                            year_list = set(year_list)
                            if len(year_list.intersection(set(self.years))) != 0:
                                self.years = set(year_list.intersection(set(self.years)))

                            for yyyy in year_list:
                                self.update_time_hie(yyyy)
                                ranges[yyyy] += 1

                            level_confidence += 1
                        # print(
                        #     f'Based on face {name}, the corresponding loc are {loc_refresh}, and year range{yyyy}')

                if len(names) != 0:
                    cele_exist = True

            if len(self.main_patch.find('people')) != 0:

                if 'no' not in self.main_patch.simple_query('Are those people wearing'
                                                            ' regionally? Answer Yes or No.').lower():
                    rgch = self.main_patch.simple_query('What special are the people shown on this image wearing? '
                                                        'Short Answer:')
                    loc = self.gpt_query(f'What are the most relevant city where {rgch} are commonly '
                                         f'wore? Answer with the country and avoid any prompts. Short Answer:')

                    if self.check_valid(loc):
                        loc = self.loc_complete(loc)
                        if self.check_confident(loc):
                            # rec_txt(self.txt_dir, f'corresponding loc of people_{cnt}\'s {rgch}:{loc} accepted')
                            self.update_loc_hie(loc)

                            reg_chas.append(rgch)

                # all cele
                if cele_exist:

                    names = list(set(names))
                    neatnames = []
                    for i in names:
                        if i.lower() not in neatnames:
                            neatnames.append(i.lower())
                    all_cels = ', '.join(neatnames)
                    print(f'Name of all celebrities: {all_cels}')

            if self._check_threshold(threshold):
                threshold_reached = True

        # SECOND PRINCIPLE: Keywords+Text/ Keywords+People/ Text+People

        # with Text:
        if text_clear:
            if not threshold_reached:
                # rec_txt(self.txt_dir, '---------Second Principal-----------')
                # rec_txt(self.txt_dir, '---------text+keywords-----------')

                prompt = f'What is the most relevant city that has keywords {concrete_keywords}, and is ' \
                         f'related to the text {cleaned_txt}? Answer with the city only and avoid any prompts. ' \
                         f'Short Answer:'
                # else:
                #     prompt = f'What is the most relevant city where {langs} are commonly used and has ' \
                #              f'{concrete_keywords}? Answer with the city only and avoid any prompts. Short Answer:'

                loc = self.gpt_query(prompt)
                if self.location_result:
                    location_candidate = self.gpt_query(f'"{loc}"\n"{self.location_result}"\n'
                                                        f'Are these two locations pointing to the same place? If '
                                                        f'they are, answer the complete location only. Otherwise, '
                                                        f'answer No. Concise Answer:')
                    if 'no' not in location_candidate.lower() and location_candidate is not None:
                        loc_confidence += 1
                        loc = location_candidate
                        self.location_result = location_candidate

                if self.check_valid(loc):
                    loc = self.loc_complete(loc)

                    if self.check_confident(loc):
                        self.update_loc_hie(loc)

            if self._check_threshold(threshold):
                # print('Threshold reached at level 2')
                threshold_reached = True

        # # with celebrities
        if cele_exist:

            for n in names:

                if self.location_result and 'no' not in self.gpt_query(f'"{n}"\n"{self.location_result}"\n'
                                                                       f'Given above celebrity and location, is there '
                                                                       f'any event that happened in the location and '
                                                                       f'the celebrity participated? Answer yes or '
                                                                       f'no.').lower():
                    locs = [self.location_result]

                # rec_txt(self.txt_dir, f'----------GPT answers----------')
                else:
                    prompt = f'What are the 3 most possible cities associated with {n} and {concrete_keywords}? Answer ' \
                             f'with the cities only, separate with commas and avoid any prompts. Short Answer:'
                    locs = self.gpt_query(prompt).split(', ')
                    locs = list(set(locs))

                # kept
                for loc in locs:
                    if self.check_valid(loc):
                        loc = self.loc_complete(loc)
                        if self.check_confident(loc):
                            loc = loc.replace('\n', '')
                            self.update_loc_hie(loc)

                    if self._check_threshold(threshold):
                        threshold_reached = True
                        break

        #     # print(f'Second Principle with cels and Keywords:{self.city_threshold}')

        # THIRD PRINCIPLE
        # with text and cele
        if text_clear and cele_exist:
            # rec_txt(self.txt_dir, f'---------Text and Cele---------')
            for n in names:
                if not threshold_reached:

                    if self.location_result:
                        prompt = f'"{concrete_keywords}"\n"{cleaned_txt}"\n"{n}"\n' \
                                 f'Given above text, keywords and celebrity for an event, is there a strong connection ' \
                                 f'between the information and the location "{self.location_result}"? Answer yes or no:'
                        if 'no' not in self.gpt_query(prompt).lower():
                            loc_confidence += 1
                        if loc_confidence >= 3:
                            self.location_result = self.location_result.replace('\n', '')
                            self.update_loc_hie(self.location_result)
                    else:
                        prompt = f'What are the 3 most relevant cities with "{concrete_keywords} and {n} and ' \
                                 f'{cleaned_txt}"? Answer with the cities only, separate with commas and avoid any ' \
                                 f'prompts. Short Answer:'
                        locs = self.gpt_query(prompt).split(', ')
                        locs = list(set(locs))
                        # print(f'With Prompt "{prompt}", gpt answers: {locs}')
                        # rec_txt(self.txt_dir, f'With Prompt "{prompt}", gpt answers: {locs}')
                        for loc in locs:
                            if self.check_valid(loc):
                                loc = self.loc_complete(loc)
                                if self.check_confident(loc):
                                    loc = loc.replace('\n', '')
                                    # rec_txt(self.txt_dir, f'loc {loc} accepted')
                                    self.update_loc_hie(loc)
                            # else:
                            #     print(f'Loc {loc} from celes_key_lan_txt refused')
                            #     rec_txt(self.txt_dir, f'loc {loc} refused!!!!')
                            #     print(f'is it because of the confidence? ')

        # location search
        txt_flag = False
        if not text_clear and not cele_exist:

            keys = f'Where {self.event} {concrete_keywords} photo'
            # keys = f'Where, {self.event}, {concrete_keywords}, photo'
            # self.main_patch.print_and_save_image(save_path=(self.pics_dir + 'keys_' + keys + '.png'))
            print('Keys:', keys)
            snippet, date = self.web_retrieve_pic(keys, self.web_url, limit=5)
            if snippet is None or len(snippet) == 0:
                pass
            else:
                for i in range(min([len(snippet), len(date)])):
                    if not threshold_reached:
                        if self.location_result and 'no' not in self.gpt_query(
                                f'"{snippet[i]}"\n"{self.location_result}"\n'
                                f'Given above text and location, are they pointing to the same'
                                f'location? Answer yes or no:'):
                            loc_confidence += 1
                            if loc_confidence >= 3:
                                self.update_loc_hie(self.location_result)
                        else:
                            location = self.gpt_query(
                                f'What is the most possible city associated with {snippet[i]}? Answer the location only and avoid any '
                                f'prompts. Short Answer:')

                            if self.check_valid(location):
                                location = self.loc_complete(location)
                                if self.check_confident(location):
                                    self.update_loc_hie(location)

        if text_clear:
            if not threshold_reached:
                # first hierarchy
                keys = f'Where {cleaned_txt} {self.event} photo'
                # self.main_patch.print_and_save_image(save_path=(self.pics_dir + 'keys_' + keys + '.png'))
                print('Keys:', keys)
                # if not txt_flag:
                #     rec_txt(self.path_dir, f'{self.cnt}')
                #     txt_flag = True
                snippet, date = self.web_retrieve_pic(keys, self.web_url, limit=5)
                # rec_txt(self.txt_dir, f'Search Key: {keys}\nSearched Results: {snippet} with corresponding '
                #                       f'publishing date:{date}')
                if snippet is None or len(snippet) == 0:
                    pass
                else:
                    for i in range(min([len(snippet), len(date)])):
                        if not threshold_reached:
                            if self.location_result and 'no' not in self.gpt_query(
                                    f'"{snippet[i]}"\n"{self.location_result}"\n'
                                    f'Given above text and location, are they pointing to the same'
                                    f'location? Answer yes or no:'):
                                loc_confidence += 1
                                if loc_confidence >= 3:
                                    self.update_loc_hie(self.location_result)
                            else:
                                location = self.gpt_query(
                                    f'What is the most possible city associated with {snippet[i]}? Answer the location only and avoid any '
                                    f'prompts. Short Answer:')

                                if self.check_valid(location):
                                    location = self.loc_complete(location)
                                    if self.check_confident(location):
                                        self.update_loc_hie(location)

        #                             # date_ = self.gpt_query(
        #                             #     f'What is the date mentioned in {date[i]}? Answer the date in yyyy-mm-dd '
        #                             #     f'only and avoid any prompts. Short Answer:')
        #                             # if self.n_timerange > 1:
        #                             #     if date_.split('-')[0] not in self.years:
        #                             #         continue
        #                             #     else:
        #                             #         self.update_time_hie(date_)
        #                             # else:
        #                             #     self.update_time_hie_part(date_, n=1)

        if cele_exist:
            #
            if not threshold_reached:
                for n in names:
                    keys = f'Where {n} {self.event} photo'
                    print('Keys:', keys)
                    # self.main_patch.print_and_save_image(save_path=(self.pics_dir + str(self.cnt) + '++' + keys + f'.png'))
                    # if not txt_flag:
                    #     rec_txt(self.path_dir, f'{self.cnt}')
                    #     txt_flag = True
                    snippet, date = self.web_retrieve_pic(keys, self.web_url, limit=5)
                    # rec_txt(self.txt_dir, f'Search Key: {keys}\nSearched Results: {snippet} with corresponding '
                    #                       f'publishing date:{date}')
                    if snippet is None or len(snippet) == 0:
                        pass
                    else:
                        for i in range(min([len(snippet), len(date)])):
                            if not threshold_reached:
                                if self.location_result and 'no' not in self.gpt_query(
                                        f'"{snippet[i]}"\n"{self.location_result}"\n'
                                        f'Given above text and location, are they pointing to the same'
                                        f'location? Answer yes or no:'):
                                    loc_confidence += 1
                                    if loc_confidence >= 3:
                                        self.update_loc_hie(self.location_result)
                                else:
                                    location = self.gpt_query(
                                        f'What is the most possible city associated with {snippet[i]}? Answer the location only and avoid any '
                                        f'prompts. Short Answer:')
                                    if self.check_valid(location):
                                        location = self.loc_complete(location)
                                        if self.check_confident(location):
                                            self.update_loc_hie(location)
                                        # rec_txt(self.txt_dir, f'Loc {location} updated in cele search')
                                        #     date_ = self.gpt_query(
                                        #         f'What is the date mentioned in {date[i]}? Answer the date in yyyy-mm-dd '
                                        #         f'only and avoid any prompts. Short Answer:')
                                        #     if self.n_timerange > 1:
                                        #         if date_.split('-')[0] not in self.years:
                                        #             continue
                                        #         else:
                                        #             self.update_time_hie(date_)
                                        #     else:
                                        #         self.update_time_hie_part(date_, n=1)

        if cele_exist and text_clear:  ## all three entities are available
            # second hierarchy
            if not threshold_reached:
                # keys = ', '.join([cleaned_txt, names[-1]]) + ', photo'
                for n in names:
                    keys = f'Where {n} {cleaned_txt} {self.event} photo'
                    print('Keys:', keys)
                    # self.main_patch.print_and_save_image(save_path=(self.pics_dir + str(self.cnt) + '++' + keys + f'.png'))
                    # if not txt_flag:
                    #     rec_txt(self.path_dir, f'{self.cnt}')
                    #     txt_flag = True
                    snippet, date = self.web_retrieve_pic(keys, self.web_url, limit=5)
                    if snippet is None or len(snippet) == 0:
                        pass
                    else:
                        for i in range(min([len(snippet), len(date)])):
                            if not threshold_reached:
                                if self.location_result and 'no' not in self.gpt_query(
                                        f'"{snippet[i]}"\n"{self.location_result}"\n'
                                        f'Given above text and location, are they pointing to the '
                                        f'same location? Answer yes or no:'):
                                    loc_confidence += 1
                                    if loc_confidence >= 3:
                                        self.update_loc_hie(self.location_result)
                                else:
                                    location = self.gpt_query(
                                        f'What is the most possible city associated with {snippet[i]}? Answer the location only and avoid any '
                                        f'prompts. Short Answer:')
                                    if self.check_valid(location):
                                        location = self.loc_complete(location)
                                        if self.check_confident(location):
                                            self.update_loc_hie(location)
                                        # rec_txt(self.txt_dir, f'text and cele key {keys} searched {location} accpeted')
                                        #     date_ = self.gpt_query(
                                        #         f'What is the date mentioned in {date[i]}? Answer the date in yyyy-mm-dd only and '
                                        #         f'avoid any prompts. Short Answer:')
                                        #     if self.n_timerange > 1:
                                        #         if date_.split('-')[0] not in self.years:
                                        #             continue
                                        #         else:
                                        #             self.update_time_hie(date_)
                                        #     else:
                                        #         self.update_time_hie_part(date_, n=0.3)

        maj_vote_cont = max(self.continent_threshold, key=self.continent_threshold.get)
        maj_vote_coun = max(self.country_threshold[maj_vote_cont], key=self.country_threshold[maj_vote_cont].get)
        maj_vote_city = max(self.city_threshold[maj_vote_coun], key=self.city_threshold[maj_vote_coun].get)
        print(f'Major Vote City, Country, Continent: {maj_vote_cont, maj_vote_coun, maj_vote_cont}')
        if self.location_result and loc_confidence >= 5:
            location = self.location_result
        elif maj_vote_city == 'city':
            if maj_vote_coun != 'country':
                location = maj_vote_coun
            else:
                if maj_vote_cont != 'continent':
                    location = maj_vote_cont
                else:
                    final_loc = self.main_patch.simple_query('What is the most possible country of this image? '
                                                             'Short Answer:')
                    location = self.loc_complete(final_loc)
        else:
            location = maj_vote_city

        blip2_loc = self.main_patch.simple_query(
            'What is the most possible city, country or continent when this image was '
            'taken? Short Answer:')
        prompt = 'a photo taken in '
        only_loc = location
        location = self.main_patch.best_text_match([prompt + self.loc_complete(blip2_loc), prompt + location])

        # search for time

        if not cele_exist and not text_clear:
            # we only have concrete keywords
            key = f'When {self.event}  photo'
            keys = f'When {self.event} {location} photo'
            # self.main_patch.print_and_save_image(save_path=(self.pics_dir + 'sum_' + keys + '.png'))
            print('Keys with location:', keys)
            snippet, date = self.web_retrieve_pics(key, self.web_url, limit=5, location=keys)

            if snippet is None or len(snippet) == 0:
                pass
            else:
                for i in range(min([len(snippet), len(date)])):
                    date_ = self.gpt_query(
                        f'What is the date mentioned in {date[i]}? Answer the date in '
                        f'yyyy-mm-dd only and avoid any prompts. Short Answer:')
                    if time_root:
                        if 'no' not in self.gpt_query(f'"{snippet[i]}"\n"{date_}"\n'
                                                      f'Candidate time: "{self.date_result}"\n'
                                                      f'Is any of the snippet or the following date close to the '
                                                      f'candidate time? Answer yes or no:'):

                            date_result = self.gpt_query(
                                f'What is the date mentioned in {self.date_result}? Answer the date in '
                                f'yyyy-mm-dd only and avoid any prompts. Short Answer:')

                            if level_confidence >= 2:
                                self.update_time_hie(date_result)
                            else:
                                self.update_time_hie_part(date_result)

                        else:
                            date_to_compare = [date_result, date_]
                            date_to_update = self.main_patch.best_text_match(date_to_compare)
                            self.update_time_hie_part(date_to_update)
                    else:
                        date_to_compare = [self.date_can, date_]
                        date_to_update = self.main_patch.best_text_match(date_to_compare)
                        self.update_time_hie_part(date_to_update)

        if cele_exist:
            if not threshold_reached:
                # we only have concrete keywords
                for n in names:
                    key = f'When {n} {self.event}  photo'
                    keys = f'When {n} {self.event} {location} photo'
                    # self.main_patch.print_and_save_image(save_path=(self.pics_dir + 'cele_' + keys + '.png'))
                    print('Keys with location:', keys)
                    snippet, date = self.web_retrieve_pics(key, self.web_url, limit=5, location=keys)
                    if snippet is None or len(snippet) == 0:
                        pass
                    else:
                        for i in range(min([len(snippet), len(date)])):
                            date_ = self.gpt_query(
                                f'What is the date mentioned in {date[i]}? Answer the date in '
                                f'yyyy-mm-dd only and avoid any prompts. Short Answer:')
                            if time_root:
                                if 'no' not in self.gpt_query(f'"{snippet[i]}"\n"{date_}"\n'
                                                              f'Candidate time: "{self.date_result}"\n'
                                                              f'Is any of the snippet or the following date close to the '
                                                              f'candidate time? Answer yes or no:'):

                                    date_result = self.gpt_query(
                                        f'What is the date mentioned in {self.date_result}? Answer the date in '
                                        f'yyyy-mm-dd only and avoid any prompts. Short Answer:')

                                    if level_confidence >= 2:
                                        self.update_time_hie(date_result)
                                    else:
                                        self.update_time_hie_part(date_result)

                                else:
                                    date_to_compare = [date_result, date_]
                                    date_to_update = self.main_patch.best_text_match(date_to_compare)
                                    self.update_time_hie_part(date_to_update)
                            else:
                                date_to_compare = [self.date_can, date_]
                                date_to_update = self.main_patch.best_text_match(date_to_compare)
                                self.update_time_hie_part(date_to_update)

        if text_clear:
            if not threshold_reached:
                # we only have concrete keywords
                key = f'When {self.event} {cleaned_txt} photo'
                keys = f'When {self.event} {cleaned_txt} {location} photo'
                # self.main_patch.print_and_save_image(save_path=(self.pics_dir + 'txt_' + keys + '.png'))
                print('Keys with location:', keys)
                snippet, date = self.web_retrieve_pics(key, self.web_url, limit=5, location=keys)
                if snippet is None or len(snippet) == 0:
                    pass
                else:
                    for i in range(min([len(snippet), len(date)])):
                        date_ = self.gpt_query(
                            f'What is the date mentioned in {date[i]}? Answer the date in '
                            f'yyyy-mm-dd only and avoid any prompts. Short Answer:')
                        if time_root:
                            if 'no' not in self.gpt_query(f'"{snippet[i]}"\n"{date_}"\n'
                                                          f'Candidate time: "{self.date_result}"\n'
                                                          f'Is any of the snippet or the following date close to the '
                                                          f'candidate time? Answer yes or no:'):

                                date_result = self.gpt_query(
                                    f'What is the date mentioned in {self.date_result}? Answer the date in '
                                    f'yyyy-mm-dd only and avoid any prompts. Short Answer:')

                                if level_confidence >= 2:
                                    self.update_time_hie(date_result)
                                else:
                                    self.update_time_hie_part(date_result)

                            else:
                                date_to_compare = [date_result, date_]
                                date_to_update = self.main_patch.best_text_match(date_to_compare)
                                self.update_time_hie_part(date_to_update)
                        else:
                            date_to_compare = [self.date_can, date_]
                            date_to_update = self.main_patch.best_text_match(date_to_compare)
                            self.update_time_hie_part(date_to_update)

        if cele_exist and text_clear:  ## all three entities are available
            if not threshold_reached:
                for n in names:
                    key = f'When {n} {cleaned_txt} {self.event} photo'
                    keys = f'When {n} {cleaned_txt} {self.event} {location} photo'
                    # self.main_patch.print_and_save_image(save_path=(self.pics_dir + 'both_' + keys + '.png'))
                    print('Keys with location:', keys)
                    # self.main_patch.print_and_save_image(save_path=(self.pics_dir + str(self.cnt) + '++' + keys + f'.png'))
                    # if not txt_flag:
                    #     rec_txt(self.path_dir, f'{self.cnt}')
                    #     txt_flag = True
                    snippet, date = self.web_retrieve_pics(key, self.web_url, limit=5, location=keys)
                    if snippet is None or len(snippet) == 0:
                        pass
                    else:
                        for i in range(min([len(snippet), len(date)])):
                            date_ = self.gpt_query(
                                f'What is the date mentioned in {date[i]}? Answer the date in '
                                f'yyyy-mm-dd only and avoid any prompts. Short Answer:')
                            if time_root:
                                if 'no' not in self.gpt_query(f'"{snippet[i]}"\n"{date_}"\n'
                                                              f'Candidate time: "{self.date_result}"\n'
                                                              f'Is any of the snippet or the following date close to the '
                                                              f'candidate time? Answer yes or no:'):

                                    date_result = self.gpt_query(
                                        f'What is the date mentioned in {self.date_result}? Answer the date in '
                                        f'yyyy-mm-dd only and avoid any prompts. Short Answer:')

                                    if level_confidence >= 2:
                                        self.update_time_hie(date_result)
                                    else:
                                        self.update_time_hie_part(date_result)

                                else:
                                    date_to_compare = [date_result, date_]
                                    date_to_update = self.main_patch.best_text_match(date_to_compare)
                                    self.update_time_hie_part(date_to_update)
                            else:
                                date_to_compare = [self.date_can, date_]
                                date_to_update = self.main_patch.best_text_match(date_to_compare)
                                self.update_time_hie_part(date_to_update)

        maj_vote_year = max(self.year_threshold, key=self.year_threshold.get)
        if len(self.month_threshold[maj_vote_year].keys()) > 0:
            maj_vote_month = max(self.month_threshold[maj_vote_year], key=self.month_threshold[maj_vote_year].get)
            if len(self.date_threshold[maj_vote_month].keys()) > 0:
                maj_vote_day = max(self.date_threshold[maj_vote_month], key=self.date_threshold[maj_vote_month].get)
            else:
                maj_vote_day = maj_vote_month
        else:
            maj_vote_day = maj_vote_year

        if self.date_result is None:
            date_output = maj_vote_day
        else:
            if self.str_to_bool(
                    self.gpt_query(f'"{maj_vote_day}"\n"{date_result}"\n. '
                                   f'Given above two dates, are they pointing to the same year and month? '
                                   f'Answer with yes or no. Short Answer: ')):
                date_output = maj_vote_day

            else:

                date_output = self.gpt_query(
                    f'What is the date from cleaning the date clue {date_result}? Answer the '
                    f'date in yyyy-mm-dd only and avoid any prompts. Short Answer:')

        blip2_date = self.main_patch.simple_query('What is the most possible date when this image was '
                                                  'taken? Short Answer:')
        prompt = 'a photo taken in '
        if ranges:
            yyyy = str(date_output[-4:])
            if yyyy in ranges:
                pass
            elif yyyy not in ranges:
                return self.main_patch.best_text_match([prompt + max(ranges, key=ranges.get),
                                                        prompt + blip2_date,
                                                        prompt + date_output]), \
                       location, date_result, only_loc

        return self.main_patch.best_text_match([prompt + blip2_date, prompt + date_output]) \
            , location, date_output, only_loc

    def baseline_blip(self):

        date = self.main_patch.simple_query('In what date was this image taken?').replace('on ', '')
        date_result = self.gpt_query(f'Convert the date {date} to yyyy-mm-dd. Answer the corresponding'
                                     f' yyyy-mm-dd only. Short Answer:')

        loc = self.main_patch.simple_query('What is the most possible city or country where this image was possibly'
                                           'taken? Short Answer:')

        loc_result = self.loc_complete(loc).replace('\n', '')

        return date_result, loc_result

    def baseline_GPT4(self):

        base64_image = encode_image(self.img_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        # date
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "When was this image taken? Answer with the time only. Short Answer:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            date = response.json()['choices'][0]['message']['content']
        except:
            date = ''

        # location
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Where was this image taken? Answer with the location only. Short Answer:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        try:
            loc = response.json()['choices'][0]['message']['content']
        except:
            loc = ''

        return date, loc

    def get_ready(self):

        # initial definition
        self.entities['celebrities'] = self.main_patch.find('celebrities or famous faces')
        self.entities['text'] = self.main_patch.find('words')
