import logging
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
import clip
import os
import numpy as np
import re
import random


LABEL_TO_IDX = {'NoRel':0, 'Identical':1, 'Hierarchical':2}
IDX_TO_LABEL = {0:'NoRel', 1:'Identical', 2:'Hierarchical'}

class M2E2HRTrain(Dataset):
    def __init__(self, cfg, logger=None):
        if logger is None:
            logger = logging.getLogger('M2E2HR.dataset')
        self.logger = logger
        self.pseudo_annotation_file = cfg.DATASET.TRAIN.PSEUDO_LABEL_FILE_PATH
        self.embed_dir = cfg.DATASET.TRAIN.VIDEO_EMBED_DIR
        self.all_events_file = cfg.DATASET.TRAIN.ALL_EVENTS_FILE
        self.video_shots_file = cfg.DATASET.TRAIN.VIDEO_SHOTS
        self.grounded_events_file = cfg.DATASET.TRAIN.GROUNDED_EVENTS_FILE
        self.video_events_context_length = cfg.MODEL.VIDEO_EVENTS_CONTEXT_LENGTH
        self.video_embed_dim = cfg.MODEL.VID_EMBED_DIM

        with open(self.all_events_file) as f:
            self.all_events = json.load(f)
        with open(self.video_shots_file) as f:
            self.video_shots = json.load(f)
        with open(self.pseudo_annotation_file) as f:
            self.data = json.load(f)
        with open(self.grounded_events_file) as f:
            self.grounded_events = json.load(f)

        self.target_class_list = []
        self.target_class_count = [0 for _ in range(len(LABEL_TO_IDX))]
        self.all_event_event_relations = []
        self.all_hierarchical_event_event_relations = []
        self.all_coref_event_event_relations = []
        self.all_norel_event_event_relations = []
        num_mismatch_of_totale2epairs_vs_foundpairs = 0
        for vid_name, e2es in self.data.items():
            hierarchical_rels_set = set()
            coref_rels_set = set()
            norel_rels_set = set()
            for e2e in e2es:
                for grounded_segment in e2e['grounded_segments']:
                    hierarchical_data_dict={
                        'vid': vid_name,
                        'video_segment': tuple(grounded_segment),
                        'event_sent': e2e['sent_1'],
                        'event_start_char': e2e['e1_start_char'],
                        'label': LABEL_TO_IDX['Hierarchical']
                    }
                    coref_data_dict={
                        'vid': vid_name,
                        'video_segment': tuple(grounded_segment),
                        'event_sent': e2e['sent_2'],
                        'event_start_char': e2e['e2_start_char'],
                        'label': LABEL_TO_IDX['Identical']
                    }
                    hierarchical_tuple = tuple(hierarchical_data_dict.values())
                    coref_tuple = tuple(coref_data_dict.values())
                    # Remove duplicate hierarchical relations due to one-to-many parent-child text e2es
                    # And many-to-one child to video segment grounding
                    if hierarchical_tuple not in hierarchical_rels_set:
                        self.all_hierarchical_event_event_relations.append(hierarchical_data_dict)
                        hierarchical_rels_set.add(hierarchical_tuple)
                    # Remove duplicate identical relaitons due to many-to-one parent-child text e2es
                    # And one-to-one child to video segment grounding
                    if coref_tuple not in coref_rels_set:
                        self.all_coref_event_event_relations.append(coref_data_dict)
                        coref_rels_set.add(coref_tuple)

            grounded_events_in_vid = self.grounded_events[vid_name]
            for event in grounded_events_in_vid:
                for grounded_segment in event['grounded_segments']:
                    coref_data_dict={
                        'vid': vid_name,
                        'video_segment': tuple(grounded_segment),
                        'event_sent': event['sent'],
                        'event_start_char': event['event_char_id'],
                        }
                    coref_tuple = tuple(coref_data_dict.values())
                    # Remove those relations which are in hierarchical
                    if coref_tuple + (LABEL_TO_IDX['Hierarchical'],) in hierarchical_rels_set:
                        continue
                    # Remove duplicate identical relaitons due to many-to-one parent-child text e2es
                    # And one-to-one child to video segment grounding
                    coref_data_dict['label'] = LABEL_TO_IDX['Identical']
                    coref_tuple = tuple(coref_data_dict.values())
                    if coref_tuple not in coref_rels_set:
                        self.all_coref_event_event_relations.append(coref_data_dict)
                        coref_rels_set.add(coref_tuple)

            all_events_in_vid = self.all_events[vid_name]
            all_vid_segments_in_vid = self.video_shots[vid_name]
            all_events_in_vid_set = set()
            vid_no_rel_event_event_rels = []
            for event in all_events_in_vid:
                event_tuple = tuple(event.values())
                # Removing duplicate sentences and events
                if event_tuple in all_events_in_vid_set:
                    continue
                all_events_in_vid_set.add(event_tuple)
                for video_segment in all_vid_segments_in_vid:
                    norel_data_dict={
                        'vid': vid_name,
                        'video_segment': tuple(video_segment['segment']),
                        'event_sent': event['sent'],
                        'event_start_char': event['event_char_id'],
                    }
                    norel_tuple = tuple(norel_data_dict.values())
                    # Skip those relations which are hierarchical or identical
                    if norel_tuple + (LABEL_TO_IDX['Hierarchical'],) in hierarchical_rels_set or \
                        norel_tuple + (LABEL_TO_IDX['Identical'],) in coref_rels_set:
                        continue
                    # Sanity Check: skip duplicate
                    if norel_tuple in norel_rels_set:
                        continue
                    norel_rels_set.add(norel_tuple)
                    norel_data_dict['label'] = LABEL_TO_IDX['NoRel']
                    vid_no_rel_event_event_rels.append(norel_data_dict)
            if cfg.DATASET.TRAIN.DOWNSAMPLE_NORELS != None:
                selected_no_rel_event_event_rels = random.choices(vid_no_rel_event_event_rels, 
                                                    k=min(cfg.DATASET.TRAIN.DOWNSAMPLE_NORELS, len(vid_no_rel_event_event_rels)))
            else:
                selected_no_rel_event_event_rels = vid_no_rel_event_event_rels
            self.all_norel_event_event_relations.extend(selected_no_rel_event_event_rels)

            if len((all_events_in_vid_set)) * len(all_vid_segments_in_vid) != len(hierarchical_rels_set) + len(coref_rels_set) + len(norel_rels_set):
                num_mismatch_of_totale2epairs_vs_foundpairs += 1
        
        self.all_event_event_relations = self.all_hierarchical_event_event_relations + self.all_coref_event_event_relations + self.all_norel_event_event_relations
        self.target_class_list = [LABEL_TO_IDX['Hierarchical']] * len(self.all_hierarchical_event_event_relations) + \
                                    [LABEL_TO_IDX['Identical']] * len(self.all_coref_event_event_relations) + \
                                    [LABEL_TO_IDX['NoRel']] * len(self.all_norel_event_event_relations)
        self.target_class_count[LABEL_TO_IDX['Hierarchical']] = len(self.all_hierarchical_event_event_relations)
        self.target_class_count[LABEL_TO_IDX['Identical']] = len(self.all_coref_event_event_relations)
        self.target_class_count[LABEL_TO_IDX['NoRel']] = len(self.all_norel_event_event_relations)

        self.logger.info('Length of all event event relations: {}'.format(len(self.all_event_event_relations)))
        self.logger.info('Length of all hierarchical event event relations: {}'.format(len(self.all_hierarchical_event_event_relations)))
        self.logger.info('Length of all identical event event relations: {}'.format(len(self.all_coref_event_event_relations)))
        self.logger.info('Length of all norel event event relations: {}'.format(len(self.all_norel_event_event_relations)))

    def __len__(self):
        return len(self.all_event_event_relations)

    def get_shot_frames(self, shot, total_num_of_frames):
        shot_start_time, shot_end_time = shot
        shot_start_frame_no = round(shot_start_time * 3) 
        shot_end_frame_no = round(shot_end_time * 3)
        if shot_end_frame_no > total_num_of_frames:
            shot_end_frame_no = total_num_of_frames
        if shot_start_frame_no == shot_end_frame_no:
            frame_number = shot_end_frame_no if shot_end_frame_no < total_num_of_frames else total_num_of_frames-1
            frame_numbers = [frame_number]
        else:
            frame_numbers = list(range(shot_start_frame_no, shot_end_frame_no))
        return frame_numbers

    def __getitem__(self, idx):
        data_dict = self.all_event_event_relations[idx]
        vid_name = data_dict['vid']
        vid_event_segment = data_dict['video_segment']
        text_event_sent = data_dict['event_sent']
        text_event_start_char = data_dict['event_start_char']
        label = data_dict['label']

        event_trigger_id = len(text_event_sent[:text_event_start_char].split())
        text_event_sent_tokens, text_event_index_list = clip.tokenize(text_event_sent, 
                                                            trigger_index=event_trigger_id,
                                                            truncate=True)
        text_event_start_index, text_event_len = text_event_index_list[0], len(text_event_index_list)

        with open(os.path.join(self.embed_dir, vid_name + '.npy'), 'rb') as f:
            video_frame_embeds = np.load(f)

        # Extracting all video Segments
        all_vid_segments_in_vid = self.video_shots[vid_name]
        vid_subevent_idx = all_vid_segments_in_vid.index({"segment": list(vid_event_segment)})
        relevant_vid_segments = all_vid_segments_in_vid
        vid_segments_mask = torch.ones(self.video_events_context_length)

        if len(all_vid_segments_in_vid) > self.video_events_context_length:
            if vid_subevent_idx >= self.video_events_context_length:
                relevant_vid_segments_begin_idx = vid_subevent_idx - self.video_events_context_length + 1
                relevant_vid_segments_end_idx = vid_subevent_idx + 1
                vid_subevent_idx = self.video_events_context_length - 1
            else:
                relevant_vid_segments_begin_idx = 0
                relevant_vid_segments_end_idx = self.video_events_context_length

            relevant_vid_segments = all_vid_segments_in_vid[relevant_vid_segments_begin_idx:relevant_vid_segments_end_idx]
        
        vid_segments_mask[:len(relevant_vid_segments)] = 0
        vid_segments_mask = vid_segments_mask.bool()
            
        vid_segments_embed = torch.zeros(self.video_events_context_length, self.video_embed_dim)
        for i, vid_segment in enumerate(relevant_vid_segments):
            vid_segment_frame_nos = self.get_shot_frames(vid_segment['segment'], video_frame_embeds.shape[0])
            vid_segment_frame_embeds = torch.from_numpy(video_frame_embeds[vid_segment_frame_nos, ...])
            vid_segment_embed = torch.mean(vid_segment_frame_embeds, dim=0)
            vid_segments_embed[i,:] = vid_segment_embed

        return {
            'event_sent': text_event_sent_tokens.squeeze(0), 
            'event_start_idx': text_event_start_index,
            'event_len': text_event_len,
            'vid_subevent_idx': vid_subevent_idx,
            'vid_segments_embed': vid_segments_embed, 
            'vid_segments_mask': vid_segments_mask,
            'labels': label
        }

class M2E2HRVal(Dataset):
    def __init__(self, cfg):
        self.test_file = cfg.DATASET.VAL.TEST_FILE_PATH
        self.annotation_file = cfg.DATASET.VAL.ANNOT_FILE_PATH
        self.embed_dir = cfg.DATASET.VAL.VIDEO_EMBED_DIR
        self.video_shots_file = cfg.DATASET.VAL.VIDEO_SHOTS
        self.video_events_context_length = cfg.MODEL.VIDEO_EVENTS_CONTEXT_LENGTH
        self.video_embed_dim = cfg.MODEL.VID_EMBED_DIM

        with open(self.test_file) as f:
            self.data = json.load(f)
        with open(self.annotation_file) as f:
            self.annotations = json.load(f)
        with open(self.video_shots_file) as f:
            self.video_shots = json.load(f)

        self.all_event_event_relations = []
        sen_beg_end_pattern = re.compile(r'[\r\n\.]')
        for vid_name, annots in self.annotations.items():
            article = vid_name + '\n' + self.data[vid_name]['article']
            for annot in annots:
                event_start_in_article = annot['text_start']
                event_end_in_article = annot['text_end']
                
                event_sent_start_match = sen_beg_end_pattern.search(article[event_start_in_article::-1])
                if event_sent_start_match:
                    event_start_char = event_sent_start_match.start() - 1
                else:
                    event_start_char = event_start_in_article
                event_sent_start = event_start_in_article - event_start_char
                event_sent_end_match = sen_beg_end_pattern.search(article, event_end_in_article)
                if event_sent_end_match:
                    event_sent_end = event_sent_end_match.start()
                else:
                    event_sent_end = len(article)
                event_sent = article[event_sent_start: event_sent_end]
                if event_sent[event_start_char-1] != ' ':
                    event_sent = event_sent[:event_start_char] + ' ' + event_sent[event_start_char:]
                    event_start_char = event_start_char + 1

                event_event_info = {
                    'vid': vid_name,
                    'video_segment': (annot['video_start'], annot['video_end']),
                    'event_sent': event_sent,
                    'event_start_char': event_start_char,
                    'label': LABEL_TO_IDX[annot['label']]
                }
                self.all_event_event_relations.append(event_event_info)


    def __len__(self):
        return len(self.all_event_event_relations)

    def get_shot_frames(self, shot, total_num_of_frames):
        shot_start_time, shot_end_time = shot
        shot_start_frame_no = round(shot_start_time * 3) 
        shot_end_frame_no = round(shot_end_time * 3)
        if shot_end_frame_no > total_num_of_frames:
            shot_end_frame_no = total_num_of_frames
        if shot_start_frame_no == shot_end_frame_no:
            frame_number = shot_end_frame_no if shot_end_frame_no < total_num_of_frames else total_num_of_frames-1
            frame_numbers = [frame_number]
        else:
            frame_numbers = list(range(shot_start_frame_no, shot_end_frame_no))
        return frame_numbers

    def __getitem__(self, idx):
        data_dict = self.all_event_event_relations[idx]
        vid_name = data_dict['vid']
        vid_event_segment = data_dict['video_segment']
        text_event_sent = data_dict['event_sent']
        text_event_start_char = data_dict['event_start_char']
        label = data_dict['label']

        event_trigger_id = len(text_event_sent[:text_event_start_char].split())
        text_event_sent_tokens, text_event_index_list = clip.tokenize(text_event_sent, 
                                                            trigger_index=event_trigger_id,
                                                            truncate=True)
        text_event_start_index, text_event_len = text_event_index_list[0], len(text_event_index_list)

        with open(os.path.join(self.embed_dir, vid_name + '.npy'), 'rb') as f:
            video_frame_embeds = np.load(f)

        # Extracting all video Segments
        vid_segments_embed = torch.zeros(self.video_events_context_length, self.video_embed_dim)
        all_vid_segments_in_vid = self.video_shots[vid_name]
        relevant_vid_segments = all_vid_segments_in_vid
        vid_segments_mask = torch.ones(self.video_events_context_length)
        try:
            vid_subevent_idx = all_vid_segments_in_vid.index({"segment": list(vid_event_segment)})
        except:
            fl = 0
            for vid_seg_tol in range(1,5):
                for cur_idx, segment in enumerate(all_vid_segments_in_vid):
                    segemnt_start, segment_end = segment['segment']
                    if abs(segemnt_start - vid_event_segment[0]) < vid_seg_tol and abs(segment_end - vid_event_segment[1]) < vid_seg_tol:
                        vid_subevent_idx = cur_idx
                        fl=1
                if fl == 1:
                    break
        
        if len(all_vid_segments_in_vid) > self.video_events_context_length:
            if vid_subevent_idx >= self.video_events_context_length:
                relevant_vid_segments_begin_idx = vid_subevent_idx - self.video_events_context_length + 1
                relevant_vid_segments_end_idx = vid_subevent_idx + 1
                vid_subevent_idx = self.video_events_context_length - 1
            else:
                relevant_vid_segments_begin_idx = 0
                relevant_vid_segments_end_idx = self.video_events_context_length
            relevant_vid_segments = all_vid_segments_in_vid[relevant_vid_segments_begin_idx:relevant_vid_segments_end_idx]
        
        vid_segments_mask[:len(relevant_vid_segments)] = 0
        vid_segments_mask = vid_segments_mask.bool() 
            
        for i, vid_segment in enumerate(relevant_vid_segments):
            vid_segment_frame_nos = self.get_shot_frames(vid_segment['segment'], video_frame_embeds.shape[0])
            vid_segment_frame_embeds = torch.from_numpy(video_frame_embeds[vid_segment_frame_nos, ...])
            vid_segment_embed = torch.mean(vid_segment_frame_embeds, dim=0)
            vid_segments_embed[i,:] = vid_segment_embed

        return {
            'event_sent': text_event_sent_tokens.squeeze(0), 
            'event_start_idx': text_event_start_index,
            'event_len': text_event_len,
            'vid_subevent_idx': vid_subevent_idx,
            'vid_segments_embed': vid_segments_embed, 
            'vid_segments_mask': vid_segments_mask,
            'labels': label
        }

class M2E2HRTestTE2VE(Dataset):
    def __init__(self, cfg):
        self.test_file = cfg.DATASET.TEST.TEST_FILE_PATH
        self.annotation_file = cfg.DATASET.TEST.ANNOT_FILE_PATH
        self.embed_dir = cfg.DATASET.TEST.VIDEO_EMBED_DIR
        self.video_shots_file = cfg.DATASET.TEST.VIDEO_SHOTS
        self.video_events_context_length = cfg.MODEL.VIDEO_EVENTS_CONTEXT_LENGTH
        self.video_embed_dim = cfg.MODEL.VID_EMBED_DIM

        with open(self.test_file) as f:
            self.data = json.load(f)
        with open(self.annotation_file) as f:
            self.annotations = json.load(f)
        with open(self.video_shots_file) as f:
            self.video_shots = json.load(f)

        self.all_event_event_relations = []
        sen_beg_end_pattern = re.compile(r'[\r\n\.]')
        for vid_name, annots in self.annotations.items():
            article = vid_name + '\n' + self.data[vid_name]['article']
            for annot in annots:
                event_start_in_article = annot['text_start']
                event_end_in_article = annot['text_end']
                event_mention = article[event_start_in_article: event_end_in_article]
                
                event_sent_start_match = sen_beg_end_pattern.search(article[event_start_in_article::-1])
                if event_sent_start_match:
                    event_start_char = event_sent_start_match.start() - 1
                else:
                    event_start_char = event_start_in_article
                event_sent_start = event_start_in_article - event_start_char
                event_sent_end_match = sen_beg_end_pattern.search(article, event_end_in_article)
                if event_sent_end_match:
                    event_sent_end = event_sent_end_match.start()
                else:
                    event_sent_end = len(article)
                event_sent = article[event_sent_start: event_sent_end]
                if event_sent[event_start_char-1] != ' ':
                    event_sent = event_sent[:event_start_char] + ' ' + event_sent[event_start_char:]
                    event_start_char = event_start_char + 1

                event_event_info = {
                    'vid': vid_name,
                    'video_segment': (annot['video_start'], annot['video_end']),
                    'event_sent': event_sent,
                    'event_start_char': event_start_char,
                    'label': LABEL_TO_IDX[annot['label']],
                    'article_event_start': annot['text_start'],
                    'article_event_end': annot['text_end'],
                    'event_mention': event_mention
                }
                self.all_event_event_relations.append(event_event_info)


    def __len__(self):
        return len(self.all_event_event_relations)

    def get_shot_frames(self, shot, total_num_of_frames):
        shot_start_time, shot_end_time = shot
        shot_start_frame_no = round(shot_start_time * 3) 
        shot_end_frame_no = round(shot_end_time * 3)
        if shot_end_frame_no > total_num_of_frames:
            shot_end_frame_no = total_num_of_frames
        if shot_start_frame_no == shot_end_frame_no:
            frame_number = shot_end_frame_no if shot_end_frame_no < total_num_of_frames else total_num_of_frames-1
            frame_numbers = [frame_number]
        else:
            frame_numbers = list(range(shot_start_frame_no, shot_end_frame_no))
        return frame_numbers

    def __getitem__(self, idx):
        data_dict = self.all_event_event_relations[idx]
        vid_name = data_dict['vid']
        vid_event_segment = data_dict['video_segment']
        text_event_sent = data_dict['event_sent']
        text_event_start_char = data_dict['event_start_char']
        label = data_dict['label']

        event_trigger_id = len(text_event_sent[:text_event_start_char].split())
        text_event_sent_tokens, text_event_index_list = clip.tokenize(text_event_sent, 
                                                            trigger_index=event_trigger_id,
                                                            truncate=True)
        text_event_start_index, text_event_len = text_event_index_list[0], len(text_event_index_list)

        with open(os.path.join(self.embed_dir, vid_name + '.npy'), 'rb') as f:
            video_frame_embeds = np.load(f)

        # Extracting all video Segments
        vid_segments_embed = torch.zeros(self.video_events_context_length, self.video_embed_dim)
        all_vid_segments_in_vid = self.video_shots[vid_name]
        relevant_vid_segments = all_vid_segments_in_vid
        vid_segments_mask = torch.ones(self.video_events_context_length)
        try:
            vid_subevent_idx = all_vid_segments_in_vid.index({"segment": list(vid_event_segment)})
        except:
            fl = 0
            for vid_seg_tol in range(1,6):
                for cur_idx, segment in enumerate(all_vid_segments_in_vid):
                    segemnt_start, segment_end = segment['segment']
                    if abs(segemnt_start - vid_event_segment[0]) < vid_seg_tol and abs(segment_end - vid_event_segment[1]) < vid_seg_tol:
                        vid_subevent_idx = cur_idx
                        fl=1
                if fl == 1:
                    break
        
        if len(all_vid_segments_in_vid) > self.video_events_context_length:
            if vid_subevent_idx >= self.video_events_context_length:
                relevant_vid_segments_begin_idx = vid_subevent_idx - self.video_events_context_length + 1
                relevant_vid_segments_end_idx = vid_subevent_idx + 1
                vid_subevent_idx = self.video_events_context_length - 1
            else:
                relevant_vid_segments_begin_idx = 0
                relevant_vid_segments_end_idx = self.video_events_context_length
            relevant_vid_segments = all_vid_segments_in_vid[relevant_vid_segments_begin_idx:relevant_vid_segments_end_idx]
        
        vid_segments_mask[:len(relevant_vid_segments)] = 0
        vid_segments_mask = vid_segments_mask.bool() 
            
        for i, vid_segment in enumerate(relevant_vid_segments):
            vid_segment_frame_nos = self.get_shot_frames(vid_segment['segment'], video_frame_embeds.shape[0])
            vid_segment_frame_embeds = torch.from_numpy(video_frame_embeds[vid_segment_frame_nos, ...])
            vid_segment_embed = torch.mean(vid_segment_frame_embeds, dim=0)
            vid_segments_embed[i,:] = vid_segment_embed

        return {
            'event_sent': text_event_sent_tokens.squeeze(0), 
            'event_start_idx': text_event_start_index,
            'event_len': text_event_len,
            'vid_subevent_idx': vid_subevent_idx,
            'vid_segments_embed': vid_segments_embed, 
            'vid_segments_mask': vid_segments_mask,
            'labels': label,
            'vid_names': vid_name,
            'vid_event_segments': vid_event_segment,
            'article_event_start': data_dict['article_event_start'],
            'article_event_end': data_dict['article_event_end'],
            'event_mentions': data_dict['event_mention']
        }

class M2E2HRTestIETE2VE(Dataset):
    def __init__(self, cfg):
        self.test_file = cfg.DATASET.TEST.TEST_FILE_PATH
        self.annotation_file = cfg.DATASET.TEST.ANNOT_FILE_PATH
        self.embed_dir = cfg.DATASET.TEST.VIDEO_EMBED_DIR
        self.video_shots_file = cfg.DATASET.TEST.VIDEO_SHOTS
        self.all_events_file = cfg.DATASET.TEST.ALL_EVENTS_FILE
        self.video_events_context_length = cfg.MODEL.VIDEO_EVENTS_CONTEXT_LENGTH
        self.video_embed_dim = cfg.MODEL.VID_EMBED_DIM

        with open(self.test_file) as f:
            self.data = json.load(f)
        with open(self.annotation_file) as f:
            self.annotations = json.load(f)
        with open(self.video_shots_file) as f:
            self.video_shots = json.load(f)
        with open(self.all_events_file) as f:
            self.all_text_events = json.load(f)

        self.annotated_event_event_relations = {}
        for vid_name, annots in self.annotations.items():
            article = vid_name + '\n' + self.data[vid_name]['article']
            for annot in annots:
                event_start_in_article = annot['text_start']
                event_end_in_article = annot['text_end']
                event_mention = article[event_start_in_article: event_end_in_article]
                
                annotation_key = (
                    vid_name,
                    (annot['video_start'], annot['video_end']),
                    annot['text_start'],
                    annot['text_end'],
                    event_mention,
                )
                self.annotated_event_event_relations[annotation_key] = LABEL_TO_IDX[annot['label']]

        self.all_event_event_relations = []
        for vid_name in self.annotations:
            events = self.all_text_events[vid_name]
            vid_shots = self.video_shots[vid_name]
            for event in events:
                for vid_shot in vid_shots:
                    test_key = (
                        vid_name,
                        tuple(vid_shot['segment']),
                        event['event_char_begin_id_relative_to_article'],
                        event['event_char_end_id_relative_to_article'],
                        event['event_mention']
                    )
                    if test_key in self.annotated_event_event_relations:
                        test_label = self.annotated_event_event_relations.pop(test_key)
                    else:
                        test_label = 0

                    event_event_info = {
                        'vid': test_key[0],
                        'video_segment': test_key[1],
                        'event_sent': event['sent'],
                        'event_start_char': event['event_char_id'],
                        'label': test_label,
                        'article_event_start': test_key[2],
                        'article_event_end': test_key[3],
                        'event_mention': test_key[4]
                    }
                    self.all_event_event_relations.append(event_event_info)
        
        self.annotated_relations_not_in_test = []
        for k, v in self.annotated_event_event_relations.items():
            self.annotated_relations_not_in_test.append({
                'vid_name': k[0],
                'vid_event_segments': k[1],
                'article_event_start': k[2],
                'article_event_end': k[3],
                'event_mention': k[4],
                'label': v
            })

    def __len__(self):
        return len(self.all_event_event_relations)

    def get_shot_frames(self, shot, total_num_of_frames):
        shot_start_time, shot_end_time = shot
        shot_start_frame_no = round(shot_start_time * 3) 
        shot_end_frame_no = round(shot_end_time * 3)
        if shot_end_frame_no > total_num_of_frames:
            shot_end_frame_no = total_num_of_frames
        if shot_start_frame_no == shot_end_frame_no:
            frame_number = shot_end_frame_no if shot_end_frame_no < total_num_of_frames else total_num_of_frames-1
            frame_numbers = [frame_number]
        else:
            frame_numbers = list(range(shot_start_frame_no, shot_end_frame_no))
        return frame_numbers

    def __getitem__(self, idx):
        data_dict = self.all_event_event_relations[idx]
        vid_name = data_dict['vid']
        vid_event_segment = data_dict['video_segment']
        text_event_sent = data_dict['event_sent']
        text_event_start_char = data_dict['event_start_char']
        label = data_dict['label']

        event_trigger_id = len(text_event_sent[:text_event_start_char].split())
        text_event_sent_tokens, text_event_index_list = clip.tokenize(text_event_sent, 
                                                            trigger_index=event_trigger_id,
                                                            truncate=True)
        text_event_start_index, text_event_len = text_event_index_list[0], len(text_event_index_list)

        with open(os.path.join(self.embed_dir, vid_name + '.npy'), 'rb') as f:
            video_frame_embeds = np.load(f)

        # Extracting all video Segments
        vid_segments_embed = torch.zeros(self.video_events_context_length, self.video_embed_dim)
        all_vid_segments_in_vid = self.video_shots[vid_name]
        relevant_vid_segments = all_vid_segments_in_vid
        vid_segments_mask = torch.ones(self.video_events_context_length)
        try:
            vid_subevent_idx = all_vid_segments_in_vid.index({"segment": list(vid_event_segment)})
        except:
            fl = 0
            for vid_seg_tol in range(1,6):
                for cur_idx, segment in enumerate(all_vid_segments_in_vid):
                    segemnt_start, segment_end = segment['segment']
                    if abs(segemnt_start - vid_event_segment[0]) < vid_seg_tol and abs(segment_end - vid_event_segment[1]) < vid_seg_tol:
                        vid_subevent_idx = cur_idx
                        fl=1
                if fl == 1:
                    break
        
        if len(all_vid_segments_in_vid) > self.video_events_context_length:
            if vid_subevent_idx >= self.video_events_context_length:
                relevant_vid_segments_begin_idx = vid_subevent_idx - self.video_events_context_length + 1
                relevant_vid_segments_end_idx = vid_subevent_idx + 1
                vid_subevent_idx = self.video_events_context_length - 1
            else:
                relevant_vid_segments_begin_idx = 0
                relevant_vid_segments_end_idx = self.video_events_context_length
            relevant_vid_segments = all_vid_segments_in_vid[relevant_vid_segments_begin_idx:relevant_vid_segments_end_idx]
        
        vid_segments_mask[:len(relevant_vid_segments)] = 0
        vid_segments_mask = vid_segments_mask.bool() 
            
        for i, vid_segment in enumerate(relevant_vid_segments):
            vid_segment_frame_nos = self.get_shot_frames(vid_segment['segment'], video_frame_embeds.shape[0])
            vid_segment_frame_embeds = torch.from_numpy(video_frame_embeds[vid_segment_frame_nos, ...])
            vid_segment_embed = torch.mean(vid_segment_frame_embeds, dim=0)
            vid_segments_embed[i,:] = vid_segment_embed

        return {
            'event_sent': text_event_sent_tokens.squeeze(0), 
            'event_start_idx': text_event_start_index,
            'event_len': text_event_len,
            'vid_subevent_idx': vid_subevent_idx,
            'vid_segments_embed': vid_segments_embed, 
            'vid_segments_mask': vid_segments_mask,
            'labels': label,
            'vid_names': vid_name,
            'vid_event_segments': vid_event_segment,
            'article_event_start': data_dict['article_event_start'],
            'article_event_end': data_dict['article_event_end'],
            'event_mentions': data_dict['event_mention']
        }


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from configs import get_cfg_defaults

    cfg = get_cfg_defaults()
    # cfg.DATASET.TRAIN.PSEUDO_LABEL_FILE_PATH = '/home/hammad/kairos/Subevent_EventSeg/output/m2e2_rels_finer_modified_grounded.json'
    # cfg.DATASET.TRAIN.VIDEO_EMBED_DIR = '/dvmm-filer3c/projects/kairos-multimodal-event/data/m2e2/frames_embeds/'
    # cfg.DATASET.TRAIN.ALL_EVENTS_FILE = '/home/hammad/kairos/ETypeClus/outputs/m2e2/m2e2_all_events.json'
    # cfg.DATASET.TRAIN.VIDEO_SHOTS = '/dvmm-filer3c/projects/kairos-multimodal-event/data/m2e2/shot_proposals.json'
    # cfg.DATASET.TRAIN.GROUNDED_EVENTS_FILE = '/home/hammad/kairos/ETypeClus/outputs/m2e2/m2e2_all_events_grounded.json'

    # cfg.DATASET.TRAIN.PSEUDO_LABEL_FILE_PATH = '/home/hammad/kairos/data/grounding/total_grounding_data.json'
    # cfg.DATASET.TRAIN.VIDEO_EMBED_DIR = '/home/hammad/kairos/data/videos_embeddings_correct'
    # cfg.DATASET.TRAIN.ALL_EVENTS_FILE =  '/home/hammad/kairos/data/grounding/grounded_vid_events.json'
    # cfg.DATASET.TRAIN.VIDEO_SHOTS = '/home/hammad/kairos/data/videos_shots/total_shots_data.json'
    # cfg.DATASET.TRAIN.GROUNDED_EVENTS_FILE = '/home/hammad/kairos/data/grounding/grounded_events_net.json'

    # checkDataset = M2E2HRTrain(cfg)
    # dataloader = DataLoader(checkDataset, batch_size=1024, shuffle=True)
    # batch = next(iter(dataloader))
    # print('Train DataLoader')
    # print(len(dataloader))
    # print(batch['event_sent'].shape)
    # print(batch['event_start_idx'].shape)
    # print(batch['event_len'].shape)
    # print(batch['vid_subevent_idx'].shape) 
    # print(batch['vid_segments_embed'].shape)    
    # print(batch['vid_segments_mask'].shape)
    # print(batch['labels'].shape)
    # for _ in dataloader:
    #     pass

    cfg.DATASET.VAL.TEST_FILE_PATH = '/home/hammad/kairos/data/test.json'
    cfg.DATASET.VAL.ANNOT_FILE_PATH = '/home/hammad/kairos/data/annotations/validation.json'
    cfg.DATASET.VAL.VIDEO_EMBED_DIR = '/home/hammad/kairos/data/videos_embeddings_test'
    cfg.DATASET.VAL.VIDEO_SHOTS = '/home/hammad/kairos/data/videos_shots_test/shot_proposals_test.json'

    checkValDataset = M2E2HRVal(cfg)
    valDataloader = DataLoader(checkValDataset, batch_size=8, shuffle=True)
    batch = next(iter(valDataloader))
    print('Val DataLoader')
    print(len(valDataloader))
    print(batch['event_sent'].shape)
    print(batch['event_start_idx'].shape)
    print(batch['event_len'].shape)
    print(batch['vid_subevent_idx'].shape) 
    print(batch['vid_segments_embed'].shape)    
    print(batch['vid_segments_mask'].shape)
    print(batch['labels'].shape)
    # for _ in valDataloader:
    #     pass

    cfg.DATASET.TEST.TEST_FILE_PATH = '/home/hammad/kairos/data/test.json'
    cfg.DATASET.TEST.ANNOT_FILE_PATH = '/home/hammad/kairos/data/annotations/annots_refined/test_new.json'
    cfg.DATASET.TEST.VIDEO_EMBED_DIR = '/home/hammad/kairos/data/videos_embeddings_test'
    cfg.DATASET.TEST.VIDEO_SHOTS = '/home/hammad/kairos/data/videos_shots_test/shot_proposals_test.json'
    cfg.DATASET.TEST.ALL_EVENTS_FILE = '/home/hammad/kairos/data/extracted_events_test/events.json'

    checkTestDataset = M2E2HRTestTE2VE(cfg)
    testDataloader = DataLoader(checkTestDataset, batch_size=8, shuffle=True)
    batch = next(iter(testDataloader))
    print('Test DataLoader')
    print(len(testDataloader))
    print(batch['event_sent'].shape)
    print(batch['event_start_idx'].shape)
    print(batch['event_len'].shape)
    print(batch['vid_subevent_idx'].shape) 
    print(batch['vid_segments_embed'].shape)    
    print(batch['vid_segments_mask'].shape)
    print(batch['labels'].shape)
    print(batch['vid_names'])
    print(batch['vid_event_segments'])
    print(batch['article_event_start'])
    print(batch['article_event_end'])
    # for _ in testDataloader:
    #     pass

    checkIETE2VETestDataset = M2E2HRTestIETE2VE(cfg)
    testDataloader = DataLoader(checkIETE2VETestDataset, batch_size=8, shuffle=True)
    batch = next(iter(testDataloader))
    print('Test DataLoader')
    print(len(testDataloader))
    print(batch['event_sent'].shape)
    print(batch['event_start_idx'].shape)
    print(batch['event_len'].shape)
    print(batch['vid_subevent_idx'].shape) 
    print(batch['vid_segments_embed'].shape)    
    print(batch['vid_segments_mask'].shape)
    print(batch['labels'].shape)
    print(batch['vid_names'])
    print(batch['vid_event_segments'])
    print(batch['article_event_start'])
    print(batch['article_event_end'])
    # for _ in testDataloader:
    #     pass