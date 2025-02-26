import clip
import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

def get_clip_model(gpu_num):
    device = 'cuda:{}'.format(gpu_num)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess 

def get_frame_embedding(vid_names, vid_frames_dir, vid_embed_dir, gpu_num, model, preprocess, batch_size=512):
    device = 'cuda:{}'.format(gpu_num)

    for vid_name in tqdm(vid_names):
        vid_path = os.path.join(vid_frames_dir, vid_name)
        try:
            frame_names = os.listdir(vid_path)
        except:
            print('Frames not extracted yet:', vid_name)
            continue
        frame_names_sorted = sorted(frame_names, key=lambda x: int(x.split('.jpg')[0]))

        images = []
        for frame in frame_names_sorted:
            frame_path = os.path.join(vid_path, frame)
            image = preprocess(Image.open(frame_path).convert("RGB"))
            images.append(image)

        image_input = torch.tensor(np.stack(images)).to(device)

        # TODO: This was wrong for all image lens > 512 and < 1024 starting 512 needs to be deleted
        # total_vid_div = int(len(images) / batch_size)
        # image_inputs = []
        # i = 0
        # for i in range(total_vid_div):
        #     image_inputs.append(image_input[i*batch_size: (i+1)*batch_size])
        # if i == 0:
        #     image_inputs.append(image_input[i*batch_size:])
        # elif (i+1)*batch_size < len(images):
        #     image_inputs.append(image_input[(i+1)*batch_size:])

        total_vid_div = int(image_input.shape[0] / batch_size)
        image_inputs = []
        i = 0
        image_inputs.append(image_input[0:batch_size])
        for i in range(1,total_vid_div):
            image_inputs.append(image_input[i*batch_size: (i+1)*batch_size])
        if (i+1)*batch_size < image_input.shape[0]:
            image_inputs.append(image_input[(i+1)*batch_size:])

        image_features = []
        with torch.no_grad():
            for i in range(len(image_inputs)):
                image_features.append(model.encode_image(image_inputs[i]))

        # image_input = torch.tensor(np.stack(images))
        # image_features = []
        # with torch.no_grad():
        #     for images_batch in tqdm(DataLoader(image_input, batch_size=batch_size)):
        #         image_features.append(model.encode_image(images_batch.to(device)))
        
        image_features = torch.cat(image_features, dim=0)

        vid_embed_path = os.path.join(vid_embed_dir, vid_name + '.npy')
        with open(vid_embed_path, 'wb') as f:
            np.save(f, image_features.cpu().numpy())

def get_image_inputs(image_input, batch_size=512):
    total_vid_div = int(image_input.shape[0] / batch_size)
    image_inputs = []
    i = 0
    image_inputs.append(image_input[0:batch_size].shape)
    for i in range(1,total_vid_div):
        image_inputs.append(image_input[i*batch_size: (i+1)*batch_size].shape)
    if (i+1)*batch_size < image_input.shape[0]:
        image_inputs.append(image_input[(i+1)*batch_size:].shape)
    return image_inputs

if __name__ == '__main__':
    image_input = np.random.randint(128, size=(128,3,128,128))
    c = get_image_inputs(image_input)