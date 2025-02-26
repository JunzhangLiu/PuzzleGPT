import torch
from torch.utils.data import DataLoader
from dataset.dataset import M2E2HRTrain, M2E2HRVal, M2E2HRTestTE2VE, M2E2HRTestIETE2VE
from torch.utils.data import WeightedRandomSampler

def make_data_loader(cfg, type='train'):
    if type == 'train':
        dataset = M2E2HRTrain(cfg)
        is_shuffle = True
        pin_memory = True
        sampler = None
        if cfg.DATASET.TRAIN.USE_WEIGHTED_SAMPLING:
            target_list = dataset.target_class_list
            class_weights = 1 / torch.tensor(dataset.target_class_count, dtype=torch.float)
            class_weights_all = class_weights[target_list]
            weighted_sampler = WeightedRandomSampler(
                                    weights=class_weights_all,
                                    num_samples=len(class_weights_all),
                                    replacement=True
                                )
            sampler = weighted_sampler
            is_shuffle = False

    elif type == 'val':
        dataset = M2E2HRVal(cfg)
        is_shuffle = False
        pin_memory = False
        sampler = None

    elif type == 'testTE2VE':
        dataset = M2E2HRTestTE2VE(cfg)
        is_shuffle = False
        pin_memory = False
        sampler = None

    elif type == 'testIETE2VE':
        dataset = M2E2HRTestIETE2VE(cfg)
        is_shuffle = False
        pin_memory = False
        sampler = None

    else:
        raise ValueError('Dataloader type should either be train, val or test')

    dataloader = DataLoader(dataset, shuffle=is_shuffle, 
                            batch_size=cfg.DATALOADER.BATCH_SIZE,
                            sampler = sampler,
                            num_workers=cfg.DATALOADER.NUM_WORKERS, 
                            pin_memory=pin_memory)

    return dataloader