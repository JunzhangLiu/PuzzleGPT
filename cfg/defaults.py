from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
_C.MODEL.TEXT_EMBED_DIM = 512
_C.MODEL.VID_EMBED_DIM = 512
_C.MODEL.PR_FOR_CLASS = ['No Relations', 'Identical', 'Hierarchical']
_C.MODEL.MODEL_SCORE_METRIC_CLASS = ['Identical', 'Hierarchical']
_C.MODEL.NAME = "BasicMLPModel"
_C.MODEL.DEVICE = "cuda"
_C.MODEL.LOAD_TRAINER_STATE = True
_C.MODEL.TEXT_ENCODE_ATTN_SPAN = 10
# Can be None, all, half or random
_C.MODEL.NEG_SAMPLING_TYPE = None
_C.MODEL.LOSS_WEIGHT = None
_C.MODEL.VIDEO_EVENTS_CONTEXT_LENGTH = 77
_C.MODEL.ADD_CONTEXTUAL_TRANSFORMER = False
_C.MODEL.CONTEXTUAL_TRANSFORMER_LAYERS = 1
_C.MODEL.CONTEXTUAL_TRANSFORMER_HEADS = 8
_C.MODEL.ADD_KB_COMMONSENSE = False
_C.MODEL.KB_CS_CKPT_PATH = '/home/hammad/kairos/tools/kb_cs_best_ckpt'
_C.MODEL.KB_CS_OUT_DIM = 512
_C.MODEL.KB_CS_OUT_DIM = 512
_C.MODEL.EVENTS_ELEMENTWISE_SUB = False
_C.MODEL.EVENTS_ELEMENTWISE_MUL = False
_C.MODEL.MIL_FACTOR = None
_C.MODEL.USE_FOCAL_LOSS = False
_C.MODEL.FOCAL_LOSS_GAMMA = 2


# -----------------------------------------------------------------------------
# Train Parameters
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.LR = 1e-3
_C.SOLVER.NUM_EPOCHS = 40
_C.SOLVER.LOG_PERIOD = 20
_C.SOLVER.TEST_PERIOD = 1000
# Checkpoint period should be a multiple of test period so that best model checkpoints are saved correctly
_C.SOLVER.CHECKPOINT_PERIOD = 1000

# -----------------------------------------------------------------------------
# Dataloader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 4 
_C.DATALOADER.NUM_WORKERS = 4

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.PSEUDO_LABEL_FILE_PATH = '/home/hammad/kairos/Subevent_EventSeg/output/m2e2_rels_finer_modified.json'
_C.DATASET.TRAIN.VIDEO_EMBED_DIR = '/dvmm-filer3c/projects/kairos-multimodal-event/data/m2e2/frames_embeds/'
_C.DATASET.TRAIN.ALL_EVENTS_FILE = None
_C.DATASET.TRAIN.VIDEO_SHOTS = None
_C.DATASET.TRAIN.GROUNDED_EVENTS_FILE = None
# When using weighted sampler specify loss weights to be None
_C.DATASET.TRAIN.USE_WEIGHTED_SAMPLING = False
# Always use weighted sampler with downsample norels otherwise loss weights are disrupted
_C.DATASET.TRAIN.DOWNSAMPLE_NORELS = None

_C.DATASET.VAL = CN()
_C.DATASET.VAL.TEST_FILE_PATH = '/home/hammad/kairos/data/test.json'
_C.DATASET.VAL.ANNOT_FILE_PATH = '/home/lovish/github-event-relation/multi-modal-event/annotations/saved_annotations_transformed/validation.json'
_C.DATASET.VAL.VIDEO_EMBED_DIR = '/home/hammad/kairos/data/videos_embeddings_test/'
_C.DATASET.VAL.VIDEO_SHOTS = None

_C.DATASET.TEST = CN()
_C.DATASET.TEST.TEST_FILE_PATH = '/home/sssak/viper/hammad_data/test.json'
_C.DATASET.TEST.ANNOT_FILE_PATH = '/home/sssak/viper/hammad_data/validation.json'
_C.DATASET.TEST.VIDEO_EMBED_DIR = '/home/sssak/viper/hammad_data/videos_embeddings_test'
_C.DATASET.TEST.VIDEO_SHOTS ='/home/sssak/viper/hammad_data/shot_proposals_test.json'
_C.DATASET.TEST.ALL_EVENTS_FILE = '/home/sssak/viper/hammad_data/video'

# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.TYPES = ['TE2VE']
_C.TEST.PRUNE_IDENTICAL_WITH_CLIP = False
_C.TEST.PRUNE_IDENTICAL_WITH_CLIP_THRESH = 5.0

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""