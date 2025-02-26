import torch
import logging
import os
import torch.nn as nn


class Checkpointer(object):
    def __init__(
        self,
        model,
        device,
        optimizer=None,
        save_dir="",
        logger=None,
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger('M2E2HR.checkpointer')
        self.logger = logger

    def save(self, name, is_best, **kwargs):
        if not self.save_dir:
            self.logger.info("No save dir to save checkpoint")
            return

        data = {}
        if isinstance(self.model, nn.DataParallel):
            data["model"] = self.model.module.state_dict()
        else:
            data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)
        if is_best:
            self.tag_best_checkpoint(save_file)

    def load(self, f=None, use_latest=True, load_trainer_state=True, use_best=False):
        if self.has_checkpoint() and use_latest:
            # So not using best
            assert(use_best == False)
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        elif self.has_best_checkpoint() and use_best:
            # So not using latest
            assert(use_latest == False)
            f = self.get_best_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if load_trainer_state and "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def has_best_checkpoint(self):
        save_file = os.path.join(self.save_dir, "best_checkpoint")
        return os.path.exists(save_file)

    def get_best_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "best_checkpoint")
        try:
            with open(save_file, "r") as f:
                best_saved = f.read()
                best_saved = best_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            best_saved = ""
        return best_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def tag_best_checkpoint(self, best_filename):
        save_file = os.path.join(self.save_dir, "best_checkpoint")
        with open(save_file, "w") as f:
            f.write(best_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=self.device)

    def _load_model(self, checkpoint):
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint.pop("model"))
        else: 
            self.model.load_state_dict(checkpoint.pop("model"))