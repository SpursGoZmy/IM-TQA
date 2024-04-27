import logging
import os
import socket
from util.args_help import fill_from_args, fill_from_dict
import ujson as json
import time
import random
import numpy as np
import paddle

logger = logging.getLogger(__name__)

class HypersBase:
    """
    This should be the base hyperparameters class, others should extend this.
    """
    def __init__(self):
        self.local_rank, self.global_rank, self.world_size = -1,0,1
        # required parameters initialized to the datatype
        self.model_type = ''   #'albert'
        self.model_name_or_path = ''   # albert-base-v2
        self.resume_from = ''  # to resume training from a checkpoint
        self.config_name = ''
        self.tokenizer_name = ''
        self.cache_dir = ''
        self.do_lower_case = False
        self.gradient_accumulation_steps = 1  #2
        self.learning_rate = 5e-5   #2e-5
        self.weight_decay = 0.0  # previous default was 0.01  # 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_instances = 0  # previous default was 0.1 of total  # 100000
        self.warmup_fraction=0.1
        self.num_train_epochs = 3   #3
        self.no_cuda = False
        self.n_gpu = 1
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'  # previous default was O2
        self.full_train_batch_size = 8  # previous default was 32    #64
        self.per_gpu_eval_batch_size = 8
        self.output_dir = ''  # where to save model
        self.save_total_limit = 1  # limit to number of checkpoints saved in the output dir
        self.save_steps = 0  # do we save checkpoints every N steps? (TODO: put in terms of hours instead)
        self.use_tensorboard = False
        self.log_on_all_nodes = False
        self.server_ip = ''
        self.server_port = ''
        self.__required_args__ = []   #required args,must specify in the command line

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def set_gradient_accumulation_steps(self):
        """
        when searching for full_train_batch_size in hyperparameter tuning we need to update
        the gradient accumulation steps to stay within GPU memory constraints
        :return:
        """
        if self.n_gpu * self.world_size * self.per_gpu_train_batch_size > self.full_train_batch_size:
            self.per_gpu_train_batch_size = self.full_train_batch_size // (self.n_gpu * self.world_size)
            self.gradient_accumulation_steps = 1
        else:
            self.gradient_accumulation_steps = self.full_train_batch_size // \
                                               (self.n_gpu * self.world_size * self.per_gpu_train_batch_size)

    def _basic_post_init(self):
        # Setup CUDA, GPU

        self.device = paddle.device.get_device()  #"gpu:0" or "cpu"
        if self.n_gpu > 0:
            # 64 /(1*1*4)
            self.per_gpu_train_batch_size = self.full_train_batch_size // \
                                            (self.n_gpu * self.world_size * self.gradient_accumulation_steps)
        else:
            self.per_gpu_train_batch_size = self.full_train_batch_size // self.gradient_accumulation_steps

        self.stop_time = None
        if 'TIME_LIMIT_MINS' in os.environ:
            self.stop_time = time.time() + 60 * (int(os.environ['TIME_LIMIT_MINS']) - 5)

    def _post_init(self):
        self._basic_post_init()

        self._setup_logging()


        logger.warning(
            "On %s, Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            socket.gethostname(),
            self.local_rank,
            self.device,
            self.n_gpu,
            bool(self.local_rank != -1),
            self.fp16,
        )
        logger.info(f'hypers:\n{self}')

    def _setup_logging(self):
        # force our logging style
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if self.log_on_all_nodes:
            grank = self.global_rank
            class HostnameFilter(logging.Filter):
                hostname = socket.gethostname()
                if '.' in hostname:
                    hostname = hostname[0:hostname.find('.')]  # the first part of the hostname

                def filter(self, record):
                    record.hostname = HostnameFilter.hostname
                    record.global_rank = grank
                    return True

            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.addFilter(HostnameFilter())
            format = logging.Formatter('%(hostname)s[%(global_rank)d] %(filename)s:%(lineno)d - %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')
            handler.setFormatter(format)
            logging.getLogger('').addHandler(handler)
        else:
            logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
        if self.global_rank != 0 and not self.log_on_all_nodes:
            try:
                logging.getLogger().setLevel(logging.WARNING)
            except:
                pass

    def to_dict(self):
        d = self.__dict__.copy()
        del d['device']
        return d

    def from_dict(self, a_dict):
        fill_from_dict(self, a_dict)
        self._basic_post_init()  # setup device and per_gpu_batch_size
        return self

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def fill_from_args(self):
        fill_from_args(self)
        self._post_init()
        return self


def fill_hypers_from_args(hypers: HypersBase):
    fill_from_args(hypers)
    
    hypers._post_init()
