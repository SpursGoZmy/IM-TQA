import numpy as np
import logging

try:
    from apex import amp
except ModuleNotFoundError:
    pass
from paddle_util.hypers_base import HypersBase
from util.reporting import Reporting
#from transformers import (
#    AdamW,
#    get_linear_schedule_with_warmup,
#)
import paddle
from paddlenlp.transformers import LinearDecayWithWarmup
from paddle.optimizer import AdamW

logger = logging.getLogger(__name__)


def set_seed(args: HypersBase):
    args.set_seed()


class LossHistory:
    def __init__(self, one_epoch_batch_count, *, loss_points_per_epoch=10, recency_weight=0.001):
        self.avg_loss = 0
        self.batch_count = 0
        self.recency_weight = recency_weight
        self.loss_history = []
        self.record_loss_every = max(1, one_epoch_batch_count // loss_points_per_epoch)
        # one_epoch_batch_count=args.train_instances //(args.full_train_batch_size // args.gradient_accumulation_steps)
        #                        =#360664 //(64 //4)=360664 //16

    def note_loss(self, loss_val, *, hypers: HypersBase = None):
        self.batch_count += 1
        rweight = max(self.recency_weight, 1.0 / self.batch_count)
        self.avg_loss = (1.0 - rweight) * self.avg_loss + rweight * loss_val
        if self.batch_count % self.record_loss_every == 0:
            
            self.loss_history.append(self.avg_loss)
            logger.info(f'loss point {self.batch_count//self.record_loss_every} = {self.avg_loss}')
            return True
        return False


class TransformerOptimize:
    """
    Collects standard steps to train transformer
    call step_loss after computing each loss
    """
    def __init__(self, hypers: HypersBase, num_instances_to_train_over: int, model):
        self.step = 0
        self.global_step = 0
        self.hypers = hypers
        self.model = model   #ErnieForSequenceClassification
        instances_per_step = hypers.full_train_batch_size // hypers.gradient_accumulation_steps  # 64//4=16, batch_size
        self.reporting = Reporting(recency_weight=0.0001 * instances_per_step)
        args = self.hypers

        self.t_total = num_instances_to_train_over // args.full_train_batch_size  # 360664*3 //64, steps of optimizer
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "norm"]
        optimizer_grouped_parameters = [
            {  # params with weight_decay
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },  #
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        #if hasattr(args, 'warmup_fraction') and args.warmup_fraction > 0 and args.warmup_instances <= 0:
        #    warmup_instances = args.warmup_fraction * num_instances_to_train_over
        #if warmup_instances < 0:
        #    warmup_instances = 0
        assert args.warmup_fraction > 0
        logger.info("warmup fraction: %s"%(str(args.warmup_fraction)))
        self.scheduler = LinearDecayWithWarmup(learning_rate=args.learning_rate, total_steps =self.t_total, 
                                                warmup=float(args.warmup_fraction))
        if self.hypers.max_grad_norm > 0:
            self.grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=self.hypers.max_grad_norm)
        else:
            self.grad_clip = None

        self.optimizer = AdamW(learning_rate=self.scheduler,parameters=optimizer_grouped_parameters,grad_clip=self.grad_clip,epsilon=args.adam_epsilon)
        #self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        #self.scheduler = get_linear_schedule_with_warmup(
        #    self.optimizer, num_warmup_steps=warmup_instances // args.full_train_batch_size,
        #    num_training_steps=self.t_total
        #)

        # Check if saved optimizer or scheduler states exist
        

        logger.info("***** Running training *****")
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.full_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

    def should_continue(self):
        return self.global_step < self.t_total

    def backward_on_loss(self, loss, **moving_averages):
        if self.hypers.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        loss_val = loss.item()
        if self.hypers.gradient_accumulation_steps > 1:
            loss = loss / self.hypers.gradient_accumulation_steps
        if self.hypers.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.reporting.moving_averages(loss=loss_val, **moving_averages)   
        return loss_val

    def optimizer_step(self):
        if self.global_step >= self.t_total:
            logger.warning(f'Warning, exceeded total steps! {self.global_step} step of {self.t_total}')
            return False
        if (self.step + 1) % self.hypers.gradient_accumulation_steps == 0:
                
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hypers.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.optimizer.clear_grad()  
            self.global_step += 1
        self.step += 1
        return True

    def step_loss(self, loss, **moving_averages):
        loss_val = self.backward_on_loss(loss, **moving_averages)
        if self.optimizer_step():
            return loss_val
        else:
            return None
