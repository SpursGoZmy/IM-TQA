import pickle
import logging
from dataloader.distloader_seq_pair import SeqPairDataloader, standard_json_mapper
from paddle_util.hypers_base import HypersBase, fill_hypers_from_args
from paddle_util.transformer_optimize import TransformerOptimize, set_seed, LossHistory
import numpy as np
import time
import os
import ujson as json
from paddle_util.validation import score_metrics, multiclass_score_metrics
import paddle
from paddlenlp.transformers import ErnieForSequenceClassification,ErnieTokenizer ,ErnieTinyTokenizer,BertForSequenceClassification,BertTokenizer
logger = logging.getLogger(__name__)

class SeqPairHypers(HypersBase):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.max_seq_length = 128
        self.num_labels = 2
        self.single_sequence = False
        self.additional_special_tokens = ''
        self.is_separate = False
        # for reasonable values see the various params.json under
        #    https://github.com/peterliht/knowledge-distillation-pytorch
        self.kd_alpha = 0.9
        self.kd_temperature = 10.0

class SeqPairArgs(SeqPairHypers):
    def __init__(self):
        super().__init__()
        self.train_dir = ''
        self.dev_dir = ''
        self.train_instances = 0  # we need to know the total number of training instances (should just be total line count)
        self.hyper_tune = 0  # number of trials to search hyperparameters
        self.prune_after = 5
        self.save_per_epoch = False
        self.teacher_labels = ''  # the labels from the teacher for the train_dir dataset

def load_chinese_models(hypers: SeqPairHypers):
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[hypers.model_type.lower()]
    logger.info("model type:%s"%(hypers.model_type.lower()))
    assert hypers.model_type in ["ernie-tiny","ernie","bert-base-chinese"]

    if not hypers.resume_from:
            logger.info("load pretrained %s model"%(hypers.model_type))
            if hypers.model_type =="ernie-tiny":
                model = ErnieForSequenceClassification.from_pretrained('ernie-tiny',num_classes=2)
                tokenizer = ErnieTinyTokenizer.from_pretrained('ernie-tiny')
            elif hypers.model_type =="ernie":
                model = ErnieForSequenceClassification.from_pretrained('ernie-1.0-base-zh',num_classes=2)
                tokenizer = ErnieTokenizer.from_pretrained('ernie-1.0-base-zh')
                tokenizer.add_special_tokens({'additional_special_tokens': ['*']})  # add * to the vocabulary
                test_input = tokenizer("*")
                assert test_input['input_ids'][1] != tokenizer.unk_token_id

            else:
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_classes=2)
  
    else:
            logger.info("load saved model from %s "%(hypers.resume_from))
            if hypers.model_type =="ernie-tiny":
                model = ErnieForSequenceClassification.from_pretrained(hypers.resume_from)
                tokenizer = ErnieTinyTokenizer.from_pretrained(hypers.resume_from)
            elif hypers.model_type =="ernie":
                model = ErnieForSequenceClassification.from_pretrained(hypers.resume_from)
                tokenizer = ErnieTokenizer.from_pretrained(hypers.resume_from)
            else:
                tokenizer = BertTokenizer.from_pretrained(hypers.resume_from)
                model = BertForSequenceClassification.from_pretrained(hypers.resume_from)

    return model, tokenizer

def save(hypers: SeqPairHypers, model, *, save_dir=None, tokenizer=None):
    if save_dir is None:
            save_dir = hypers.output_dir
    # Create output directory if needed
    os.makedirs(save_dir, exist_ok=True)
    logger.info("Saving model checkpoint to %s", save_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model.save_pretrained(save_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

def train(args: SeqPairArgs, train_dataloader: SeqPairDataloader, model):
    # transformer_optimize
    instances_to_train_over = args.train_instances * args.num_train_epochs
    toptimizer = TransformerOptimize(args, instances_to_train_over, model)

    # Train!
    logger.info(" Num Epochs = %d", args.num_train_epochs)
    set_seed(args)
    loss_history = LossHistory(args.train_instances //
                               (args.full_train_batch_size // args.gradient_accumulation_steps))
    criterion = paddle.nn.loss.CrossEntropyLoss()
    epoch = 0
    flag = 1
    while(True):
        start_time = time.time()
        logger.info(f'On epoch {epoch}')
        random_seed_for_shuffle_data = epoch * args.seed
        train_dataloader.set_random_to_shuffle_data(random_seed_for_shuffle_data)
        if train_dataloader is None:
            break
        for batch in train_dataloader:
            ids,input_ids_batch,token_type_ids_batch,labels=batch
            if flag < 3:
                # show some examples for checking
                print("in train(), input_ids.shape:")
                print(input_ids_batch.shape)
                flag+=1
            logits = toptimizer.model(input_ids_batch,token_type_ids_batch)  # [batch, num_label]
            loss = criterion(logits, labels)

            loss_val = toptimizer.step_loss(loss)   # loss_val=loss.item()
            if loss_val is None:
                return loss_history.loss_history
            loss_history.note_loss(loss_val, hypers=args)
        epoch += 1
        logger.info(f'one group of train files took {(time.time()-start_time)/60} minutes')

    return loss_history.loss_history  #loss_historcy is a list

@paddle.no_grad()
def evaluate(args: SeqPairHypers, eval_dataloader: SeqPairDataloader, model):
    start_time = time.time()
    eval_loss = 0.0
    nb_eval_count = 0
    preds = None
    labels = None
    criterion = paddle.nn.loss.CrossEntropyLoss()
    flag = 1
    eval_dataloader.set_random_to_shuffle_data(1234)
    for batch in eval_dataloader:
            ids,input_ids_batch,token_type_ids_batch,input_labels=batch
            logits=model(input_ids_batch,token_type_ids_batch)
            tmp_eval_loss = criterion(logits, input_labels)
          
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_count += 1
            logits = logits.detach().cpu().numpy()   #[batch,num_label]
            input_labels = input_labels.detach().cpu().numpy()
            if flag == 1:
                print("in evaluate(), logits.shape: ")
                print(logits.shape)
                print("in evaluate(), input_labels.shape:")
                print(input_labels.shape)
                flag=0
            if preds is None:
                preds = logits
                labels = input_labels
            else:
                preds = np.append(preds, logits, axis=0) # [num_inst,2]
                labels = np.append(labels, input_labels, axis=0)  #[ num_inst]
    eval_loss = eval_loss / nb_eval_count
    logger.info(f'went through {nb_eval_count} batches of evaluate')
    # score
    if args.num_labels == 2:
        results = score_metrics(preds[:, 1], np.argmax(preds, axis=1), labels)
    else:
        results = multiclass_score_metrics(preds, np.argmax(preds, axis=1), labels)
    results['eval_loss'] = eval_loss

    logger.info("***** Eval results *****")
    logger.info(f'took {(time.time()-start_time)/60} mins')
    return results

def main():
    logger.info("Read command line params")
    args = SeqPairArgs()
    fill_hypers_from_args(args)
    
    # load model and tokenizer
    logger.info("Load model")
    model, tokenizer = load_chinese_models(args)
    logger.info("Successfully load model and tokenizer")
    
    logger.info("Load train dataloader for the constructed row and column representations")
    # Training
    loss_history = None
    if args.train_dir:
        assert args.train_instances > 0
        
        train_dataloader = SeqPairDataloader(args, args.per_gpu_train_batch_size, tokenizer, args.train_dir,
                                              json_mapper=standard_json_mapper)
        logger.info("Start training")
        loss_history = train(args, train_dataloader, model)
        # save model
        save(args, model, tokenizer=tokenizer)

    # Evaluation
    logger.info("Start evaluation")
    if args.dev_dir:
        args.resume_from = args.output_dir
        model, tokenizer = load_chinese_models(args)

        eval_dataloader = SeqPairDataloader(args, args.per_gpu_eval_batch_size, tokenizer, args.dev_dir,
                                             json_mapper=standard_json_mapper)
        results = evaluate(args, eval_dataloader, model)
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if args.global_rank == 0:
            # save the evaluation results like F1 and Accuracy
            output_eval_file = os.path.join(args.output_dir, f"eval_results.json")
            with open(output_eval_file, "w") as writer:
                results['hypers'] = args.to_dict()
                writer.write(json.dumps(results, indent=2) + '\n')

    logger.info(f'loss_history = {loss_history}')

if __name__ == "__main__":
    main()

