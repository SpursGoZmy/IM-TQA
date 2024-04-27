import logging
from dataloader.distloader_seq_pair import SeqPairDataloader, standard_json_mapper
from train_RCI_model import SeqPairHypers, load_chinese_models
import os
import ujson as json
from util.line_corpus import write_open
import paddle
logger = logging.getLogger(__name__)


class SeqPairArgs(SeqPairHypers):
    """
    Arguments and hypers
    """
    def __init__(self):
        super().__init__()
        self.input_dir = ''

@paddle.no_grad()
def evaluate(args: SeqPairHypers, eval_dataloder: SeqPairDataloader, model):
    model.eval()
    flag = 1
    with write_open(os.path.join(args.output_dir, f'results{args.global_rank}.jsonl.gz')) as f:
        for batch in eval_dataloder:
            ids, input_ids_batch, token_type_ids_batch, labels = batch
            logits = model(input_ids_batch, token_type_ids_batch).detach().cpu().numpy()
            
            if flag == 1:
                print("show one inference result")
                print("len(logits):",len(logits))
                print("logits:")
                print(logits)
                flag = 0
            for id, pred in zip(ids, logits):
                assert type(id) == str
                pred = [float(p) for p in pred]
                assert len(pred) == args.num_labels
                assert all([type(p) == float for p in pred])
                f.write(json.dumps({'id': id, 'predictions': pred})+'\n')


def main():
    args = SeqPairArgs().fill_from_args()
    args.set_seed()
    #logger.info("loading saved model from %s"%(args.model_name_or_path))
    args.resume_from = args.model_name_or_path
    # load model and tokenizer
    model, tokenizer = load_chinese_models(args)   #loading saved model

    eval_dataloader = SeqPairDataloader(args, args.per_gpu_eval_batch_size, tokenizer, args.input_dir,
                                        json_mapper=standard_json_mapper)
    evaluate(args, eval_dataloader, model)

if __name__ == "__main__":
    main()
