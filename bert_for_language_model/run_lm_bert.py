# -*- coding: utf-8 -*-
# @DateTime :2020/12/22 下午8:51
# @Author   :zhi.liu

# ------------------------------------------------------------------------------
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
        Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={'help': 'The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.'})
    model_type: Optional[str] = field(default=None, metadata={'help': "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'where do you want to store the pretrained models downloaded from s3'})


@dataclass
class DataTrainingArguments:
    """
       Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data_file: Optional[str] = field(default=None, metadata={'help': 'The input training data file (a text file).'})
    eval_data_file: Optional[str] = field(default=None, metadata={'help': 'An optioinal input evaluation data file to evaluation data file to evaluate the perplexity on (a text file).'})
    line_by_line: bool = field(default=False, metadata={'help': 'Whether distinct lines of text in the dataset are to be handled as distinct sequences.'})
    mlm: bool = field(default=False, metadata={'help': 'Train with masked-language modeling loss instead of language modeling.'})
    mlm_probability: float = field(default=0.15, metadata={'help': 'Ratio of tokens to mask for masked language modeling loss'})
    plm_probability: float = field(default=1 / 6, metadata={'help': 'Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling.'})
    max_span_length: int = field(default=5, metadata={'help': 'Maximum length of a span of masked tokens for permutation language modeling.'})
    block_size: int = field(default=1, metadata={
        'help': 'Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evalution sets'})


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError('Cannot do evaluation without an evalution data file. Either supply a file to -eval_data_file or remove the --do_eval argument.')
    if (os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir):
        raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')

    # setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evalution parameter %s", training_args)

    # set seed
    set_seed(training_args.seed)

    # Distributed training
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiation a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir)
    else:
        logger.info("Training new model from scratch.")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ['bert', 'roberta', 'distilbert', 'camembert'] and not data_args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the"
            "--mlm flag (masked language modeling)."
        )
    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # get datasets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability)

    # initialize our trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True
    )

    # Training
    if training_args.do_train:
        model_path = (model_args.model_name_or_path if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path) else None)
        trainer.train(model_path=model_path)
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evalution
    results = {}
    if training_args.do_eval:
        logger.info(f"*** Evaluae ***")
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output['eval_loss'])
        result = {'perplexity': perplexity}

        output_eval_file = os.path.join(training_args.output_dir, 'eval_results_lm.txt')
        if trainer.is_world_master():
            with open(output_eval_file, 'w') as writer:
                logger.info("***** Eval results ******")
                for key in sorted(result.keys()):
                    logger.info(f"   {key} = {result[key]}")
                    writer.write(f"{key} = {result[key]}\n")
        results.update(result)
    return results


def _mp_fn(index):
    # for xla_span
    main()


if __name__ == '__main__':
    main()

''' RUN
    python run_lm_bert.py     --output_dir=output     --model_type=bert     --model_name_or_path=bert-base-chinese     --do_train     --train_data_file=train.txt     --do_eval     --eval_data_file=eval.txt     --mlm --per_device_train_batch_size=4
''''
