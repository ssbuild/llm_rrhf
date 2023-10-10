# -*- coding: utf-8 -*-
# @Author  : ssbuild
# @Time    : 2023/9/22 9:03
import copy
import json
import os
import random
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper
from aigc_zoo.model_zoo.auto.dpo_model import PetlArguments,LoraConfig
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser, PretrainedConfig
from data_factory.data_processer import DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, CorpusPreprocess, TokenIdsMaker
from config import *
from torch.nn import functional as F


data_conf = {
    "src_max_length": None,
    "dst_max_length": None,
}


def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper_Base(DataHelper):
    index = 1

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper_Base, self).__init__(*args, **kwargs)

    def load_tokenizer_and_config(self, *args, **kwargs):
        ret = super().load_tokenizer_and_config(*args, **kwargs)
        self._preprocess_tokenizer_config()
        return ret
    def _preprocess_tokenizer_config(self):
        model_args = self.model_args
        tokenizer = self.tokenizer
        config = self.config
        if "llama" in model_args.model_name_or_path.lower() and tokenizer.bos_token_id != DEFAULT_BOS_TOKEN:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
            if tokenizer.pad_token_id is None or tokenizer.pad_token_id == -1:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({
                "pad_token": tokenizer.eos_token,
            })
        if config.decoder_start_token_id is None:
            config.decoder_start_token_id = config.bos_token_id
        assert config.decoder_start_token_id == config.bos_token_id



    def on_get_labels(self, files: typing.List[str]):
        D = ['score']
        label2id = {label: i for i, label in enumerate(D)}
        id2label = {i: label for i, label in enumerate(D)}
        return label2id, id2label


    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer

        pair_data = data

        data_conf["sptoken"] = [config.bos_token_id]
        d = TokenIdsMaker.process(pair_data, tokenizer, max_seq_length,**data_conf)
        if self.index < 3:
            print(d)
        return d

    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        tokenizer = self.tokenizer
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            d = CorpusPreprocess.process(tokenizer,lines)
            D.extend(d)
        return D

    def collate_fn(self, batch):
        o = {k: [] for k in batch[0].keys()}
        for i, b in enumerate(batch):
            for k in b:
                x = torch.tensor(b[k])
                if x.dim() > 1:
                    x = torch.transpose(x, 1, 0)
                o[k].append(x)

        pad_token_id = self.tokenizer.pad_token_id
        for k in batch[0].keys():
            pad_val = -100 if 'label' in k else pad_token_id
            val = o[k]
            if pad_val is not None:
                val = torch.nn.utils.rnn.pad_sequence(
                    val, batch_first=True, padding_value=pad_val
                )
            else:
                val = torch.stack(val)

            if val.dim() > 2:
                val = torch.transpose(val, 2, 1)
            val = torch.reshape(val, (-1, *val.size()[2:]))
            o[k] = val
        return o


    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = None
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)
