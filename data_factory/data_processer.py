# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 8:32
import copy
import json
import typing
import numpy as np
from transformers import PreTrainedTokenizer

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"



class CorpusPreprocess:

    @classmethod
    def process(cls, tokenizer, lines):
        D = []
        for i, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            prompt = jd['prompt']
            response = jd['response']
            score = jd['scores']
            D.append((prompt, response, score))
        return D



class TokenIdsMaker:
    @classmethod
    def stop_response(cls,res):
        stops = ['\n\nHuman:', '\n\nAssistant:', '\n\nhuman:', '\n\nassistant:']
        for stop in stops:
            if res.find(stop) >= 0:
                res = res[:res.find(stop)].strip()
        return res

    @classmethod
    def trunction_ids(cls,a_ids: typing.List,b_ids: typing.List,max_seq_length,sptoken):
        while len(a_ids) + len(b_ids) > max_seq_length - len(sptoken) - 1:
            if len(a_ids) > len(b_ids):
                a_ids.pop(0)
            else:
                b_ids.pop(-1)

    @classmethod
    def process(cls, pair_data, tokenizer: PreTrainedTokenizer, max_seq_length: int, sptoken, src_max_length,
                dst_max_length):

        ds = []
        prompt, responses, scores = pair_data

        a_ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        if src_max_length is not None and src_max_length > 0:
            a_ids = a_ids[:src_max_length]
        maxlen = 0
        for score,response in zip(scores,responses):
            response = cls.stop_response(response)
            b_ids = tokenizer.encode(response, truncation=True, max_length=max_seq_length, add_special_tokens=False)
            if dst_max_length is not None and dst_max_length > 0:
                b_ids = b_ids[:dst_max_length]

            a_ids_ = copy.deepcopy(a_ids)
            cls.trunction_ids(a_ids_, b_ids, max_seq_length, sptoken)
            input_ids = a_ids_ + b_ids + [tokenizer.eos_token_id]
            attention_mask = [1] * len(input_ids)
            pos = len(a_ids_) + len(sptoken)
            labels = [-100] * pos + input_ids[pos:]
            maxlen = max(maxlen,len(input_ids))
            ds.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "score": score
            })

        return {
            "input_ids": np.stack([np.asarray(_["input_ids"] + [0] * (maxlen - len(_["input_ids"])), dtype=np.int32) for _ in ds]),
            "attention_mask": np.stack(
                [ np.asarray(_[ "attention_mask" ] + [ 0 ] * (maxlen - len(_[ "attention_mask" ])), dtype=np.int32) for _ in
                  ds ]),
            "labels": np.stack([np.asarray(_["labels"] + [-100] * (maxlen - len(_["labels"])), dtype=np.int32) for _ in ds]),
            "scores": np.stack([np.asarray(_["score"], dtype=np.float32) for _ in ds]),
        }







class TokenIdsMakerForGLM(TokenIdsMaker):
    @classmethod
    def process(cls, pair_data, tokenizer: PreTrainedTokenizer, max_seq_length: int, sptoken, src_max_length,
                dst_max_length):

        ds = [ ]
        prompt, responses, scores = pair_data

        a_ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        if src_max_length is not None and src_max_length > 0:
            a_ids = a_ids[ :src_max_length ]
        maxlen = 0
        for score, response in zip(scores, responses):
            response = cls.stop_response(response)
            b_ids = tokenizer.encode(response, truncation=True, max_length=max_seq_length, add_special_tokens=False)
            if dst_max_length is not None and dst_max_length > 0:
                b_ids = b_ids[ :dst_max_length ]

            a_ids_ = copy.deepcopy(a_ids)
            cls.trunction_ids(a_ids_, b_ids, max_seq_length, sptoken)
            input_ids = a_ids_ + sptoken + b_ids + [ tokenizer.eos_token_id ]
            pos = len(a_ids_)
            labels = [ -100 ] * pos + input_ids[ pos: ]

            maxlen = max(maxlen, len(input_ids))
            ds.append({
                "input_ids": input_ids,
                "labels": labels,
                "score": score,
                "ctxlens": len(input_ids)
            })

        return {
            "input_ids": np.stack(
                [ np.asarray(_[ "input_ids" ] + [ 0 ] * (maxlen - len(_[ "input_ids" ])), dtype=np.int32) for _ in
                  ds ]),
            "labels": np.stack(
                [ np.asarray(_[ "labels" ] + [ -100 ] * (maxlen - len(_[ "labels" ])), dtype=np.int32) for _ in ds ]),
            "scores": np.stack([ np.asarray(_[ "score" ], dtype=np.float32) for _ in ds ]),
            "ctxlens": np.stack([ np.asarray(_[ "ctxlens" ], dtype=np.float32) for _ in ds ]),
        }

class TokenIdsMakerForGLM2(TokenIdsMaker):
    @classmethod
    def process(cls, pair_data, tokenizer: PreTrainedTokenizer, max_seq_length: int, src_max_length, sptoken,
                dst_max_length):

        ds = [ ]
        prompt, responses, scores = pair_data

        a_ids = tokenizer.encode(prompt, truncation=True, max_length=max_seq_length, add_special_tokens=False)
        if src_max_length is not None and src_max_length > 0:
            a_ids = a_ids[ :src_max_length ]
        maxlen = 0
        for score, response in zip(scores, responses):
            response = cls.stop_response(response)
            b_ids = tokenizer.encode(response, truncation=True, max_length=max_seq_length, add_special_tokens=False)
            if dst_max_length is not None and dst_max_length > 0:
                b_ids = b_ids[ :dst_max_length ]

            a_ids_ = copy.deepcopy(a_ids)
            cls.trunction_ids(a_ids_, b_ids, max_seq_length, sptoken)
            input_ids = sptoken + a_ids_  + b_ids + [ tokenizer.eos_token_id ]
            pos = len(a_ids_) + len(sptoken)
            labels = [ -100 ] * pos + input_ids[ pos: ]
            maxlen = max(maxlen, len(input_ids))
            ds.append({
                "input_ids": input_ids,
                "labels": labels,
                "score": score
            })

        return {
            "input_ids": np.stack(
                [ np.asarray(_[ "input_ids" ] + [ 0 ] * (maxlen - len(_[ "input_ids" ])), dtype=np.int32) for _ in
                  ds ]),
            "labels": np.stack(
                [ np.asarray(_[ "labels" ] + [ -100 ] * (maxlen - len(_[ "labels" ])), dtype=np.int32) for _ in ds ]),
            "scores": np.stack([ np.asarray(_[ "score" ], dtype=np.float32) for _ in ds ]),
        }