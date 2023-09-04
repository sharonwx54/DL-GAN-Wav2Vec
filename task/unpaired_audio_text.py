# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from dataclasses import dataclass, field
import logging
import math
import os
from typing import Optional
import torch

from fairseq.logging import metrics
from fairseq.tasks import FairseqTask, register_task
from dataset import ExtractedFeaturesDataset, RandomInputDataset
import editdistance

from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)
from fairseq.dataclass import FairseqDataclass

logger = logging.getLogger(__name__)

@dataclass
class UnpairedAudioTextConfig(FairseqDataclass):
    # TODO: configure paths
    data: str = "/content/DeepLearningHW5GAN/11-685-s23-hw5/train_dev_data/audio_feats"
    text_data: str = "/content/DeepLearningHW5GAN/11-685-s23-hw5/train_dev_data/text_data"
    kenlm_path: Optional[str] = None # leave this to none as we are not using a language model
    labels: str = "phn"
    ctc_eval: bool = False
    shuffle: bool = True
    uppercase: Optional[bool] = False
    sort_by_length = False
    skipwords: Optional[str] = ""
    vocab_usage_power: float = 2

    word_decoder_config = None
    word_kenlm_path: Optional[str] = None


class UnpairedAudioText(FairseqTask):
    """ """

    cfg: UnpairedAudioTextConfig

    def __init__(
        self,
        cfg: UnpairedAudioTextConfig=UnpairedAudioTextConfig
    ):
        super().__init__(cfg)
        #print(os.getcwd())
        dict_path = os.path.join(cfg.text_data, "dict.txt")
        target_dictionary = Dictionary.load(dict_path)
        
        self._target_dictionary = target_dictionary

        self.num_symbols = (
            len([s for s in target_dictionary.symbols if not s.startswith("madeup")])
            - target_dictionary.nspecial
        )
        self.sil_id = (
            target_dictionary.index("<SIL>") if "<SIL>" in target_dictionary else -1
        )
        self.load_dataset("train")
        self.load_dataset("valid")
        self.kenlm = None
        if cfg.kenlm_path is not None:
            import kenlm

            self.kenlm = kenlm.Model(cfg.kenlm_path) # TODO: Call the kenlm model, just with the kenlm path from config

        self.word_kenlm = None
        if cfg.word_kenlm_path is not None:
            import kenlm

            self.word_kenlm = kenlm.Model(cfg.word_kenlm_path) # TODO: Call the kenlm model, but with the word kenlm path from config

        self.uppercase = cfg.uppercase
        self.skipwords = set(cfg.skipwords.split(","))

        def str_postprocess(s):
            s = " ".join(w for w in s.split() if w not in self.skipwords)
            s = s.upper() if self.uppercase else s
            return s

        self.str_postprocess = str_postprocess
        self.compute_lm_score = lambda s: self.kenlm.score(self.str_postprocess(s))

        self.compute_word_score = None
        if cfg.word_decoder_config is not None:
            self.kaldi_decoder = KaldiDecoder(cfg.word_decoder_config, beam=10)
             # TODO: Call the Kaldi Decoder with the word decoder config and beam == 10

            def compute_word_score(logits, padding):
                res = self.kaldi_decoder.decode(logits, padding) #TODO: Call decode of kaldi_decoder that you defined above, with arguments as logits and padding
                for r in res:
                    r = r.result()
                    assert len(r) == 1
                    r = r[0]
                    yield r["score"], r["words"]

            self.compute_word_score = compute_word_score
    
    def test_step(self, sample, model):
        res = model(
            **sample["net_input"],
            dense_x_only=True,
        )

        dense_x = res["logits"]
        padding_mask = res["padding_mask"]

        word_scores = None
        if self.compute_word_score is not None:
            word_scores = self.compute_word_score(dense_x.cpu(), padding_mask.cpu())

        z = dense_x.argmax(-1)
        z[padding_mask] = self.target_dictionary.pad()

        vocab_seen = torch.zeros(self.num_symbols, dtype=torch.bool)

        c_err = 0
        c_len = 0
        pred_c_len = 0
        lm_score_sum = 0

        pred = []
        for i, (x, t, id) in enumerate(
            zip(
                z,
                sample["target"] if "target" in sample else [None] * len(z),
                sample["id"],
            )
        ):

            if t is not None:
                t = t[(t >= self.target_dictionary.nspecial)]
            x = x[
                (x >= self.target_dictionary.nspecial)
                & (x < (self.num_symbols + self.target_dictionary.nspecial))
            ]
            if self.sil_id >= 0:
                x = x[x != self.sil_id]

            vocab_seen[x - self.target_dictionary.nspecial] = True

            pred_units_arr = x
            if self.cfg.ctc_eval:
                pred_units_arr = pred_units_arr.unique_consecutive()
                pred_units_arr = pred_units_arr[pred_units_arr != 0]

            if id == 0:
                if t is not None:
                    logger.info(f"REF: {self.target_dictionary.string(t)}")
                    print(f"REF: {self.target_dictionary.string(t)}")
                logger.info(f"HYP: {self.target_dictionary.string(pred_units_arr)}")
                print(f"HYP: {self.target_dictionary.string(pred_units_arr)}")

                if self.kenlm is not None:
                    if t is not None:
                        ref_lm_s = self.compute_lm_score(
                            self.target_dictionary.string(t)
                        )
                        logger.info(
                            f"LM [REF]: {ref_lm_s}, {math.pow(10, -ref_lm_s / (len(t) + 1))}"
                        )

                    hyp_lm_s = self.compute_lm_score(
                        self.target_dictionary.string(pred_units_arr)
                    )
                    logger.info(
                        f"LM [HYP]: {hyp_lm_s}, {math.pow(10, -hyp_lm_s / (len(pred_units_arr) + 1))}"
                    )

            pred_units_arr = pred_units_arr.tolist()
            pred.append(self.target_dictionary.string(pred_units_arr))

            pred_c_len += len(pred_units_arr)

            if t is not None:
                t = t.tolist()
                c_err += editdistance.eval(pred_units_arr, t)
                c_len += len(t)
            else:
                c_len = pred_c_len

            if self.kenlm is not None:
                pred_str = self.target_dictionary.string(pred_units_arr)
                lm_score = self.compute_lm_score(pred_str) # TODO: Call compute_lm_score on the pred_str
                lm_score_sum += lm_score

        kaldi_score_sum = 0
        word_lm_sum = 0
        num_words = 0
        if word_scores is not None:
            for score, words in word_scores:
                kaldi_score_sum += score
                num_words += len(words)
                if self.word_kenlm is not None:
                    word_lm_sum += self.kenlm.score(" ".join(words))

        try:
            world_size = get_data_parallel_world_size()
        except:
            world_size = 1

        logging_output = {
            "edit_distance": c_err,
            "_num_chars": c_len,
            "_num_pred_chars": pred_c_len,
            "nsentences": z.size(0),
            "_world_size": world_size,
            "_lm_score_sum": lm_score_sum,
            "_kaldi_score_sum": kaldi_score_sum,
            "_word_lm_sum": word_lm_sum,
            "_num_words": num_words,
            "_vocab_seen": vocab_seen,
        }

        return logging_output, pred

    def valid_step(self, sample, model):
        res = model(
            **sample["net_input"],
            dense_x_only=True,
        )

        dense_x = res["logits"]
        padding_mask = res["padding_mask"]

        word_scores = None
        if self.compute_word_score is not None:
            word_scores = self.compute_word_score(dense_x.cpu(), padding_mask.cpu())

        z = dense_x.argmax(-1)
        z[padding_mask] = self.target_dictionary.pad()

        vocab_seen = torch.zeros(self.num_symbols, dtype=torch.bool)

        c_err = 0
        c_len = 0
        pred_c_len = 0
        lm_score_sum = 0
        for i, (x, t, id) in enumerate(
            zip(
                z,
                sample["target"] if "target" in sample else [None] * len(z),
                sample["id"],
            )
        ):

            if t is not None:
                t = t[(t >= self.target_dictionary.nspecial)]
            x = x[
                (x >= self.target_dictionary.nspecial)
                & (x < (self.num_symbols + self.target_dictionary.nspecial))
            ]
            if self.sil_id >= 0:
                x = x[x != self.sil_id]

            vocab_seen[x - self.target_dictionary.nspecial] = True

            pred_units_arr = x
            if self.cfg.ctc_eval:
                pred_units_arr = pred_units_arr.unique_consecutive()
                pred_units_arr = pred_units_arr[pred_units_arr != 0]

            if id == 0:
                if t is not None:
                    logger.info(f"REF: {self.target_dictionary.string(t)}")
                logger.info(f"HYP: {self.target_dictionary.string(pred_units_arr)}")

                if self.kenlm is not None:
                    if t is not None:
                        ref_lm_s = self.compute_lm_score(
                            self.target_dictionary.string(t)
                        )
                        logger.info(
                            f"LM [REF]: {ref_lm_s}, {math.pow(10, -ref_lm_s / (len(t) + 1))}"
                        )

                    hyp_lm_s = self.compute_lm_score(
                        self.target_dictionary.string(pred_units_arr)
                    )
                    logger.info(
                        f"LM [HYP]: {hyp_lm_s}, {math.pow(10, -hyp_lm_s / (len(pred_units_arr) + 1))}"
                    )

            pred_units_arr = pred_units_arr.tolist()

            pred_c_len += len(pred_units_arr)

            if t is not None:
                t = t.tolist()
                c_err += editdistance.eval(pred_units_arr, t)
                c_len += len(t)
            else:
                c_len = pred_c_len

            if self.kenlm is not None:
                pred_str = self.target_dictionary.string(pred_units_arr)
                lm_score = self.compute_lm_score(pred_str) # TODO: Call compute_lm_score on the pred_str
                lm_score_sum += lm_score

        kaldi_score_sum = 0
        word_lm_sum = 0
        num_words = 0
        if word_scores is not None:
            for score, words in word_scores:
                kaldi_score_sum += score
                num_words += len(words)
                if self.word_kenlm is not None:
                    word_lm_sum += self.kenlm.score(" ".join(words))

        try:
            world_size = get_data_parallel_world_size()
        except:
            world_size = 1

        logging_output = {
            "edit_distance": c_err,
            "_num_chars": c_len,
            "_num_pred_chars": pred_c_len,
            "nsentences": z.size(0),
            "_world_size": world_size,
            "_lm_score_sum": lm_score_sum,
            "_kaldi_score_sum": kaldi_score_sum,
            "_word_lm_sum": word_lm_sum,
            "_num_words": num_words,
            "_vocab_seen": vocab_seen,
        }

        return logging_output
      
    @property
    def target_dictionary(self):
        return self._target_dictionary
      
    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        has_unpaired_text = os.path.exists(
            os.path.join(self.cfg.text_data, f"{split}.idx")
        )

        self.datasets[split] = ExtractedFeaturesDataset(
            path=data_path,
            split=split,
            min_length=3,
            max_length=None,
            labels=None if has_unpaired_text else task_cfg.labels,
            label_dict=self.target_dictionary,
            shuffle=getattr(task_cfg, "shuffle", True),
            sort_by_length=task_cfg.sort_by_length,
        )

        logger.info(f"split {split} has unpaired text? {has_unpaired_text}")
        if has_unpaired_text:
            text_dataset = data_utils.load_indexed_dataset(
                os.path.join(self.cfg.text_data, split), self.target_dictionary
            )
            text_dataset = StripTokenDataset(text_dataset, self.target_dictionary.eos())
            self.datasets[split] = RandomInputDataset(
                self.datasets[split],
                text_dataset,
                ["random_label"],
                add_to_input=True,
                pad_idx=self.target_dictionary.pad(),
            )
    def reduce_metrics(self, logging_outputs, criterion):
      pass