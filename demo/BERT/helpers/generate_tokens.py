#!/usr/bin/env python3

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from os.path import join as jp
import numpy as np
from transformers import BertTokenizer

TEST_INPUT_FN = 'test_inputs.weights_int32'

parser = argparse.ArgumentParser(description='DATA to TensorRT Weight ')
parser.add_argument('-t', '--text', type=str, required=True, help='txt')
parser.add_argument('-o', '--output', type=str, required=True, help='txt')

opt = parser.parse_args()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = []
with open(opt.text, "r") as f:
    for line in f:
        line = line.strip()
        tokens_ids = tokenizer.encode(line)
        print(tokens_ids)
        inputs.append(tokens_ids)

max_seq_len = np.max([len(seq) for seq in inputs])

input_tokes = []
input_masks = []
input_segment_ids = []
for tokens_ids in inputs:
    input_mask = [1 for _ in range(len(tokens_ids))]
    segment_ids = [0 for _ in range(max_seq_len)]
    pad_range = max_seq_len - len(tokens_ids)
    tokens_ids += [0 for _ in range(pad_range)]
    input_mask += [0 for _ in range(pad_range)]
    print(tokens_ids)
    input_tokes.append(tokens_ids)
    input_masks.append(input_mask)
    input_segment_ids.append(segment_ids)


fd = {'input_ids:0': np.array(input_tokes, dtype=np.int32),
      'input_mask:0': np.array(input_masks, dtype=np.int32),
      'segment_ids:0': np.array(input_segment_ids, dtype=np.int32)}


out_fn = jp(opt.output, TEST_INPUT_FN)
with open(out_fn, 'wb') as output_file:
    count = 3
    output_file.write("{}\n".format(count).encode('ASCII'))
    idx = 0
    for k, v in fd.items():
        outname = '{}_{}'.format(k[:-2], idx)
        shape = v.shape
        shape_str = '{} '.format(len(shape)) + ' '.join(
            [str(d) for d in shape])
        output_file.write(
            "{} 3 {} ".format(outname, shape_str).encode('ASCII'))
        output_file.write(v.tobytes())
        output_file.write("\n".encode('ASCII'))
