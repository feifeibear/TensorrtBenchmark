#!/bin/bash
set +xe

SEQ_LEN=(10 20 40 60 80 100 200 300 400 500)
BATCH_SIZE=(1 20)

for batch_size in ${BATCH_SIZE[*]}
do
  for seq_len in ${SEQ_LEN[*]}
  do
# python ./helpers/convert_weights_pytorch.py -o ./bert
# python ./helpers/generate_data.py -b ${batch_size} -s ${seq_len} -o ./${batch_size}_${seq_len}/
# python helpers/generate_tokens.py -t ./helpers/text.txt -o  ./
./build/sample_bert_model  -d ./ -d ./${batch_size}_${seq_len}/  --nheads  12
done
done
