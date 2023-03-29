
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq


export CUDA_VISIBLE_DEVICES=3



python $FAIRSEQ_PATH/fairseq_cli/hydra_train.py \
  --config-dir $FAIRSEQ_PATH//examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech_100h \
  task.data=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french_mls_100h/ task.label_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french_mls_100h/l12_recovered/500 task.labels='["km"]' model.label_rate=50 

