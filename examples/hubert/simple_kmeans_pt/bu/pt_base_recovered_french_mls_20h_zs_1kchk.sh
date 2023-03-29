
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq


export CUDA_VISIBLE_DEVICES=1



python $FAIRSEQ_PATH/fairseq_cli/hydra_train.py \
  --config-dir $FAIRSEQ_PATH//examples/hubert/config/pretrain \
  --config-name hubert_base_librispeech_20h_zs_1kchk \
  task.data=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french_mls_20h_zs/ task.label_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french_mls_20h_zs/l12_recovered/500 task.labels='["km"]' model.label_rate=50 

