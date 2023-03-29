
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

export CUDA_VISIBLE_DEVICES=0

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french_mls_50hm
split=train
nshard=10
lab_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french_mls_50hm/

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_large_ll60k
ckpt_path=${ckpt_folder}/${model}.pt

layers="9 11 12"

nshard_iter="$((${nshard}-1))"



for layer in ${layers}
do

feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/centroids_french_mls_50hm_${model}/l_${layer}
mkdir -p ${feat_dir}
rm -r ${feat_dir}/*

for split in train;
do
	for rank in $(seq 0 ${nshard_iter})
	do
		python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

	done
done



source deactivate
source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate

model_path=/disk/scratch1/ramons/data/kmeans_models/french_mls_50hm_${model}_l${layer}

python compute_clusters.py ${feat_dir} train ${model_path} ${nshard}
exit

rm -rf ${feat_dir}

done
