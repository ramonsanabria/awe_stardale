
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=0

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="7"

models="hubert_base_ls960"
#model=non_trained
n_cluster=
pooling_methods="avg"


splits="xitsonga" 
nshard=10
#nshard=1

for model in ${models}
do

for split in ${splits}
do

	ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/hubert_data/raw_fixed/zsc/${model}_${split}/${layer}/
		ckpt_path=${ckpt_folder}/${model}.pt
		rm -r ${feat_dir}/*

		#extracing features all ranks
		mkdir -p ${feat_dir}

		for rank in $(seq 0 ${nshard_iter})
		do
			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

		done
		exit

	done
done
done
