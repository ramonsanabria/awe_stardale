
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=3

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="9"

#model=hubert_large_ll60k
models="recover_xitsonga_1echk"
#model=non_trained
n_cluster=
pooling_methods="avg"


splits="xitsonga" 
nshard=10
#nshard=1

for model in ${models}
do

for ch_iter in 360
do
for split in ${splits}
do

	ckpt_folder=/disk/scratch1/ramons/data/hubert_models/${model}
	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/${model}_${split}_${ch_iter}/${layer}/norm
		ckpt_path=${ckpt_folder}/checkpoints/checkpoint${ch_iter}.pt



		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do
			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

		done


		for pooling in ${pooling_methods}
		do
			echo "POOLING: ${pooling}, SPLIT: ${split}"

			feat_pooled_dir=/disk/scratch1/ramons/data/hubert_data/word_pooled_ssl/zsc/${model}_${ch_iter}/${layer}/norm/
			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "libri" ]] || [[ "${split}" == "libri_all" ]] || [[ "${split}" == "libri_dev" ]] 
                        then
				python generate_word_embeddings_single_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
			else
				                                                                                                   	python generate_word_embeddings_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}


			fi

		done
		rm -rf ${feat_dir}/*
	done
done
done
done
