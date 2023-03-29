
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=1

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv/sse_ramons
layers="7 9 11"

model=recover_nchlt_tso_lrscratch
ckpt_folder=/disk/scratch1/ramons/data/hubert_models/${model}
model_chkpt=352_152000
n_cluster=
pooling_methods="avg"


splits="xitsonga"
nshard=10

for split in ${splits}
do

	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/sse_ramons/${model}/${layer}/norm
		ckpt_path=${ckpt_folder}/checkpoints/checkpoint_${model_chkpt}.pt


		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do
			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

		done


		for pooling in ${pooling_methods}
		do
			echo "POOLING: ${pooling}, SPLIT: ${split}"

			feat_pooled_dir=/disk/scratch1/ramons/data/hubert_data/sse_ramons_pooled/zsc/${model}/${layer}/norm/
			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "libri" ]] || [[ "${split}" == "test-clean" ]] || [[ "${split}" == "dev-clean" ]] || [[ "${split}" == "dev-clean_150" ]] || [[ "${split}" == "librispeech" ]]
                        then
				python generate_sse_embeddings_single_norm.py cross ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
				python generate_sse_embeddings_single_norm.py in ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
			else        
                continue
			fi
		done
	done
	rm -r ${feat_dir}/*
done


