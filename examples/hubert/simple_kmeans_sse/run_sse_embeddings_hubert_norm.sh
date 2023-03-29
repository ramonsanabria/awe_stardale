
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate
export CUDA_VISIBLE_DEVICES=3

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv
layers="1 11 15 19 23"

ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
model=hubert_large_ll60k
n_cluster=
pooling_methods="avg sub10"


splits="dev-clean_150"
nshard=10
#nshard=1

for split in ${splits}
do

	for layer in ${layers}
	do

		nshard_iter="$((${nshard}-1))"
		feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/sse/${model}/${layer}/norm
		ckpt_path=${ckpt_folder}/${model}.pt


		#extracing features all ranks
		for rank in $(seq 0 ${nshard_iter})
		do
			python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}

		done


		for pooling in ${pooling_methods}
		do
			echo "POOLING: ${pooling}, SPLIT: ${split}"

			feat_pooled_dir=/disk/scratch1/ramons/data/hubert_data/sse_pooled/zsc/${model}/${layer}/norm/
			if [[ "${split}" == "mboshi" ]] || [[ "${split}" == "xitsonga" ]] || [[ "${split}" == "libri" ]] || [[ "${split}" == "test-clean" ]] || [[ "${split}" == "dev-clean" ]] || [[ "${split}" == "dev-clean_150" ]]
                        then
				python generate_sse_embeddings_single_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}
			else
				                                                                                                   					python generate_sse_embeddings_norm.py ${split} ${model} ${layer} ${pooling} ${tsv_dir} ${nshard} ${feat_dir} ${feat_pooled_dir}


			fi
		done
	done
	rm -r ${feat_dir}/*
done


