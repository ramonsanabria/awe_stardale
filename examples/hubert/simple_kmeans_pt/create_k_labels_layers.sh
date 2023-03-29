
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate

export CUDA_VISIBLE_DEVICES=0

FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq

datasetname=french_mls_50hm
tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/${datasetname}
split=train
model=hubert_base_ls960
nshard=10
nclusters=500


ckpt_folder=/disk/scratch1/ramons/data/hubert_models/
ckpt_path=${ckpt_folder}/${model}.pt

layers='9 10 11 12'

nshard_iter="$((${nshard}-1))"


for layer in ${layers}
do

feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/${model}_l${layer}_${datasetname}
mkdir -p ${feat_dir}
#rm -r ${feat_dir}/*

for split in train;
do
	for rank in $(seq 0 ${nshard_iter})
	do
		#python dump_hubert_feature.py ${tsv_dir} ${split} ${ckpt_path} ${layer} ${nshard} ${rank} ${feat_dir}
        continue

	done
done
done


source deactivate
source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate

for layer in ${layers}
do

feat_dir=/disk/scratch1/ramons/data/hubert_data/raw/zsc/${model}_l${layer}_${datasetname}
km_folder=/disk/scratch1/ramons/data/hubert_data/raw/zsc/${datasetname}/${model}_l${layer}/
mkdir -p ${km_folder}
km_path=${km_folder}/k${nclusters}_l${layer}.model

#python learn_kmeans.py ${feat_dir} train ${nshard} ${km_path} ${nclusters} --percent 1

lab_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/${datasetname}/${model}_l${layer}/${nclusters}
        for split in train;
        do

        echo ${km_path}
        for rank in $(seq 0 ${nshard_iter})
        do
            #python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${lab_dir}
            continue

        done
        final_folder=${lab_dir}/${model}_l${layer}/${nclusters}


        for rank in $(seq 0 ${nshard_iter}); do
          cat ${lab_dir}/${split}_${rank}_${nshard}.km
        done > ${lab_dir}/${split}.km

        for rank in $(seq 0 ${nshard_iter}); do
          rm ${lab_dir}/${split}_${rank}_${nshard}.km
        done 

        rm -rf ${lab_dir}/dict.km.txt

        for x in $(seq 0 $(($nclusters - 1))); do
              echo "$x 1"
         done >> ${lab_dir}/dict.km.txt


        done
done

#rm -r ${feat_dir}
