
source /disk/scratch1/ramons/myenvs/fairseq/bin/activate


FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/fairseq

tsv_dir=/disk/scratch1/ramons/data/zerospeech/tsv/
set_kmeans="libri_100"
set_target="french_50hm"


nshard=10
rank=1

feat_dir_kmeans=/disk/scratch1/ramons/data/hubert_data/raw/zsc/en_pt/mfcc_${set_kmeans}
feat_dir_target=/disk/scratch1/ramons/data/hubert_data/raw/zsc/en_pt/mfcc_${set_target}

lab_dir=/disk/scratch1/ramons/data/zerospeech/tsv/pt/french/

nshard_iter="$((${nshard}-1))"

mkdir -p ${feat_dir_kmeans}

rm -r ${feat_dir_kmeans}/*


for rank in $(seq 0 ${nshard_iter})
do
    python dump_mfcc_feature.py ${tsv_dir} ${set_kmeans} ${nshard} ${rank} ${feat_dir_kmeans}
    exit
done


source deactivate
source /disk/scratch1/ramons/myenvs/p3_dave/bin/activate

for nclusters in 500;
do


    km_path=/disk/scratch1/ramons/data/hubert_data/raw/zsc/${set_kmeans}/k${nclusters}_mfcc.model
    python learn_kmeans.py ${feat_dir_kmeans} ${set_kmeans} ${nshard} ${km_path} ${nclusters} --percent 1

    for rank in $(seq 0 ${nshard_iter})
    do
        python dump_mfcc_feature.py ${tsv_dir} ${set_target} ${nshard} ${rank} ${feat_dir}
    done



    mkdir -p ${final_folder}
    for rank in $(seq 0 ${nshard_iter})
    do
        python dump_km_label.py ${feat_dir} ${split} ${km_path} ${nshard} ${rank} ${final_folder}
    done
    for rank in $(seq 0 ${nshard_iter}); do
        cat ${final_folder}/${split}_${rank}_${nshard}.km
    done > ${final_folder}/valid_en.km
    for rank in $(seq 0 ${nshard_iter}); do
        rm ${final_folder}/${split}_${rank}_${nshard}.km
    done 
    for x in $(seq 0 $($nclusters - 1)); do
        echo "$x 1"
    done >> ${final_folder}/dict.km.txt


done



