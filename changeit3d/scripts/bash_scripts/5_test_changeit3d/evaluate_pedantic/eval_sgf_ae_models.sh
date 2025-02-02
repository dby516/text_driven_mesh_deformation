models=(\    
    'decoupling_mag_direction/idpen_0.01_sc_True'\
    'decoupling_mag_direction/idpen_0.01_sc_False'\
    'coupled/idpen_0.01_sc_True'\
    'coupled/idpen_0.01_sc_False'\
    'decoupling_mag_direction/idpen_0.025_sc_True'\
    'decoupling_mag_direction/idpen_0.025_sc_False'\
    'coupled/idpen_0.025_sc_True'\
    'coupled/idpen_0.025_sc_False'\
    'decoupling_mag_direction/idpen_0.05_sc_True'\
    'decoupling_mag_direction/idpen_0.05_sc_False'\
    'coupled/idpen_0.05_sc_True'\
    'coupled/idpen_0.05_sc_False'\
    'decoupling_mag_direction/idpen_0.075_sc_True'\
    'decoupling_mag_direction/idpen_0.075_sc_False'\
    'coupled/idpen_0.075_sc_True'\
    'coupled/idpen_0.075_sc_False'\
    'decoupling_mag_direction/idpen_0.1_sc_True'\
    'decoupling_mag_direction/idpen_0.1_sc_False'\
    'coupled/idpen_0.1_sc_True'\
    'coupled/idpen_0.1_sc_False')


script=/home/panos/Git_Repos/changeit3d/changeit3d/scripts/evaluate_change_it_3d.py
top_model_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data/pretrained/changers/sgf_based/all_shapetalk_classes
top_data_dir=/home/panos/Git_Repos/changeit3d/changeit3d/data
gpu_id=0
sub_sample_dataset=5000 ## SGF reconstructions take a while; we reduce the test-set to a sub-sampled version of 5,000 shapes

for model in ${models[@]}; do
    
    pretrained_changeit3d=$top_model_dir/$model/best_model.pt        
    log_dir=$(dirname $pretrained_changeit3d)/results

    # use for evaluating nearest-neighbor baseline
    # log_dir=$(dirname $pretrained_changeit3d)/results_with_nearest_neighb_baseline
    # --evaluate_retrieval_version True\
                
    python $script\
        -pretrained_changeit3d $pretrained_changeit3d\
        -shape_talk_file $top_data_dir/shapetalk/language/shapetalk_preprocessed_public_version_0.csv\
        -latent_codes_file $top_data_dir/pretrained/shape_latents/sgf_latent_codes.pkl\
        -vocab_file $top_data_dir/shapetalk/language/vocabulary.pkl\
        -top_pc_dir $top_data_dir/shapetalk/point_clouds/scaled_to_align_rendering\
        --shape_generator_type sgf\
        --pretrained_shape_classifier $top_data_dir/pretrained/pc_classifiers/rs_2022/all_shapetalk_classes/best_model.pkl\
        --shape_part_classifiers_top_dir $top_data_dir/pretrained/part_predictors/shapenet_core_based\
        --pretrained_oracle_listener $top_data_dir/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_2023/listener_dgcnn_based/ablation1/best_model.pkl\
        --log_dir $log_dir\
        --sub_sample_dataset $sub_sample_dataset\
        --save_reconstructions True\
        --gpu_id
done
