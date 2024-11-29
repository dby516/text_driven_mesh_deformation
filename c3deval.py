import torch
import numpy as np
import os.path as osp
from functools import partial
import open3d as o3d
import os
from itertools import chain

from changeit3d.in_out.changeit3d_net import prepare_input_data
from changeit3d.in_out.language_contrastive_dataset import LanguageContrastiveDataset
from changeit3d.in_out.pointcloud import pc_loader_from_npz, uniform_subsample
from changeit3d.in_out.basics import pickle_data
from changeit3d.in_out.basics import create_logger
from changeit3d.in_out.arguments import parse_evaluate_changeit3d_arguments

from changeit3d.utils.basics import parallel_apply
from changeit3d.models.model_descriptions import load_pretrained_changeit3d_net
from changeit3d.models.model_descriptions import load_pretrained_pc_ae

from changeit3d.evaluation.auxiliary import pc_ae_transform_point_clouds, sgf_transform_point_clouds
from changeit3d.external_tools.sgf.loader import initialize_and_load_sgf
from changeit3d.utils.visualization import visualize_point_clouds_3d_v2, plot_3d_point_cloud, plot_point_cloud_with_attention


'''Preparations'''
# Specify paths
top_data_dir = '../autodl-fs/shapetalk'
shape_talk_file = f'{top_data_dir}/language/shapetalk_preprocessed_public_version_0.csv'
vocab_file = f'{top_data_dir}/language/vocabulary.pkl'
top_pc_dir = f'{top_data_dir}/scaled_to_align_rendering'
pretrained_oracle_listener = f'{top_data_dir}/pretrained/listeners/oracle_listener/all_shapetalk_classes/rs_2023/listener_dgcnn_based/ablation1/best_model.pkl'
pretrained_shape_classifier =  f'{top_data_dir}/pretrained/pc_classifiers/rs_2022/all_shapetalk_classes/best_model.pkl'
shape_part_classifiers_top_dir = f'{top_data_dir}/pretrained/part_predictors/shapenet_core_based'
latent_codes_file = f'{top_data_dir}/pretrained/shape_latents/pcae_latent_codes.pkl'

# Load pretrained Generator(auto encoder) (pcae)
pretrained_shape_generator = f'{top_data_dir}/pretrained/pc_autoencoders/pointnet/rs_2022/points_4096/all_classes/scaled_to_align_rendering/08-07-2022-22-23-42/best_model.pt'
selected_ablation = 'decoupling_mag_direction/idpen_0.05_sc_True/' # decoupled and with self-contrast=True

# Load pretrained Editor
pretrained_shape_editor = f'{top_data_dir}/pretrained/changers/pcae_based/all_shapetalk_classes/{selected_ablation}/best_model.pt'

'''Training Configs (when using command line)'''
notebook_arguments = []

notebook_arguments.extend(['-shape_talk_file', shape_talk_file])
notebook_arguments.extend(['-latent_codes_file', latent_codes_file])
notebook_arguments.extend(['-vocab_file', vocab_file])
notebook_arguments.extend(['-pretrained_changeit3d', pretrained_shape_editor])
notebook_arguments.extend(['-top_pc_dir', top_pc_dir])
notebook_arguments.extend(['--shape_generator_type', 'pcae'])
notebook_arguments.extend(['--pretrained_oracle_listener', pretrained_oracle_listener])
notebook_arguments.extend(['--pretrained_shape_classifier', pretrained_shape_classifier])
notebook_arguments.extend(['--shape_part_classifiers_top_dir', shape_part_classifiers_top_dir])
notebook_arguments.extend(['--n_sample_points', '2048'])
notebook_arguments.extend(['--gpu_id', '0'])

if 'pretrained_shape_generator' in  locals():
    notebook_arguments.extend(['--pretrained_shape_generator', pretrained_shape_generator])

# if 'sub_sample_dataset' in  locals():
#     notebook_arguments.extend(['--sub_sample_dataset', sub_sample_dataset])    
    

args = parse_evaluate_changeit3d_arguments(notebook_arguments)
args.batch_size = max(1, args.batch_size // 2) # use less memory
args.gpu_id = 0
logger = create_logger(args.log_dir)

df, shape_to_latent_code, shape_latent_dim, vocab = prepare_input_data(args, logger)

'''Load pretrained models'''
# Load pretrained editor
logger.info('Loading pretrained ChangetIt3DNet (C3DNet)')
c3d_net, best_epoch, c3d_args = load_pretrained_changeit3d_net(args.pretrained_changeit3d, shape_latent_dim, vocab)
# device = torch.device("cuda:" + str(args.gpu_id))
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


c3d_net = c3d_net.to(device)
logger.info(f'The model is variant `{c3d_args.shape_editor_variant}` trained with {c3d_args.identity_penalty} identity penalty and Self-Contrast={c3d_args.self_contrast}.')
logger.info(f'Loaded at epoch {best_epoch}.')

print(f"Successfully load pretrained Editor: {args.pretrained_changeit3d}")

# Load pretrained Generator
if args.shape_generator_type == 'pcae':
    pc_ae, pc_ae_args = load_pretrained_pc_ae(args.pretrained_shape_generator)
    pc_ae = pc_ae.to(device)
    pc_ae = pc_ae.eval()

print(f"Successfully load pretrained Generator:{args.shape_generator_type}")


'''Load input data: original PointCloud, utterance'''
print("Loading input data...")


# Dataloader
def to_stimulus_func(x):
    return shape_to_latent_code[x]

split = 'test'
ndf = df[df.changeit_split == split].copy()
ndf.reset_index(inplace=True, drop=True)

if args.sub_sample_dataset is not None:
    np.random.seed(args.random_seed)
    ndf = ndf.sample(args.sub_sample_dataset)
    ndf.reset_index(inplace=True, drop=True)
    
dataset = LanguageContrastiveDataset(ndf,
                                     to_stimulus_func,
                                     n_distractors=1,
                                     shuffle_items=False)  # important, source (distractor) now is first

dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         worker_init_fn=lambda _ : np.random.seed(args.random_seed)) 


print("Successfully load input data!")

check_loader = dataloader
gt_classes = check_loader.dataset.df.source_object_class

'''Decode edits'''
print("Decoding latent editing direction & magnitude...")

if args.shape_generator_type == 'pcae':
    transformation_results = pc_ae_transform_point_clouds(pc_ae,
                                                          c3d_net,
                                                          check_loader,
                                                          stimulus_index=0,
                                                          scales=[0, 1],  # use "0" to get also the simple reconstruction of the decoder (no edit)
                                                          device=device)


transformed_shapes = transformation_results['recons'][1]
# Sample point-clouds to desired granularity for evaluation
if transformed_shapes.shape[-2] != args.n_sample_points:
    transformed_shapes = np.array([uniform_subsample(s, args.n_sample_points, args.random_seed)[0] for s in transformed_shapes])
language_used = [vocab.decode_print(s) for s in transformation_results['tokens']]
gt_pc_files = check_loader.dataset.df.source_uid.apply(lambda x: osp.join(args.top_pc_dir, x + ".npz")).tolist()

'''Testing'''
sentences = ndf.utterance_spelled.values
gt_classes = gt_classes.values
pc_loader =  partial(pc_loader_from_npz, n_samples=args.n_sample_points, random_seed=args.random_seed)
gt_pcs = parallel_apply(gt_pc_files, pc_loader, n_processes=20) # or, gt_pcs = [pc_loader(m) for m in gt_pc_files]
gt_pcs = np.array(gt_pcs)

print(args.n_sample_points, transformed_shapes.shape, gt_pcs.shape, gt_classes.shape, sentences.shape)

'''Save inference results generated by the original Editor'''
# for i in range(len(transformed_shapes)):
#     # Try on 1 case
#     transformed_shape = transformed_shapes[i]
#     language_used_ = language_used[i]
#     gt_pc_file = gt_pc_files[i]
    
#     # if args.save_reconstructions:
#     #     outputs = dict()
#     #     outputs['transformed_shapes'] = transformed_shapes
#     #     outputs['language_used'] = language_used
#     #     outputs['gt_input_pc_files'] = gt_pc_files
#     #     pickle_data(osp.join(args.log_dir, 'evaluation_outputs.pkl'), outputs)

#     # Load ground-truth input point-clouds
#     pc_loader =  partial(pc_loader_from_npz, n_samples=args.n_sample_points, random_seed=args.random_seed)
#     gt_pcs = pc_loader(gt_pc_file)
#     sentence = sentences[i]
#     gt_class = gt_classes[i]

#     # ---- Visualization ----
#     os.makedirs(f"results/{gt_class}", exist_ok=True)

#     vis_axis_order = [0, 2, 1]
#     pic = visualize_point_clouds_3d_v2([gt_pcs, transformed_shape], fig_title=language_used_)
#     pic.save(f'results/{gt_class}/{sentence}.png')

#     # save point cloud
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(np.array(transformed_shape))
#     # Perform MLS interpolation (smoothing)
#     pcd_mls = point_cloud.voxel_down_sample(voxel_size=0.01)  # Downsample if needed
#     o3d.geometry.PointCloud.estimate_normals(pcd_mls, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
#     # Save or display the smoothed/interpolated point cloud
#     o3d.io.write_point_cloud(f"results/{gt_class}/{sentence}_{gt_class}.ply", pcd_mls)

# print("Finish testing! Results have been saved")

# ---11/22---
'''Evaluation'''
from changeit3d.evaluation.all_metrics import run_all_metrics

# run_all_metrics(transformed_shapes, gt_pcs, gt_classes, sentences, vocab, args, logger)

'''
Use masking to restrict changes to the target region.
'''
import torch
import numpy as np
import pandas as pd
import os.path as osp
from torch.utils.data import DataLoader
from functools import partial

from changeit3d.evaluation.semantic_part_based import mark_part_reference_in_sentences, masks_of_referred_parts_for_pcs
from changeit3d.in_out.pointcloud import (PointcloudDataset, 
                                          swap_axes_of_pointcloud,
                                          center_in_unit_sphere)

from changeit3d.models.shape_part_segmentors import shape_net_parts_segmentor_inference_, load_shape_net_parts_segmentor


network_input_transformations = dict()

## this is the transformation we use in our pretrained pc-based-classifier
## YOU MUST Change this accordingly if you use a different classifier
network_input_transformations['shape_classifier'] = partial(center_in_unit_sphere, in_place=False)

def pc_transform_for_part_predictor(pc):
    pc = swap_axes_of_pointcloud(pc, [0, 2, 1])
    pc = center_in_unit_sphere(pc)
    return pc        


gt_classes = pd.Series(gt_classes, name='shape_class')  #convert to pandas for ease of use via .groupby    
tokens = pd.Series([sent.split() for sent in sentences])
max_len = tokens.apply(len).max()
tokens_encoded = np.stack(tokens.apply(lambda x: vocab.encode(x, max_len=max_len))) # encode to ints and put them in a N-shape array

## this is the transformation we use in our pretrained part-predictor-classifier
network_input_transformations['part_predictor'] = pc_transform_for_part_predictor

scale_chamfer_by = 1000
batch_size_for_cd = 2048

# Store problematic samples for evaluation
mask_transformed_shapes = []
mask_combined_shapes = []
mask_gt_pcs = []
mask_gt_classes = []
mask_sentences = []

if args.shape_part_classifiers_top_dir:    
    shape_classes = ['chair', 'lamp', 'table']
    for shape_class in shape_classes:
        # Loading a part-clf to measure the localized-Geometric-Difference (l-GD) score
        file_location = osp.join(args.shape_part_classifiers_top_dir, f'best_seg_model_{shape_class}.pth')
        evaluating_part_predictor = load_shape_net_parts_segmentor(file_location, shape_class)
        evaluating_part_predictor = evaluating_part_predictor.to(device)

        idx_per_class = (gt_classes[gt_classes == shape_class].index).tolist()
        input_gt = gt_pcs[idx_per_class]
        input_trans = transformed_shapes[idx_per_class]
        
        gt_loader = DataLoader(PointcloudDataset(input_gt, pc_transform=network_input_transformations['part_predictor']), batch_size=128, num_workers=10, shuffle=False)        
        trans_loader = DataLoader(PointcloudDataset(input_trans, pc_transform=network_input_transformations['part_predictor']), batch_size=128, num_workers=10, shuffle=False)
        
        # STEP-1: Group point clouds by their target class
        _, ref_parts_idx = mark_part_reference_in_sentences(tokens[idx_per_class], gt_classes[idx_per_class])
        print('ref_parts_idx: ', len(ref_parts_idx))
        sentences_ = sentences[idx_per_class]
        print("len:", input_gt.shape, input_trans.shape)

        # Initialize storage for predictions by class
        pred_parts_gt = []
        pred_parts_trans = []
        attention_scores_gt = []
        attention_scores_trans = []

        # Flatten the list of lists and find unique target classes
        target_classes = list(set(chain.from_iterable(ref_parts_idx)))
        print("target_classes: ", target_classes)

        # Perform inference
        for target_class in target_classes:
            # Perform inference for ground truth loader
            pred_part_gt, att_score_gt = shape_net_parts_segmentor_inference_(
                evaluating_part_predictor, 
                gt_loader, 
                target_class=target_class, 
                device=device
            )
            pred_parts_gt.append(pred_part_gt)
            attention_scores_gt.append(att_score_gt)

            # Perform inference for transformed loader
            pred_part_trans, att_score_trans = shape_net_parts_segmentor_inference_(
                evaluating_part_predictor, 
                trans_loader, 
                target_class=target_class, 
                device=device
            )
            pred_parts_trans.append(pred_part_trans)
            attention_scores_trans.append(att_score_trans)
            print("Finish inference for target class: ", target_class)

        # Debugging: Print shapes and ensure correct alignment
        print(f"pred_parts_gt.shape: {len(pred_parts_gt)}, {len(pred_part_gt[0])}")
        print(f"pred_parts_trans.shape: {len(pred_parts_trans)}, {len(pred_parts_trans[0])}")
        print(f"attention_scores_gt.shape: {len(attention_scores_gt)}, {len(attention_scores_gt[0])}")
        print(f"attention_scores_trans.shape: {len(attention_scores_trans)}, {len(attention_scores_trans[0])}")


        # STEP-2: Find the actual references that do part-specific language to focus
        cnt = 0
        # Iterate over reference indices and point clouds
        for idx, gt, trans in zip(ref_parts_idx, input_gt, input_trans):
            save_directory = f"../autodl-fs/output_attention_plots/{shape_class}/{sentences_[cnt]}"  # Directory to save plots
            if len(idx) > 0:
                target_class = idx[0]
                # Get predictions for the current target class
                pred_parts_gt_class = pred_parts_gt[target_class]
                pred_parts_trans_class = pred_parts_trans[target_class]
                attention_scores_gt_class = attention_scores_gt[target_class]
                attention_scores_trans_class = attention_scores_trans[target_class]

                # Retrieve specific prediction for the current index
                pred_part_gt = pred_parts_gt_class[cnt]
                attention_score_gt = attention_scores_gt_class[cnt]
                pred_part_trans = pred_parts_trans_class[cnt]
                attention_score_trans = attention_scores_trans_class[cnt]

                plot_point_cloud_with_attention(
                    pc=gt,  # Shape: [2048, 3]
                    attention_scores=attention_score_gt,  # Shape: [2048]
                    save_dir=save_directory,
                    visualization_pc_axis=[0,2,1]
                )
                
                # Define the mask based on attention scores
                mask_gt = attention_score_gt < 0.5  # Points with low attention in `input_gt`
                mask_trans = attention_score_trans > 0.5  # Points with high attention in `input_trans`
                combined_points = np.concatenate((gt[mask_gt], trans[mask_trans]), axis=0)  # Add points from input_trans
                
                # Store problematic samples
                mask_combined_shapes.append(combined_points)
                mask_transformed_shapes.append(trans)
                mask_gt_pcs.append(gt)
                mask_gt_classes.append(shape_class)
                mask_sentences.append(sentences_[cnt])
            else:
                combined_points = trans
                pass
            
            # Save transformed shape with attention
            os.makedirs(f"{save_directory}", exist_ok=True)
            pic = visualize_point_clouds_3d_v2([gt, combined_points, trans], fig_title=sentences_[cnt])
            pic.save(f'{save_directory}/combined.png')
            print(f"Combined points plot saved to {save_directory}/combined.png")

            cnt += 1
        
'''Evaluation'''
# on original editor
run_all_metrics(mask_transformed_shapes, mask_gt_pcs, mask_gt_classes, mask_sentences, vocab, args, logger)

# on attentioned editor
run_all_metrics(mask_combined_shapes, mask_gt_pcs, mask_gt_classes, mask_sentences, vocab, args, logger)


       
        