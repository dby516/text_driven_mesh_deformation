import torch
import numpy as np
from ..external_tools.part_segmentation_pytorch.pointnet.model import PointNetDenseCls
from ..in_out.datasets.shape_net_parts import ShapeNetParts

##
## shape_net_parts: i.e., use the code of Fei trained with Eric's shapenetcore parts
##

def generate_attention_map(point_cloud, attention_scores):
    return np.hstack((point_cloud, attention_scores.reshape(-1, 1)))

def generate_attention_from_probabilities(predictions, target_class=None):
    """
    Generate an attention map from segmentation probabilities.
    Args:
        predictions: Tensor of shape [batch_size, num_points, num_classes].
        target_class: Class index to focus on. If None, use max probabilities.
    Returns:
        Attention map: Tensor of shape [batch_size, num_points].
    """
    # Convert logits to probabilities using softmax
    probabilities = torch.softmax(predictions, dim=-1)

    if target_class is not None:
        # Use probabilities of the target class
        attention = probabilities[:, :, target_class]
    else:
        # Use maximum probability across all classes
        attention, _ = probabilities.max(dim=-1)

    # Normalize attention scores to [0, 1]
    attention = (attention - attention.min(dim=1, keepdim=True)[0]) / (
        attention.max(dim=1, keepdim=True)[0] - attention.min(dim=1, keepdim=True)[0] + 1e-8
    )
    return attention

@torch.no_grad()
def shape_net_parts_segmentor_inference_(segmentor, pc, target_class=0, device="cuda", bcn_format=True):
    """
    Perform inference using the shape segmentation model and generate attention maps.
    
    Args:
        segmentor: The pretrained segmentation model (PointNetDenseCls).
        dataloader: DataLoader for the input point clouds.
        target_class: region specified in utterance
        device: Device to perform computation (e.g., 'cuda' or 'cpu').
        bcn_format: BCN format
    
    Returns:
        part_predictions: List of segmentation predictions for each batch.
        attention_scores: List of attention maps for each batch.
    """
    '''With no batch'''
    # segmentor.eval()
    # pc_tensor = torch.tensor(pc, dtype=torch.float32).to(device)
    # pc = pc_tensor.to(device)
    # # Perform the prediction using the evaluating_part_predictor
    # prediction_logits = segmentor(pc)[0]
    # attention_score = generate_attention_from_probabilities(prediction_logits, target_class=target_class).cpu().numpy()
    # part_prediction = torch.argmax(prediction_logits, -1).cpu().numpy()
    '''With batches'''
    segmentor.eval()
    part_predictions = []
    attention_scores = []

    
    for batch in pc:
        pc = batch['pointcloud'].to(device)

        if bcn_format:
            pc = pc.transpose(2, 1).contiguous()

        prediction_logits = segmentor(pc)[0]

        attention_map = generate_attention_from_probabilities(prediction_logits, target_class=target_class)

        prediction = torch.argmax(prediction_logits, -1)
        part_predictions.append(prediction.cpu().numpy())
        attention_scores.append(attention_map.cpu().numpy())

        # part_predictions.append(prediction.cpu())
    part_predictions = np.vstack(part_predictions)  # Shape: [total_points, num_classes]
    attention_scores = np.vstack(attention_scores)  # Shape: [total_points]
    
    return part_predictions, attention_scores


@torch.no_grad()
def shape_net_parts_segmentor_inference(segmentor, dataloader, bcn_format=True, device='cuda'):
    segmentor.eval()
    all_predictions = []
    for batch in dataloader:
        pc = batch['pointcloud'].to(device)

        if bcn_format:
            pc = pc.transpose(2, 1).contiguous()

        prediction_logits = segmentor(pc)[0]
        prediction = torch.argmax(prediction_logits, -1)
        all_predictions.append(prediction.cpu())
    return torch.cat(all_predictions).numpy()


def load_shape_net_parts_segmentor(segmentor_file, shape_class, feature_transform=False):
    n_parts = ShapeNetParts.n_parts[shape_class]
    part_segmentor = PointNetDenseCls(k=n_parts, feature_transform=feature_transform)
    part_segmentor.load_state_dict(torch.load(segmentor_file))
    return part_segmentor