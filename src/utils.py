import torch
def generate_mask_suffix(args):
    parts = []
    if args.use_ptv_mask:
        parts.append('ptv')
    if args.use_bm_mask:
        parts.append('bm')
    if args.use_fh_mask:
        parts.append('fh')
    if args.use_ub_mask:
        parts.append('ub')

    mask_suffix = '+'.join(parts) if parts else 'none'
    return mask_suffix


def move_tensors_to_gpu(batch_dict, device='cuda'):
    """
    Move all tensor values in the batch_dict to the specified device (GPU).

    Parameters:
    - batch_dict (dict): A dictionary containing key-value pairs, where some values are expected to be tensors.
    - device (str): The device identifier to move the tensors to (default is 'cuda' for GPU).

    Returns:
    - dict: The same dictionary with all tensors moved to the specified device.
    """
    for key, value in batch_dict.items():
        if isinstance(value, torch.Tensor):
            batch_dict[key] = value.to(device)
    return batch_dict

def count_num_mask(args):
    num_mask = 0

    # Increment num_mask for each True argument
    if args.use_ptv_mask:
        num_mask += 1
    if args.use_bm_mask:
        num_mask += 1
    if args.use_fh_mask:
        num_mask += 1
    if args.use_ub_mask:
        num_mask += 1

    return num_mask