import torch
import torch.onnx
from baselines.common import tf_util as U
import tensorflow as tf

def export_policy_to_onnx(policy, save_path):
    """
    Export TensorFlow policy to ONNX format
    """
    # Get the TF session
    sess = U.get_session()
    
    # Create dummy inputs matching your policy's input shapes
    dummy_vf = np.zeros((1,) + policy.ob_vf_shape)
    dummy_pol = np.zeros((1,) + policy.ob_pol_shape)
    
    # Export to ONNX
    torch.onnx.export(
        policy,
        (torch.from_numpy(dummy_vf), torch.from_numpy(dummy_pol)),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_vf', 'input_pol'],
        output_names=['action', 'value'],
        dynamic_axes={
            'input_vf': {0: 'batch_size'},
            'input_pol': {0: 'batch_size'},
            'action': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )