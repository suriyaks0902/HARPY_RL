from .tensorrt_inference import TensorRTInference
from .onnx_export import export_policy_to_onnx
from configs.defaults import TENSORRT_CONFIG

class OptimizedPolicy:
    def __init__(self, policy, model_path):
        # Export to ONNX
        onnx_path = model_path.replace('.ckpt', '.onnx')
        export_policy_to_onnx(policy, onnx_path)
        
        # Create TensorRT engine
        self.trt_engine = TensorRTInference(onnx_path, TENSORRT_CONFIG)
        self.original_policy = policy

    def act(self, ob_vf, ob_pol, stochastic=True):
        if stochastic:
            return self.original_policy.act(ob_vf, ob_pol, stochastic)
        return self.trt_engine.infer(ob_vf, ob_pol)