import torch
import argparse
from models.model import CFGCNN

def export_onnx(model_path, onnx_path
, input_shape=(1, 3, 224, 224), opset_version=11):
    MODEL_CONFIG_NAME = "RV1.1-1.yaml"

    """
    Load a PyTorch model and export it to ONNX format.

    Args:
        model_path (str): Path to the saved PyTorch model (.pt or .pth).
        onnx_path (str): Desired output path for the ONNX file.
        input_shape (tuple): Shape of the dummy input tensor.
        opset_version (int): ONNX opset version to use.
    """
    model = CFGCNN(cfg_name=MODEL_CONFIG_NAME)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"Model successfully exported to {onnx_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export a PyTorch model to ONNX format'
    )
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to the PyTorch model file (.pt or .pth)'
    )
    parser.add_argument(
        '--onnx-path', type=str, required=True,
        help='Output path for the ONNX file'
    )
    parser.add_argument(
        '--input-shape', type=int, nargs='+', default=[1, 3, 224, 224],
        help='Shape of the input tensor, e.g., --input-shape 1 3 224 224'
    )
    parser.add_argument(
        '--opset-version', type=int, default=11,
        help='ONNX opset version to use'
    )
    args = parser.parse_args()
    export_onnx(
        args.model_path,
        args.onnx_path,
        tuple(args.input_shape),
        args.opset_version
    )
