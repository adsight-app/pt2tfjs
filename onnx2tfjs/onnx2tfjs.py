# Transcribed https://colab.research.google.com/github/cj-mills/tfjs-yolox-unity-tutorial/blob/main/notebooks/ONNX-to-TF-to-TFJS-Colab.ipynb

import os
from pathlib import Path
import onnx
from scc4onnx import order_conversion
from onnxsim import simplify
from onnx_tf.backend import prepare

import click

@click.command()
@click.argument("onnx_model_path", type=click.Path(exists=True))
def convert_onnx_to_tfjs(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    input_name = onnx_model.graph.input[0].name

    print('Ordering conversion')
    onnx_model = order_conversion(
        onnx_graph=onnx_model,
        input_op_names_and_order_dims={f"{input_name}": [0,2,3,1]},
        non_verbose=True
    )    

    print('Simplifying ONNX model')
    onnx_model, check = simplify(onnx_model)
    print(f'Checking ONNX model {check}')

    print('Preparing TF model')
    tf_rep = prepare(onnx_model)

    print('Exporting TF model')
    tf_model_dir = f"{onnx_model_path}-tf"
    Path(tf_model_dir).mkdir(parents=True, exist_ok=True)
    print(f"Converting ONNX model to TF model in {tf_model_dir}")
    tf_rep.export_graph(tf_model_dir)
    print(f"TF model saved in {tf_model_dir}")

    tfjs_model_dir = f"{tf_model_dir}-tfjs-uint8"
    Path(tfjs_model_dir).mkdir(parents=True, exist_ok=True)
    print(f"Converting ONNX model to TFJS uint8 model in {tfjs_model_dir}")
    tfjs_convert_command = [
        'uv run',
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--signature_name=serving_default',
        '--saved_model_tags=serve',
        tf_model_dir,
        tfjs_model_dir,
        '--quantize_uint8'
    ]

    command = ' '.join(tfjs_convert_command)
    print('Command to convert ONNX model to TFJS uint8 model:')
    print(command)
    os.system(command)
    print(f"TFJS uint8 model saved in {tfjs_model_dir}")

if __name__ == "__main__":
    convert_onnx_to_tfjs()