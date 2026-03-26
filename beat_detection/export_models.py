#!/usr/bin/env python
"""
Export madmom beat detection models to ONNX format.

Run this script ONCE in an environment where madmom is installed.
After export, madmom is no longer needed for inference.

Usage:
    python export_models.py [--output-dir models/] [--single]

Requirements (export only):
    pip install madmom onnx numpy
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np


def find_processors_recursive(processor):
    """Recursively find all leaf processors in a madmom pipeline."""
    results = []
    if hasattr(processor, 'processors'):
        for p in processor.processors:
            results.extend(find_processors_recursive(p))
    else:
        results.append(processor)
    return results


def extract_filterbank(processor):
    """Walk processor tree to find and extract the filterbank matrix."""
    all_procs = find_processors_recursive(processor)
    for p in all_procs:
        # FilteredSpectrogramProcessor stores the filterbank
        if hasattr(p, 'filterbank'):
            return np.array(p.filterbank, dtype=np.float32)
    return None


def extract_nn_ensemble(processor):
    """Extract the NeuralNetworkEnsemble from the processor pipeline."""
    from madmom.ml.nn import NeuralNetworkEnsemble
    all_procs = find_processors_recursive(processor)
    for p in all_procs:
        if isinstance(p, NeuralNetworkEnsemble):
            return p
    # Try top-level processors
    if hasattr(processor, 'processors'):
        for p in processor.processors:
            if isinstance(p, NeuralNetworkEnsemble):
                return p
    return None


def extract_layers_flat(model):
    """Recursively extract all leaf layers from a neural network model."""
    from madmom.ml.nn.layers import (
        FeedForwardLayer, RecurrentLayer, LSTMLayer, GRULayer,
        BidirectionalLayer
    )

    layers = []

    def _walk(layer):
        # If it has sub-layers, recurse
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                _walk(sub)
        elif isinstance(layer, BidirectionalLayer):
            # Mark as bidirectional
            layers.append(('bidirectional', layer))
        elif isinstance(layer, (LSTMLayer, GRULayer)):
            layers.append(('lstm', layer))
        elif isinstance(layer, RecurrentLayer):
            layers.append(('rnn', layer))
        elif isinstance(layer, FeedForwardLayer):
            layers.append(('dense', layer))
        else:
            # Unknown layer type, try to include anyway
            layers.append(('unknown', layer))

    _walk(model)
    return layers


def extract_lstm_weights(lstm_layer):
    """Extract weight matrices from a madmom LSTMLayer."""
    ig = lstm_layer.input_gate
    fg = lstm_layer.forget_gate
    cell = lstm_layer.cell
    og = lstm_layer.output_gate

    hidden_size = ig.bias.shape[0]
    input_size = ig.weights.shape[0]

    return {
        'input_size': int(input_size),
        'hidden_size': int(hidden_size),
        # Weights: madmom stores as (input_size, hidden_size)
        # ONNX expects (hidden_size, input_size) — so transpose
        'ig_W': ig.weights.T.astype(np.float32),
        'fg_W': fg.weights.T.astype(np.float32),
        'cell_W': cell.weights.T.astype(np.float32),
        'og_W': og.weights.T.astype(np.float32),
        # Recurrent weights: madmom (hidden_size, hidden_size) → transpose
        'ig_R': ig.recurrent_weights.T.astype(np.float32),
        'fg_R': fg.recurrent_weights.T.astype(np.float32),
        'cell_R': cell.recurrent_weights.T.astype(np.float32),
        'og_R': og.recurrent_weights.T.astype(np.float32),
        # Biases: (hidden_size,)
        'ig_b': ig.bias.astype(np.float32),
        'fg_b': fg.bias.astype(np.float32),
        'cell_b': cell.bias.astype(np.float32),
        'og_b': og.bias.astype(np.float32),
        # Initial states
        'h_init': getattr(lstm_layer, 'init',
                          np.zeros(hidden_size, dtype=np.float32)).astype(np.float32),
        'c_init': getattr(lstm_layer, 'cell_init',
                          np.zeros(hidden_size, dtype=np.float32)).astype(np.float32),
    }


def extract_dense_weights(dense_layer):
    """Extract weights from a madmom FeedForwardLayer."""
    return {
        'weights': dense_layer.weights.astype(np.float32),
        'bias': dense_layer.bias.astype(np.float32),
        'activation': 'sigmoid',  # beat models use sigmoid output
    }


def build_onnx_model(lstm_weights_list, dense_weights):
    """Build an ONNX model from extracted weights.

    The model processes a sequence of feature frames and outputs
    beat activations, with LSTM states as inputs/outputs for
    online (streaming) use.
    """
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    opset = 18
    nodes = []
    initializers = []
    graph_inputs = []
    graph_outputs = []

    num_features = lstm_weights_list[0]['input_size']

    # --- Input: features (seq_len, 1, num_features) ---
    graph_inputs.append(
        helper.make_tensor_value_info(
            'features', TensorProto.FLOAT,
            ['seq_len', 1, num_features]
        )
    )

    current_input = 'features'

    # --- LSTM layers ---
    for i, lw in enumerate(lstm_weights_list):
        H = lw['hidden_size']
        I = lw['input_size']

        # ONNX LSTM gate order: i, o, f, c
        # W: (num_directions=1, 4*H, input_size)
        W = np.zeros((1, 4 * H, I), dtype=np.float32)
        W[0, 0*H:1*H, :] = lw['ig_W']    # input gate
        W[0, 1*H:2*H, :] = lw['og_W']    # output gate
        W[0, 2*H:3*H, :] = lw['fg_W']    # forget gate
        W[0, 3*H:4*H, :] = lw['cell_W']  # cell gate

        # R: (1, 4*H, H)
        R = np.zeros((1, 4 * H, H), dtype=np.float32)
        R[0, 0*H:1*H, :] = lw['ig_R']
        R[0, 1*H:2*H, :] = lw['og_R']
        R[0, 2*H:3*H, :] = lw['fg_R']
        R[0, 3*H:4*H, :] = lw['cell_R']

        # B: (1, 8*H)  — first 4*H are input biases, next 4*H are recurrent (zeros)
        B = np.zeros((1, 8 * H), dtype=np.float32)
        B[0, 0*H:1*H] = lw['ig_b']
        B[0, 1*H:2*H] = lw['og_b']
        B[0, 2*H:3*H] = lw['fg_b']
        B[0, 3*H:4*H] = lw['cell_b']
        # Recurrent biases (4*H to 8*H) stay zero

        W_name = f'W_{i}'
        R_name = f'R_{i}'
        B_name = f'B_{i}'

        initializers.extend([
            numpy_helper.from_array(W, name=W_name),
            numpy_helper.from_array(R, name=R_name),
            numpy_helper.from_array(B, name=B_name),
        ])

        # State inputs
        h_in_name = f'h_{i}_in'
        c_in_name = f'c_{i}_in'
        graph_inputs.extend([
            helper.make_tensor_value_info(h_in_name, TensorProto.FLOAT, [1, 1, H]),
            helper.make_tensor_value_info(c_in_name, TensorProto.FLOAT, [1, 1, H]),
        ])

        # Initial state defaults
        h_init = lw['h_init'].reshape(1, 1, H)
        c_init = lw['c_init'].reshape(1, 1, H)
        initializers.extend([
            numpy_helper.from_array(h_init, name=f'h_{i}_init_default'),
            numpy_helper.from_array(c_init, name=f'c_{i}_init_default'),
        ])

        # LSTM node outputs
        lstm_Y = f'lstm_{i}_Y'
        h_out_name = f'h_{i}_out'
        c_out_name = f'c_{i}_out'

        lstm_node = helper.make_node(
            'LSTM',
            inputs=[current_input, W_name, R_name, B_name, '', h_in_name, c_in_name],
            outputs=[lstm_Y, h_out_name, c_out_name],
            hidden_size=H,
            direction='forward',
        )
        nodes.append(lstm_node)

        # State outputs
        graph_outputs.extend([
            helper.make_tensor_value_info(h_out_name, TensorProto.FLOAT, [1, 1, H]),
            helper.make_tensor_value_info(c_out_name, TensorProto.FLOAT, [1, 1, H]),
        ])

        # Reshape LSTM output: (seq_len, 1, 1, H) → (seq_len, 1, H)
        # ONNX LSTM Y shape: (seq_length, num_directions, batch_size, hidden_size)
        reshaped_name = f'lstm_{i}_reshaped'
        shape_name = f'shape_{i}'
        shape_val = np.array([0, 1, H], dtype=np.int64)  # 0 = infer seq_len
        initializers.append(numpy_helper.from_array(shape_val, name=shape_name))

        reshape_node = helper.make_node(
            'Reshape',
            inputs=[lstm_Y, shape_name],
            outputs=[reshaped_name],
        )
        nodes.append(reshape_node)
        current_input = reshaped_name

    # --- Dense output layer: MatMul + Add + Sigmoid ---
    dw = dense_weights['weights']  # (last_hidden, 1)
    db = dense_weights['bias']     # (1,)

    dense_W_name = 'dense_W'
    dense_b_name = 'dense_b'
    initializers.extend([
        numpy_helper.from_array(dw, name=dense_W_name),
        numpy_helper.from_array(db, name=dense_b_name),
    ])

    # MatMul: (seq_len, 1, H) @ (H, 1) → (seq_len, 1, 1)
    matmul_out = 'dense_matmul'
    nodes.append(helper.make_node('MatMul', [current_input, dense_W_name], [matmul_out]))

    # Add bias
    add_out = 'dense_add'
    nodes.append(helper.make_node('Add', [matmul_out, dense_b_name], [add_out]))

    # Sigmoid
    sigmoid_out = 'activation_raw'
    nodes.append(helper.make_node('Sigmoid', [add_out], [sigmoid_out]))

    # Reshape to (seq_len,)
    final_shape_name = 'final_shape'
    initializers.append(
        numpy_helper.from_array(np.array([-1], dtype=np.int64), name=final_shape_name)
    )
    nodes.append(
        helper.make_node('Reshape', [sigmoid_out, final_shape_name], ['activation'])
    )

    graph_outputs.insert(0,
        helper.make_tensor_value_info('activation', TensorProto.FLOAT, ['seq_len'])
    )

    # --- Build graph ---
    graph = helper.make_graph(
        nodes,
        'beat_lstm',
        graph_inputs,
        graph_outputs,
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', opset)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def capture_feature_dimensions(proc):
    """Run a test signal through the processor to determine feature dimensions."""
    test_audio = np.random.randn(44100).astype(np.float32) * 0.01
    try:
        result = proc(test_audio)
        if hasattr(result, 'shape'):
            return result.shape
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Export madmom beat detection models to ONNX format'
    )
    parser.add_argument(
        '--output-dir', default=os.path.join(os.path.dirname(__file__), 'models'),
        help='Output directory for exported models (default: models/)'
    )
    parser.add_argument(
        '--single', action='store_true',
        help='Export only the first model (faster, less accurate)'
    )
    args = parser.parse_args()

    # --- Check dependencies ---
    try:
        from madmom.features.beats import RNNBeatProcessor
        from madmom.models import BEATS_LSTM
        from madmom.ml.nn import NeuralNetworkEnsemble
        from madmom.ml.nn.layers import (
            FeedForwardLayer, LSTMLayer, BidirectionalLayer
        )
    except ImportError:
        print("ERROR: madmom is not installed.")
        print("Run this script in the conda environment with madmom.")
        sys.exit(1)

    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError:
        print("ERROR: onnx package not installed.")
        print("Install with: pip install onnx")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Create processor to extract preprocessing info ---
    print("Creating RNNBeatProcessor (online mode)...")
    proc = RNNBeatProcessor(online=True, fps=100)

    # --- Extract filterbank ---
    print("Extracting filterbank...")
    filterbank = None

    # Walk processor tree
    def _find_filterbank(p, depth=0):
        if depth > 20:
            return None
        if hasattr(p, 'filterbank'):
            fb = p.filterbank
            # Check it's an actual array/matrix, not a class reference
            if isinstance(fb, np.ndarray):
                return fb.astype(np.float32)
            elif hasattr(fb, '__array__') and not isinstance(fb, type):
                try:
                    return np.array(fb, dtype=np.float32)
                except (TypeError, ValueError):
                    pass
        if hasattr(p, 'processors'):
            for sub in p.processors:
                fb = _find_filterbank(sub, depth + 1)
                if fb is not None:
                    return fb
        return None

    filterbank = _find_filterbank(proc)

    if filterbank is None:
        # Fallback: try creating filterbank directly from madmom
        try:
            from madmom.audio.filters import LogarithmicFilterbank
            bin_freqs = np.fft.rfftfreq(2048, 1.0 / 44100)
            fb = LogarithmicFilterbank(
                bin_frequencies=bin_freqs,
                num_bands=12, fmin=30, fmax=17000,
                norm_filters=True, unique_filters=True,
            )
            filterbank = np.array(fb, dtype=np.float32)
            print(f"  Created filterbank from madmom API: {filterbank.shape}")
        except Exception as e:
            print(f"  WARNING: Could not create filterbank: {e}")
            print("  Will need manual configuration.")

    if filterbank is not None:
        num_filter_bands = filterbank.shape[1]
        print(f"  Filterbank shape: {filterbank.shape} ({num_filter_bands} bands)")
        np.save(os.path.join(args.output_dir, 'filterbank.npy'), filterbank)
    else:
        num_filter_bands = None

    # --- Extract neural network models ---
    print("Extracting neural network ensemble...")
    nn_ensemble = extract_nn_ensemble(proc)
    if nn_ensemble is None:
        # Try loading models directly
        print("  Could not find ensemble in pipeline, loading models directly...")
        nn_ensemble = NeuralNetworkEnsemble.load(BEATS_LSTM)

    model_files = BEATS_LSTM
    if args.single:
        model_files = [model_files[0]]

    print(f"  Found {len(model_files)} model(s)")

    # --- Export each model ---
    for idx, model_path in enumerate(model_files):
        print(f"\nExporting model {idx + 1}/{len(model_files)}...")

        # Load the individual model
        with open(model_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        # Extract layer structure
        layers = extract_layers_flat(model)
        print(f"  Layers found: {[t for t, _ in layers]}")

        # Separate LSTM and dense layers
        lstm_weights_list = []
        dense_w = None

        for layer_type, layer in layers:
            if layer_type == 'lstm':
                lw = extract_lstm_weights(layer)
                lstm_weights_list.append(lw)
                print(f"  LSTM layer: input={lw['input_size']}, hidden={lw['hidden_size']}")
            elif layer_type == 'bidirectional':
                # For online use, we only use the forward direction
                fwd = layer.fwd_layer if hasattr(layer, 'fwd_layer') else layer.fwd
                lw = extract_lstm_weights(fwd)
                lstm_weights_list.append(lw)
                print(f"  BiLSTM layer (fwd only): input={lw['input_size']}, "
                      f"hidden={lw['hidden_size']}")
            elif layer_type == 'dense':
                dense_w = extract_dense_weights(layer)
                print(f"  Dense layer: {dense_w['weights'].shape}, "
                      f"activation={dense_w['activation']}")

        if not lstm_weights_list or dense_w is None:
            print(f"  ERROR: Could not extract complete model structure. Skipping.")
            continue

        # Build and save ONNX model
        onnx_model = build_onnx_model(lstm_weights_list, dense_w)
        onnx_path = os.path.join(args.output_dir, f'beat_lstm_{idx + 1}.onnx')
        onnx.save(onnx_model, onnx_path)
        print(f"  Saved: {onnx_path}")

    # --- Save preprocessing config ---
    num_features = lstm_weights_list[0]['input_size'] if lstm_weights_list else None
    num_lstm_layers = len(lstm_weights_list)
    hidden_sizes = [lw['hidden_size'] for lw in lstm_weights_list]

    config = {
        'sample_rate': 44100,
        'fps': 100,
        'frame_size': 2048,
        'hop_size': 441,
        'num_features': num_features,
        'num_filter_bands': num_filter_bands,
        'fmin': 30,
        'fmax': 17000,
        'log_add': 1.0,
        'log_mul': 1.0,
        'diff_ratio': 0.5,
        'positive_diffs': True,
        'num_models': len(model_files),
        'num_lstm_layers': num_lstm_layers,
        'hidden_sizes': hidden_sizes,
        'has_filterbank': filterbank is not None,
    }

    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved: {config_path}")

    # --- Save initial LSTM states ---
    init_states = {}
    for i, lw in enumerate(lstm_weights_list):
        init_states[f'h_{i}_init'] = lw['h_init']
        init_states[f'c_{i}_init'] = lw['c_init']
    np.savez(
        os.path.join(args.output_dir, 'init_states.npz'),
        **init_states
    )

    print(f"\n{'='*50}")
    print(f"Export complete!")
    print(f"Models directory: {args.output_dir}")
    print(f"Models exported: {len(model_files)}")
    print(f"Input features: {num_features}")
    print(f"Filter bands: {num_filter_bands}")
    print(f"LSTM layers: {num_lstm_layers} (hidden: {hidden_sizes})")
    print(f"\nYou can now use BeatDetector without madmom.")
    print(f"Runtime dependencies: numpy, onnxruntime")


if __name__ == '__main__':
    main()
