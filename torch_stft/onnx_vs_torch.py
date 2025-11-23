import torch
import numpy as np
import onnx
from onnxsim import simplify
import onnxruntime as ort
import time

from stft_implementation.stft import STFT


def convert_to_onnx(n_fft, hop_length):
    src_buffer = torch.arange(nfft, dtype=torch.float32)

    stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=nfft, window="hann")
    stft.eval()

    dummy_input = src_buffer.unsqueeze(0)
    onnx_name = f"stft_graph.onnx"
    torch.onnx.export(stft, (dummy_input,), onnx_name, opset_version=11)
    # load your predefined ONNX model
    model = onnx.load(onnx_name)

    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, "simplified_stft_graph.onnx")


if __name__ == "__main__":
    device = 'cpu'
    nfft = 320
    hop = nfft // 2

    convert_to_onnx(nfft, hop)
    src_buffer = torch.arange((nfft), dtype=torch.float32).unsqueeze(0)

    exp_number = 15000

    #_________________TORCH________________________
    stft = STFT(n_fft=nfft, hop_length=hop, win_length=nfft, window="hann")
    stft.eval()

    t1 = time.time()
    for i in range(exp_number):
        torch_res = stft.forward(src_buffer)
    torch_time = (time.time() - t1)/exp_number
    print(f"torch time: {torch_time:.7f}")
    # print(torch_res)

    #_________________ONNX________________________
    model_path = "stft_graph.onnx"
    session = ort.InferenceSession(model_path)
    input_tensor = src_buffer.cpu().numpy()

    t1 = time.time()
    for i in range(exp_number):
        onnx_res = session.run(None, {"input_data": input_tensor})
    onnx_time = (time.time() - t1)/exp_number
    onnx_res_transpose = np.array([onnx_res[0][0]]).T
    print(f"onnx time: {onnx_time:.7f}")
    # print(onnx_res)

    #_________________ONNX_SIMPLIFIED________________________
    model_path = "simplified_stft_graph.onnx"
    session = ort.InferenceSession(model_path)
    input_tensor = src_buffer.cpu().numpy()

    t1 = time.time()
    for i in range(exp_number):
        simplified_onnx_res = session.run(None, {"input_data": input_tensor})
    simplified_onnx_time = (time.time() - t1)/exp_number
    print(f"simplified onnx time: {simplified_onnx_time:.7f}")
    # print(simplified_onnx_res)

    # _________________SUMMARY________________________
    print(f"torch / onnx time: {torch_time/onnx_time:.4f}")
    print(f"torch / simp onnx time: {torch_time/simplified_onnx_time:.4f}")
    print(f"torch err: {torch.abs(torch_res - src_buffer).sum()}")
    onnx_dif = np.abs(input_tensor - onnx_res[0])
    onnx_dif_transpose = np.array([onnx_dif[0]]).T
    onnx_dif_transpose_sorted = np.array([np.flip(np.sort(onnx_dif[0]))]).T
    print(f"onnx err: {np.sum(np.abs(input_tensor - onnx_res[0]))}")
    print(f"onnx simp err: {np.sum(np.abs(input_tensor - simplified_onnx_res[0]))}")
