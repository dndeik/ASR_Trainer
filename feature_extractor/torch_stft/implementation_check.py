import torch
from stft_implementation.stft import STFT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import soundfile as sf
matplotlib.use('TkAgg')


def compare_spectrogram(nfft, hop, audio_path):
    device = torch.device("cpu")
    audio, _ = sf.read(audio_path)
    src_buffer = torch.from_numpy(audio).float()

    window_func = torch.hann_window(nfft).pow(0.5)
    torch_stft_res = torch.stft(src_buffer, nfft, hop, nfft, window_func, return_complex=False)
    src_buffer = src_buffer.to(device)

    window = "hann"
    stft = STFT(
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=window
    ).to(device)

    custom_stft_res = stft.transform(src_buffer.unsqueeze(0))

    # Plot
    # torch_mag = torch.sqrt(torch_stft_res[..., 0] ** 2 + torch_stft_res[..., 1] ** 2)[0]
    # print(torch.max(torch_stft_res))
    # print(torch.min(torch_stft_res))
    custom_stft_res = custom_stft_res
    print(torch.max(custom_stft_res))
    print(torch.min(custom_stft_res))
    custom_mag = torch.sqrt(custom_stft_res[..., 0] ** 2 + custom_stft_res[..., 1] ** 2 + 1e-12)[0]
    # custom_mag = custom_stft_res[..., 1]
    # custom_mag = torch.log10(custom_mag)
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    # axs[0].imshow(20 * np.log10(1 + torch_mag.cpu().data.numpy()), aspect='auto', origin='lower')
    # axs[0].set_title('Torch mag')
    axs[0].imshow(custom_mag.cpu().data.numpy(), aspect='auto', origin='lower')
    axs[0].set_title('Custom clear mag')
    root_mag = custom_mag.cpu().data.numpy()**0.3
    root_mag = np.where(root_mag<0.2, 0, root_mag)
    cbar = axs[1].imshow(root_mag, aspect='auto', origin='lower')
    axs[1].set_title('Custom log mag')
    axs[2].imshow(20 * np.log10(1 + custom_mag.cpu().data.numpy()), aspect='auto', origin='lower')
    axs[2].set_title('Custom log (1+) mag')
    fig.colorbar(cbar, ax=axs[1])
    # plt.plot(torch_mag)
    # plt.plot(custom_mag)
    # plt.grid()
    plt.show()


def check_accuracy(nfft, hop):
    device = torch.device("cpu")
    src_buffer = torch.arange(nfft * 100, dtype=torch.float32)

    src_buffer = src_buffer.to(device)

    window = "hann"
    stft = STFT(
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=window
    ).to(device)

    custom_stft_res = stft.transform(src_buffer.unsqueeze(0))
    output = stft.inverse(custom_stft_res)

    output = output.cpu().data.numpy()[..., :]
    src_buffer = src_buffer.cpu().data.numpy()[..., :]

    sorted_dif = np.array(np.flip(np.sort(np.abs(output-src_buffer)))).T
    eps = 0.001
    binary = np.where(sorted_dif > eps, 1, 0)
    print(f"Err > {eps} in {(binary.sum()/binary.shape[0]) * 100:.2f}%")
    print(f"MSE: {np.mean((output - src_buffer) ** 2):.6f}") # on order of 1e-17


def check_performance(nfft, hop, device):
    window_func = torch.hann_window(nfft).pow(0.5)

    window = "hann"
    stft = STFT(
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=window
    ).to(device)

    durations = [2**i for i in range(1, 10)]
    durations_in_min = [((el*nfft)/16000) for el in durations]
    torch_stft_times = []
    conv_stft_times = []
    exp_numbers = 30
    for step in durations:
        src_buffer = torch.arange(nfft * step, dtype=torch.float32).unsqueeze(0)
        src_buffer = src_buffer.to(device)

        t1 = time.time()
        for i in range(exp_numbers):
            torch_stft = torch.stft(src_buffer, nfft, hop, nfft, window_func, return_complex=False)
            torch_istft = torch.istft(torch_stft, nfft, hop, nfft, window_func, return_complex=False)
        torch_stft_times.append((time.time() - t1) / exp_numbers)

        t1 = time.time()
        for i in range(exp_numbers):
            stft_res = stft.transform(src_buffer)
            output = stft.inverse(stft_res)
        conv_stft_times.append((time.time() - t1) / exp_numbers)

    plt.plot(durations, torch_stft_times, label="torch stft+istft times", marker=".")
    plt.plot(durations, conv_stft_times, label="conv stft+istft times", marker=".")
    plt.xlabel("Seq duration (packets)")
    plt.ylabel("Processing time")
    plt.legend()
    plt.grid(True)
    plt.show()

def check_center_vs_not_center(nfft, hop):
    device = torch.device("cpu")
    src_buffer = torch.arange(nfft * 100, dtype=torch.float32).unsqueeze(0).to(device)

    window = "hann"
    center_stft = STFT(
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=window,
        center=True
    ).to(device)

    uncenter_stft = STFT(
        n_fft=nfft,
        hop_length=hop,
        win_length=nfft,
        window=window,
        center=False
    ).to(device)

    center_res = []
    new_samples = torch.zeros((1, nfft), device=device)
    for idx, i in enumerate(range(0, src_buffer.size()[1], hop)):
        new_samples[:, :hop] = new_samples[:, hop:]
        new_samples[:, hop:] = src_buffer[:, i:i + hop]
        center_transform = center_stft.transform(new_samples)
        ready_samples = center_stft.inverse(center_transform)[0]
        center_res.append(ready_samples[hop:])

    center_res = torch.cat(center_res, dim=0)
    print(center_res)

    uncenter_transform = uncenter_stft.transform(src_buffer)
    uncenter_res = uncenter_stft.inverse(uncenter_transform)[0]

    print(center_res.size())
    sample_num = 16001
    print(f"{sample_num} in both ver: ", center_res[sample_num], uncenter_res[sample_num])
    plt.plot(center_res)
    plt.plot(uncenter_res)
    plt.show()
    print(torch.max(torch.abs(center_res - uncenter_res)))
    print(torch.allclose(center_res, uncenter_res, atol=1e-3))


if __name__ == "__main__":
    device = 'cpu'
    nfft = 320
    hop = nfft // 2

    audio_path = r"D:\Python_Projects\!General_Scripts\NikitaAudio\clean_fileid_178396.wav"
    compare_spectrogram(nfft, hop, audio_path)
    # check_accuracy(nfft, hop)
    # check_performance(nfft, hop, "cpu")
    # check_center_vs_not_center(nfft, hop)