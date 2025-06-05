# 以下、enhance_audio.py のサンプル内容（必要に応じて調整してください）

import os
import glob

import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf


def enhance_file(input_path: str, output_path: str, sr: int = 44100):
    """
    1. FLACを読み込む
    2. 初めの1秒を「ノイズプロファイル」として取得（環境ノイズ推定）
    3. noisereduce でスペクトル・ゲーティングを実行
    4. ダイナミックレンジを正規化
    5. 高品質 FLAC で書き出し
    """
    print(f"▶ 処理開始: {input_path}")

    # 1. 音声読み込み（ステレオ維持）
    audio, orig_sr = sf.read(input_path)  # (n_samples,) or (n_samples, 2)
    if orig_sr != sr:
        audio = librosa.resample(audio.T, orig_sr=orig_sr, target_sr=sr).T
    # else: そのまま

    # 2. ノイズプロファイル推定用に先頭1秒だけ抽出
    num_noise_samples = sr * 1  # 1秒分
    if audio.ndim == 1:
        noise_clip = audio[:num_noise_samples]
    else:
        noise_clip = audio[:num_noise_samples, :]

    # 3. ノイズ除去（スペクトル・ゲーティング）
    if audio.ndim == 1:
        denoised = nr.reduce_noise(
            y=audio,
            y_noise=noise_clip,
            sr=sr,
            n_fft=2048,
            prop_decrease=1.0,
            verbose=False
        )
    else:
        denoised = np.zeros_like(audio)
        # 左チャンネル
        denoised[:, 0] = nr.reduce_noise(
            y=audio[:, 0],
            y_noise=noise_clip[:, 0],
            sr=sr,
            n_fft=2048,
            prop_decrease=1.0,
            verbose=False
        )
        # 右チャンネル
        denoised[:, 1] = nr.reduce_noise(
            y=audio[:, 1],
            y_noise=noise_clip[:, 1],
            sr=sr,
            n_fft=2048,
            prop_decrease=1.0,
            verbose=False
        )

    # 4. ダイナミックレンジ正規化（ピークを 0.99 にスケール）
    peak = np.max(np.abs(denoised))
    if peak > 0:
        denoised = denoised / peak * 0.99

    # 5. 高品質 FLAC で書き出し
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(
        file=output_path,
        data=denoised,
        samplerate=sr,
        subtype='PCM_16'  # 16bit, CD 相当
    )

    print(f"✔ 出力完了: {output_path}\n")


def batch_enhance(input_dir: str, output_dir: str):
    """
    input_dir 以下のすべての .flac を再帰的に検索し、enhance_file を実行。
    出力は output_dir に、相対パス構造を維持して保存する。
    """
    pattern = os.path.join(input_dir, '**', '*.flac')
    flac_list = glob.glob(pattern, recursive=True)
    if not flac_list:
        print("!! 入力ディレクトリに .flac ファイルが見つかりませんでした。")
        return

    print(f"⚡ {len(flac_list)} 個のファイルを処理します...\n")
    for flac_path in flac_list:
        rel_path = os.path.relpath(flac_path, start=input_dir)
        out_path = os.path.join(output_dir, rel_path)
        enhance_file(flac_path, out_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='FLACファイルを一括でノイズ除去・音量正規化して高音質化します'
    )
    parser.add_argument(
        '--input-dir',
        '-i',
        type=str,
        required=True,
        help='処理対象の FLAC があるフォルダ'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        required=True,
        help='高音質化後のファイルを保存するフォルダ'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=44100,
        help='出力サンプリングレート (デフォルト: 44100)'
    )
    args = parser.parse_args()

    batch_enhance(args.input_dir, args.output_dir)
