import os
import sounddevice as sd
import soundfile as sf

from debug_configuration import OUTPUT_DEVICES, EUROPARL_PATH


def play_audio(file_path):
    """
    Plays the specified audio file.
    Output is assumed to be redirected to a virtual device (VB-Audio Virtual Cable)
    so that local_audio_streamer can capture it as input.
    """
    data, fs = sf.read(file_path, dtype="float32")
    if OUTPUT_DEVICES is not None:
        sd.default.device = OUTPUT_DEVICES
    sd.play(data, fs)
    sd.wait()  # Wait for playback to finish


def get_first_n_audio_files(n):
    """
    Returns the first n test audio files from Europarl.
    The list is based on the 'speeches.lst' file where each line contains:
    [text_es, text_en, audio_name]
    """
    test_folder = os.path.join(EUROPARL_PATH, "v1.1", "en", "es", "test")
    audio_folder = os.path.join(EUROPARL_PATH, "v1.1", "en", "audios_wav")  # Audio files in WAV format
    list_file = os.path.join(test_folder, "speeches.lst")

    audio_files = []
    with open(list_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[:n]:
            audio_filename = line.strip() + ".wav"
            audio_path = os.path.join(audio_folder, audio_filename)
            audio_files.append(audio_path)
    return audio_files


def main():
    audio_files = get_first_n_audio_files(8)
    for file_path in audio_files:
        print("Playing:", file_path)
        play_audio(file_path)


if __name__ == "__main__":
    main()
