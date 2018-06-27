import scipy.io.wavfile as wav
import pandas as pd
from python_speech_features import mfcc
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

"""
Returns N mfcc vectors, where each one represents an mfcc vector
of winlen of audio.
"""


def create_vector_from_wav(file_name, talking):
    (rate, sig) = wav.read(file_name)
    audio_length = len(sig) / float(rate)
    mfcc_aux = mfcc(sig, rate, nfft=len(sig), winlen=audio_length)
    # a =  np.append(mfcc_aux[0], talking)
    # if len(a) != 14:
    #     print("ASDF")
    return np.append(mfcc_aux[0], talking)


def main():
    path_list = Path("Audios").iterdir()
    output_vector_length = 14
    data_frame = pd.DataFrame()
    for audio_path in path_list:
        talking_path = Path("{0}/not_talking".format(audio_path)).iterdir()
        not_talking_path = Path("{0}/talking".format(audio_path)).iterdir()
        for audio_file in talking_path:
            mfcc_feat = create_vector_from_wav(audio_file, True)
            data_frame = data_frame.append(pd.DataFrame(mfcc_feat.reshape(-1,len(mfcc_feat))))
        for audio_file in not_talking_path:
            mfcc_feat = create_vector_from_wav(audio_file, False)
            data_frame = data_frame.append(pd.DataFrame(mfcc_feat.reshape(-1, len(mfcc_feat))))

    columns = []
    for column_id in range(output_vector_length - 1):
        columns.append("mfcc_{0}".format(column_id))
    columns.append("talking")

    data_frame.columns = columns
    data_frame.to_csv('data.csv', index=False)

    # mfcc_feat = create_vector_from_wav("Audios/Blade_runner_2049/not_talking/Blade_runner_2049_0.wav", True)
    #
    # print(mfcc_feat)
    #
    # plt.plot(mfcc_feat[:13])
    # plt.show()


if __name__ == '__main__':
    main()
