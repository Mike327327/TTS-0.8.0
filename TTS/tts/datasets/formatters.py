import os
import re
import itertools # iterating over 2 lists (parczech formatter)
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

########################
# DATASETS
########################

def mailabs(root_path, meta_files=None, ignored_speakers=None):
    """Assumes each line as ```<filename>|<transcription>|<transcription>```
    """
    txt_file = os.path.join(root_path, meta_files)
    items = []
    speaker_name = ""
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0]) + ".wav"
            text = cols[1]
            speaker_name = cols[0][:7]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return items

def parczech(root_path, meta_files=None, spk_files=None):
    """
    2 files are required:
      1) metadata.txt (each line as ```<file_path>|<transcription>```)
      2) spk_mapping.txt (each line as ```<file_path>|<speaker_name>```)
      - both files are in the same order of wavs (lines correspond)
      - there must be 'wavs' folder in the path (e. g. all the wavs must be saved in the 'wavs' folder)
    """
    txt_file = os.path.join(root_path, meta_files)
    spk_file = os.path.join(root_path, spk_files)
    delimiter = "\\"

    items = []

    txt_file = open(txt_file, "r", encoding="utf-8")
    spk_file = open(spk_file, "r", encoding="utf-8")

    for (txt, spk) in zip(txt_file, spk_file):                  # iterate over 2 lists simulataneously
        # txt
        cols = txt.split("|")                                   # ...\path\to\file.wav, transcription
        relative_path = cols[0].split(delimiter)                # split on the delimiter, CAN VARY! 
        index = relative_path.index("wavs")                     # find 'wavs' in the file path, get index
        postfix = delimiter.join(relative_path[index:])         # everything from 'wavs' till the end of the path
        wav_file = os.path.join(root_path, postfix)
        text = cols[1]

        # spk
        spk_cols = spk.split("|")
        speaker_name = spk_cols[1].strip()                      # get speaker's name

        items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name, "root_path": root_path})
    return(items)
    
def coqui(root_path, meta_file, ignored_speakers=None):
    """Interal dataset formatter."""
    metadata = pd.read_csv(os.path.join(root_path, meta_file), sep="|")
    assert all(x in metadata.columns for x in ["audio_file", "text"])
    speaker_name = None if "speaker_name" in metadata.columns else "coqui"
    emotion_name = None if "emotion_name" in metadata.columns else "neutral"
    items = []
    not_found_counter = 0
    for row in metadata.itertuples():
        if speaker_name is None and ignored_speakers is not None and row.speaker_name in ignored_speakers:
            continue
        audio_path = os.path.join(root_path, row.audio_file)
        if not os.path.exists(audio_path):
            not_found_counter += 1
            continue
        items.append(
            {
                "text": row.text,
                "audio_file": audio_path,
                "speaker_name": speaker_name if speaker_name is not None else row.speaker_name,
                "emotion_name": emotion_name if emotion_name is not None else row.emotion_name,
            }
        )
    if not_found_counter > 0:
        print(f" | > [!] {not_found_counter} files not found")
    return items

def ljspeech(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items


def ljspeech_test(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file for TTS testing
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        speaker_id = 0
        for idx, line in enumerate(ttf):
            # 2 samples per speaker to avoid eval split issues
            if idx % 2 == 0:
                speaker_id += 1
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": f"ljspeech-{speaker_id}"})
    return items

def common_voice(root_path, meta_file, ignored_speakers=None):
    """Normalize the common voice meta data file to TTS format."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            if line.startswith("client_id"):
                continue
            cols = line.split("\t")
            text = cols[2]
            speaker_name = cols[0]
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_name in ignored_speakers:
                    continue
            wav_file = os.path.join(root_path, "clips", cols[1].replace(".mp3", ".wav"))
            items.append({"text": text, "audio_file": wav_file, "speaker_name": "MCV_" + speaker_name})
    return items


def libri_tts(root_path, meta_files=None, ignored_speakers=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if not meta_files:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split(".")[0]
        with open(meta_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("\t")
                file_name = cols[0]
                speaker_name, chapter_id, *_ = cols[0].split("_")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                wav_file = os.path.join(_root_path, file_name + ".wav")
                text = cols[2]
                # ignore speakers
                if isinstance(ignored_speakers, list):
                    if speaker_name in ignored_speakers:
                        continue
                items.append({"text": text, "audio_file": wav_file, "speaker_name": f"LTTS_{speaker_name}"})
    for item in items:
        assert os.path.exists(item["audio_file"]), f" [!] wav files don't exist - {item['audio_file']}"
    return items

def vctk(root_path, meta_files=None, wavs_path="wav48_silence_trimmed", mic="mic1", ignored_speakers=None):
    """VCTK dataset v0.92.

    URL:
        https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip

    This dataset has 2 recordings per speaker that are annotated with ```mic1``` and ```mic2```.
    It is believed that ```mic1``` files are the same as the previous version of the dataset.

    mic1:
        Audio recorded using an omni-directional microphone (DPA 4035).
        Contains very low frequency noises.
        This is the same audio released in previous versions of VCTK:
        https://doi.org/10.7488/ds/1994

    mic2:
        Audio recorded using a small diaphragm condenser microphone with
        very wide bandwidth (Sennheiser MKH 800).
        Two speakers, p280 and p315 had technical issues of the audio
        recordings using MKH 800.
    """
    file_ext = "flac"
    items = []
    meta_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, "r", encoding="utf-8") as file_text:
            text = file_text.readlines()[0]
        # p280 has no mic2 recordings
        if speaker_id == "p280":
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_mic1.{file_ext}")
        else:
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_{mic}.{file_ext}")
        if os.path.exists(wav_file):
            items.append({"text": text, "audio_file": wav_file, "speaker_name": "VCTK_" + speaker_id})
        else:
            print(f" [!] wav files don't exist - {wav_file}")
    return items


def vctk_old(root_path, meta_files=None, wavs_path="wav48", ignored_speakers=None):
    """homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"""
    items = []
    meta_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, "r", encoding="utf-8") as file_text:
            text = file_text.readlines()[0]
        wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + ".wav")
        items.append({"text": text, "audio_file": wav_file, "speaker_name": "VCTK_old_" + speaker_id})
    return items
