import scipy
import scipy.signal as signal
import scipy.io
import scipy.io.wavfile

import os
import numpy as np

def load_audio_wav_resample(audio_path, DUR_SECS = 2, resample_SR = 16000, START_SECS=0, return_mono=True):
    """
    Loads a .wav file, chooses the length, and resamples to the desired rate.

    Inputs
    ------
    audio_path : string
        path to the .wav file to load
    DUR_SECS : int/float, or 'full'
        length of the audio to load in in seconds, if 'full' loads the (remaining) clip
    resample_SR : float
        sampling rate for the output sound
    START_SECS : int/float
        where to start reading the sound, in seconds
    return_mono : Boolean
        if true, returns a mono version of the sound

    """
    SR, audio = scipy.io.wavfile.read(audio_path)
    if DUR_SECS!='full':
        if (len(audio))/SR<DUR_SECS:
            print("PROBLEM WITH LOAD AUDIO WAV: The sound is only %d second while you requested %d seconds long"%(int((len(audio))/SR), DUR))
            return
    if return_mono:
        if audio.ndim>1:
            audio = audio.sum(axis=1)/2
    if DUR_SECS!='full':
        audio = audio[int(START_SECS*SR):int(START_SECS*SR) + int(SR*DUR_SECS)]
    else:
        audio = audio[int(START_SECS*SR):]
    if SR != resample_SR:
        audio = scipy.signal.resample_poly(audio, resample_SR, SR, axis=0)
        SR = resample_SR
    return audio, SR

def make_pink_noise(T,rms_value=False):
    """
    Makes a segment of pink noise length T and returns a numpy array with the values

    Inputs
    ------
    T : int
        length of the pink noise to generate
    rms_value : float
        normalization factor for the pink noise, ie the rms of a test signal. default no normalization.

    Returns
    -------
    pink_noise : numpy array
        numpy array containing pink noise of length T
    rms_pint : float
        the rms of the pink noise

    """
    uneven = T%2
    X = np.random.randn(T//2+1+uneven) + 1j * np.random.randn(T//2+1+uneven)
    S = np.sqrt(np.arange(len(X))+1.)
    pink_noise = (np.fft.irfft(X/S)).real
    if uneven:
        pink_noise = pink_noise[:-1]
    rms_pink = np.sqrt(np.mean(pink_noise**2))
    if rms_value: # basic normalization of pink noise
        pink_noise = (rms_value/rms_pink)*pink_noise
        rms_pink = np.sqrt(np.mean(pink_noise**2))
    return pink_noise, rms_pink

def rms_normalize_audio(audio, rms_value=0.01):
    """
    RMS normalize an audio segment so that sqrt(mean(x_i**2))==rms_value

    Inputs
    ------
    audio : numpy array [d]
       flattened audio signal, mono
    rms_value : float
       desired rms value

    Returns
    -------
    norm_audio : numpy array [d]
       rms normalized audio
    """
    assert len(audio.shape)==1, 'Only implmented for mono audio'
   
    rms_audio = np.sqrt(np.mean(audio**2))
    norm_audio = (rms_value/rms_audio)*audio
    return norm_audio

