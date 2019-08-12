"""
Test Code for tfcochleagram

Usage: 
To test changes to the code, run the following: 
python tests_tfcochleagram.py

If new tests are added, make sure that the old ones are satisfied and then create a new test function using make_test_file_tfcochleagram.py and push the new test file with the git commit. If changes reveal bugs or modified implementations leading to difference in test files, make sure to add it in the git commit comments 

"""


from __future__ import division
from scipy.io import wavfile

import sys
if sys.version_info < (3,):
    from StringIO import StringIO as BytesIO
else:
    from io import BytesIO
import base64

sys.path.append("/om/user/jfeather/python-packages/py-cochleagram2/py-cochleagram")

import numpy as np
import scipy
import scipy.signal

import tensorflow as tf

import tfcochleagram
import os

def test_tfcochleagram_code(test_file_string='tfcochleagram_tests.npy'):
    save_test_cochs = np.load(test_file_string)
    for i in range(len(save_test_cochs)):
        with tf.Graph().as_default():
            nets = {}
            nets['input_signal'] = tf.Variable(save_test_cochs[i]['test_audio'], dtype=tf.float32)
            nets, COCH_PARAMS = tfcochleagram.cochleagram_graph(nets, **save_test_cochs[i]['COCH_PARAMS']) # use the default values
            with tf.Session() as sess:
                test_node = nets[save_test_cochs[i]['test_node_name']].eval(feed_dict = {nets['input_signal']:save_test_cochs[i]['test_audio']})
        if not np.all(np.isclose(test_node, save_test_cochs[i]['test_node'])):
            print("Test %i does not pass, Parameters are: %s"%(i, ''.join(['{0}{1}'.format(k, v) for k,v in save_test_cochs[i]['COCH_PARAMS'].items()])))

    print('All Tests Passed using test file %s'%test_file_string)


def load_audio_wav_resample(audio_path, DUR_SECS = 2, resample_SR = 16000, START_SECS=0, return_mono=True):
    """
    Loads a .wav file, chooses the length, and resamples to the desired rate. 
    
    Parameters
    ----------
    audio_path : string
        path to the .wav file to load
    DUR_SECS : int/float
        length of the audio to load in in seconds
    resample_SR : float
        sampling rate for the output sound
    START_SECS : int/float
        where to start reading the sound, in seconds
    return_mono : Boolean
        if true, returns a mono version of the sound
        
    """
    SR, audio = scipy.io.wavfile.read(audio_path)
    if (len(audio))/SR<DUR_SECS:
        print("PROBLEM WITH LOAD AUDIO WAV: The sound is only %d second while you requested %d seconds long"%(int((len(audio))/SR), DUR))
        return
    if return_mono:
        if audio.ndim>1:
            audio = audio.sum(axis=1)/2
    audio = audio[int(START_SECS*SR):int(START_SECS*SR) + int(SR*DUR_SECS)]
    if SR != resample_SR:
        audio = scipy.signal.resample_poly(audio, resample_SR, SR, axis=0)
        SR = resample_SR
    return audio, SR

def make_test_file_tfcochleagram(test_file_name='tfcochleagram_tests.npy', overwrite=False):
    """
    Use to build tests for the cochleagram code. Saves values and parameters such that the model can be rebuilt and checked after code changes. 

    Parameters
    ----------
    test_file_name : string
        location of the saved test file
    returns : list
        list containing dictionary with parameters for each test
    
    """
    if os.path.isfile(test_file_name):
        if not overwrite:
            raise FileExistsError('Testing file already exists and overwrite=False.')

    test_audio, SR = load_audio_wav_resample('speech_1.wav')
    save_test_cochs = []
    with tf.Graph().as_default():
        if len(test_audio.shape) == 1: 
            test_audio = np.expand_dims(test_audio,0) 
        nets = {}
        nets['input_signal'] = tf.Variable(test_audio, dtype=tf.float32)
        nets, COCH_PARAMS = tfcochleagram.cochleagram_graph(nets, SR, compression="none", return_coch_params=True) # use the default values
        with tf.Session() as sess:
            test_node_name = 'cochleagram'
            test_node = nets[test_node_name].eval(feed_dict = {nets['input_signal']:test_audio})
        save_test_cochs.append({'COCH_PARAMS':COCH_PARAMS, 'test_node':test_node, 'test_audio':test_audio, 'test_node_name':test_node_name})

    with tf.Graph().as_default():
        if len(test_audio.shape) == 1: 
            test_audio = np.expand_dims(test_audio,0) 
        nets = {}
        nets['input_signal'] = tf.Variable(test_audio, dtype=tf.float32)
        nets, COCH_PARAMS = tfcochleagram.cochleagram_graph(nets, SR, compression="clipped_point3", return_coch_params=True) # use the default values
        with tf.Session() as sess:
            test_node_name = 'cochleagram'
            test_node = nets[test_node_name].eval(feed_dict = {nets['input_signal']:test_audio})
        save_test_cochs.append({'COCH_PARAMS':COCH_PARAMS, 'test_node':test_node, 'test_audio':test_audio, 'test_node_name':test_node_name})

    with tf.Graph().as_default():
        if len(test_audio.shape) == 1:
            test_audio = np.expand_dims(test_audio,0)
        nets = {}
        nets['input_signal'] = tf.Variable(test_audio, dtype=tf.float32)
        nets, COCH_PARAMS = tfcochleagram.cochleagram_graph(nets, SR, compression="clipped_point3", return_coch_params=True, pad_factor=2) # use the default values
        with tf.Session() as sess:
            test_node_name = 'cochleagram'
            test_node = nets[test_node_name].eval(feed_dict = {nets['input_signal']:test_audio})
        save_test_cochs.append({'COCH_PARAMS':COCH_PARAMS, 'test_node':test_node, 'test_audio':test_audio, 'test_node_name':test_node_name})

    ## Start rFFT tests
    with tf.Graph().as_default():
        if len(test_audio.shape) == 1:
            test_audio = np.expand_dims(test_audio,0)
        nets = {}
        nets['input_signal'] = tf.Variable(test_audio, dtype=tf.float32)
        nets, COCH_PARAMS = tfcochleagram.cochleagram_graph(nets, SR, compression="none", return_coch_params=True, rFFT=True) # use the default values
        with tf.Session() as sess:
            test_node_name = 'cochleagram'
            test_node = nets[test_node_name].eval(feed_dict = {nets['input_signal']:test_audio})
        save_test_cochs.append({'COCH_PARAMS':COCH_PARAMS, 'test_node':test_node, 'test_audio':test_audio, 'test_node_name':test_node_name})

    with tf.Graph().as_default():
        if len(test_audio.shape) == 1:
            test_audio = np.expand_dims(test_audio,0)
        nets = {}
        nets['input_signal'] = tf.Variable(test_audio, dtype=tf.float32)
        nets, COCH_PARAMS = tfcochleagram.cochleagram_graph(nets, SR, compression="clipped_point3", return_coch_params=True, rFFT=True) # use the default values
        with tf.Session() as sess:
            test_node_name = 'cochleagram'
            test_node = nets[test_node_name].eval(feed_dict = {nets['input_signal']:test_audio})
        save_test_cochs.append({'COCH_PARAMS':COCH_PARAMS, 'test_node':test_node, 'test_audio':test_audio, 'test_node_name':test_node_name})

    with tf.Graph().as_default():
        if len(test_audio.shape) == 1:
            test_audio = np.expand_dims(test_audio,0)
        nets = {}
        nets['input_signal'] = tf.Variable(test_audio, dtype=tf.float32)
        nets, COCH_PARAMS = tfcochleagram.cochleagram_graph(nets, SR, compression="clipped_point3", return_coch_params=True, pad_factor=2, rFFT=True) # use the default values
        with tf.Session() as sess:
            test_node_name = 'cochleagram'
            test_node = nets[test_node_name].eval(feed_dict = {nets['input_signal']:test_audio})
        save_test_cochs.append({'COCH_PARAMS':COCH_PARAMS, 'test_node':test_node, 'test_audio':test_audio, 'test_node_name':test_node_name})

    np.save(test_file_name, save_test_cochs)
    return save_test_cochs

if __name__ == '__main__':
    test_tfcochleagram_code()
