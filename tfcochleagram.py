from __future__ import division

import numpy as np
import tensorflow as tf

from pycochleagram import cochleagram as cgram
from pycochleagram import erbfilter as erb

import scipy.signal as signal
import functools

# TODO: the gradients through the log function will be unstable. Include offset or similar for stability, 
def tflog10(x):
    """Implements log base 10 in tensorflow """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

@tf.custom_gradient
def clipped_power_compression(x):
    """
    Clip the gradients for the power compression and remove nans. Clipped values are (-1,1), so any cochleagram value below ~0.2 will be clipped.

    Recommended power compression if using the gradients to the waveform. 
    """
    e = tf.nn.relu(x) # add relu to x to avoid NaN in loss
    p = tf.pow(e,0.3)
    def grad(dy): #try to check for nans before we clip the gradients. (use tf.where)
        g = 0.3 * pow(e,-0.7)
        is_nan_values = tf.is_nan(g)
        replace_nan_values = tf.ones(tf.shape(g), dtype=tf.float32)*1
        return dy * tf.where(is_nan_values,replace_nan_values,tf.clip_by_value(g, -1, 1))
    return p, grad

def cochleagram_graph(nets, SR, ENV_SR=400, LOW_LIM=50, HIGH_LIM=8000, N=40, SAMPLE_FACTOR=4, compression='clipped_point3', WINDOW_SIZE=1001, subbands_ifft=False, pycoch_downsamp=False, input_node='input_signal', mean_subtract=False, rms_normalize=False, SMOOTH_ABS=False, include_all_keys=False, pad_factor=None, return_coch_params=False, rFFT=False, linear_params=None, custom_filts=None, custom_compression_op=None, erb_filter_kwargs={'no_lowpass':True, 'no_highpass':True}, reshape_kell2018=False, **kwargs):
    """
    Creates a tensorflow cochleagram graph using the pycochleagram erb filters to create the cochleagram with the tensorflow functions.

    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. At a minumum, nets['input_signal'] (or equivilant) should be defined containing a placeholder (if just constructing cochleagrams) or a variable (if optimizing over the cochleagrams), and can have a batch size>1.
    SR : int
        raw sampling rate in Hz for the audio.
    ENV_SR : int
        the sampling rate of the cochleagram after downsampling
    LOW_LIM : int
        Lowest frequency of the non-overcomplete filters. NOTE: if SAMPLE_FACTOR>1, then some filters will have power below this value. 
    HIGH_LIM : int
        Highest frequency placement of the non-overcomplete filters (generally the nyquist of the SR). NOTE: if SAMPLE_FACTOR>1 then some filters will have power above this value.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    compression : string. see include_compression for compression options
        determine compression type to use in the cochleagram graph.
    WINDOW_SIZE : int
        the size of a window to use for the downsampling filter
    input_node : string
        Name of the top level of nets, this is the input into the cochleagram graph. 
    mean_subtract : boolean
        If true, subtracts the mean of the waveform (explicitly removes the DC offset)
    rms_normalize : Boolean
        If true, divides the input signal by its RMS value, such that the RMS value of the sound going into the cochleagram generation is equal to 1. This option should be false if inverting cochleagrams, as it can cause problems with the gradients (#TODO)
    SMOOTH_ABS : Boolean
        If True, uses a smoother version of the absolute value for the hilbert transform sqrt(10^-3 + real(env) + imag(env)). Might help with some instability.
    include_all_keys : Boolean 
        If True, returns all of the cochleagram and subbands processing keys in the dictionary
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    return_coch_params : Boolean
        If True, returns the cochleagram generation parameters in addition to nets
    rFFT : Boolean
        If True, builds the graph using rFFT and irFFT operations whenever possible
    linear_params : list of floats
        used for the linear compression operation, [m, b] where the output of the compression is y=mx+b. m and b can be vectors of shape [1,num_filts,1] to apply different values to each frequency channel. 
    custom_filts : None, or numpy array
        if not None, a numpy array containing the filters to use for the cochleagram generation. If none, uses erb.make_erb_cos_filters from pycochleagram to construct the filterbank. If using rFFT, should contain the full filters, shape [SIGNAL_SIZE, NUMBER_OF_FILTERS]
    custom_compression_op : None or tensorflow partial function
        if specified as a function, applies the tensorflow function as a custom compression operation. Should take the input node and 'name' as the arguments
    erb_filter_kwargs : dictionary
        contains additional arguments with filter parameters to use with erb.make_erb_cos_filters
    reshape_kell2018 : boolean (False)
        if true, reshapes the output cochleagram to be 256x256 as used by kell2018
        
    Returns
    -------
    nets : dictionary
        a dictionary containing the parts of the cochleagram graph. Top node in this graph is nets['output_tfcoch_graph']
    COCH_PARAMS : dictionary (Optional)
        a dictionary containing all of the input parameters into the function


    """

    # Get the size of the input signal for graph construction 
    input_signal_shape = nets['input_signal'].get_shape().as_list()
    assert len(input_signal_shape)==2, "nets['input_signal'] must have shape [batch_size, audio_len]"
    SIGNAL_SIZE = input_signal_shape[-1]

    # Useful for saving the parameters to reproduce experiments later
    if return_coch_params: 
        COCH_PARAMS = locals()
        COCH_PARAMS.pop('nets')

    # Make a convenience wrapper for the compression function
    compression_function = functools.partial(include_compression, compression=compression, linear_params=linear_params, custom_compression_op=custom_compression_op)

    # run preprocessing operations on the input (ie rms normalization, convert to complex)
    nets = preprocess_input(nets, SIGNAL_SIZE, input_node, mean_subtract, rms_normalize, rFFT)

    # fft of the input (filtering occurs in frequency domain)
    nets = fft_of_input(nets, pad_factor, rFFT)

    # make cochlear filters and compute the cochlear subbands
    nets = extract_cochlear_subbands(nets, SIGNAL_SIZE, SR, LOW_LIM, HIGH_LIM, N, SAMPLE_FACTOR, pad_factor, rFFT, custom_filts, erb_filter_kwargs, include_all_keys, compression_function)
       
    # hilbert transform on subband fft
    nets = hilbert_transform_from_fft(nets, SR, SIGNAL_SIZE, pad_factor, rFFT)

    # absolute value of the envelopes (and expand to one channel)
    nets = abs_envelopes(nets, SMOOTH_ABS)

    # downsample and rectified nonlinearity
    nets = downsample_and_rectify(nets, SR, ENV_SR, WINDOW_SIZE, pycoch_downsamp)

    # compress cochleagram 
    nets = compression_function(nets, input_node_name='cochleagram_no_compression', output_node_name='cochleagram')

    # reshape the cochleagram to 256x256 as in kell et al. 2018 (WARNING: USES TFRESIZE WHICH ISN'T TO BE TRUSTED!)
    if reshape_kell2018:
        nets, output_node_name_coch = reshape_coch_kell_2018(nets)

    # return 
    if return_coch_params:
        return nets, COCH_PARAMS    
    else: 
        return nets

def preprocess_input(nets, SIGNAL_SIZE, input_node, mean_subtract, rms_normalize, rFFT):
    """
    Does preprocessing on the input (rms and converting to complex number)

    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. should already contain input_node
    input_node : string
        Name of the top level of nets, this is the input into the cochleagram graph.
    mean_subtract : boolean
        If true, subtracts the mean of the waveform (explicitly removes the DC offset)
    rms_normalize : Boolean # TODO: incorporate stable gradient code for RMS
        If true, divides the input signal by its RMS value, such that the RMS value of the sound going 
    rFFT : Boolean
        If true, preprocess input for using the rFFT operations

    Returns
    -------
    nets : dictionary
        updated dictionary containing parts of the cochleagram graph.

    """
    
    if rFFT:
        if SIGNAL_SIZE%2!=0:
            print('rFFT is only tested with even length signals. Change your input length.')
            return
    
    processed_input_node = input_node
    
    if mean_subtract:
        processed_input_node = processed_input_node + '_mean_subtract'
        nets[processed_input_node] = nets[input_node] - tf.reshape(tf.reduce_mean(nets[input_node],1),(-1,1))
        input_node = processed_input_node 
    
    if rms_normalize: # TODO: incoporate stable RMS normalization
        processed_input_node = processed_input_node + '_rms_normalized'
        nets['rms_input'] = tf.sqrt(tf.reduce_mean(tf.square(nets[input_node]), 1))
        nets[processed_input_node] = tf.identity(nets[input_node]/tf.reshape(nets['rms_input'],(-1,1)),'rms_normalized_input')
        input_node = processed_input_node
    
    if not rFFT:
        nets['input_signal_i'] = nets[input_node]*0.0
        nets['input_signal_complex'] = tf.complex(nets[input_node], nets['input_signal_i'], name='input_complex')
    else:
        nets['input_real'] = nets[input_node]
    return nets

def fft_of_input(nets, pad_factor, rFFT):
    """
    Computs the fft of the signal and adds appropriate padding
    
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. 'subbands' are used for the hilbert transform
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    rFFT : Boolean
        If true, cochleagram graph is constructed using rFFT wherever possible
    Returns
    -------
    nets : dictionary
        updated dictionary containing parts of the cochleagram graph with the rFFT of the input
    """
    # fft of the input
    if not rFFT:
        if pad_factor is not None:
            nets['input_signal_complex'] = tf.concat([nets['input_signal_complex'], tf.zeros([nets['input_signal_complex'].get_shape()[0], nets['input_signal_complex'].get_shape()[1]*(pad_factor-1)], dtype=tf.complex64)], axis=1)
        nets['fft_input'] = tf.fft(nets['input_signal_complex'],name='fft_of_input')
    else: 
        nets['fft_input'] = tf.spectral.rfft(nets['input_real'],name='fft_of_input') # Since the DFT of a real signal is Hermitian-symmetric, RFFT only returns the fft_length / 2 + 1 unique components of the FFT: the zero-frequency term, followed by the fft_length / 2 positive-frequency terms.

    nets['fft_input'] = tf.expand_dims(nets['fft_input'], 1, name='exd_fft_of_input')

    return nets

def extract_cochlear_subbands(nets, SIGNAL_SIZE, SR, LOW_LIM, HIGH_LIM, N, SAMPLE_FACTOR, pad_factor,  rFFT, custom_filts, erb_filter_kwargs, include_all_keys, compression_function):
    """
    Computes the cochlear subbands from the fft of the input signal
    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. 'fft_input' is multiplied by the cochlear filters
    SIGNAL_SIZE : int
        the length of the audio signal used for the cochleagram graph
    SR : int
        raw sampling rate in Hz for the audio.
    LOW_LIM : int
        Lower frequency limits for the filters.
    HIGH_LIM : int
        Higher frequency limits for the filters.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    rFFT : Boolean
        If true, cochleagram graph is constructed using rFFT wherever possible
    custom_filts : None, or numpy array
        if not None, a numpy array containing the filters to use for the cochleagram generation. If none, uses erb.make_erb_cos_filters from pycochleagram to construct the filterbank. If using rFFT, should contain th full filters, shape [SIGNAL_SIZE, NUMBER_OF_FILTERS]
    erb_filter_kwargs : dictionary
        contains additional arguments with filter parameters to use with erb.make_erb_cos_filters
    include_all_keys : Boolean
        If True, includes the time subbands and the cochleagram in the dictionary keys
    compression_function : function
        A partial function that takes in nets and the input and output names to apply compression 

    Returns
    -------
    nets : dictionary
        updated dictionary containing parts of the cochleagram graph.
    """

    # make the erb filters tensor
    nets['filts_tensor'] = make_filts_tensor(SIGNAL_SIZE, SR, LOW_LIM, HIGH_LIM, N, SAMPLE_FACTOR, use_rFFT=rFFT, pad_factor=pad_factor, custom_filts=custom_filts, erb_filter_kwargs=erb_filter_kwargs)

    # make subbands by multiplying filts with fft of input
    nets['subbands'] = tf.multiply(nets['filts_tensor'],nets['fft_input'],name='mul_subbands')

    # make the time the keys in the graph if we are returning all keys (otherwise, only return the subbands in fourier domain)
    if include_all_keys:
        if not rFFT:
            nets['subbands_ifft'] = tf.real(tf.ifft(nets['subbands'],name='ifft_subbands'),name='ifft_subbands_r')
        else:
            nets['subbands_ifft'] = tf.spectral.irfft(nets['subbands'],name='ifft_subbands')
        nets['subbands_time'] = nets['subbands_ifft']

    return nets


def hilbert_transform_from_fft(nets, SR, SIGNAL_SIZE, pad_factor, rFFT):
    """
    Performs the hilbert transform from the subband FFT -- gets ifft using only the real parts of the signal

    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. 'subbands' are used for the hilbert transform
    SR : int
        raw sampling rate in Hz for the audio.
    SIGNAL_SIZE : int
        the length of the audio signal used for the cochleagram graph
    pad_factor : int
        how much padding to add to the signal. Follows conventions of pycochleagram (ie pad of 2 doubles the signal length)
    rFFT : Boolean
        If true, cochleagram graph is constructed using rFFT wherever possible

    """

    if not rFFT:
        # make the step tensor for the hilbert transform (only keep the real components)
        if pad_factor is not None:
            freq_signal = np.fft.fftfreq(SIGNAL_SIZE*pad_factor, 1./SR)
        else:
            freq_signal = np.fft.fftfreq(SIGNAL_SIZE,1./SR)
        nets['step_tensor'] = make_step_tensor(freq_signal)

        # envelopes in frequency domain -- hilbert transform of the subbands
        nets['envelopes_freq'] = tf.multiply(nets['subbands'],nets['step_tensor'],name='env_freq')
    else:
        # make the padding to turn rFFT into a step function
        num_filts = nets['filts_tensor'].get_shape().as_list()[1]
        num_batch = tf.shape(nets['subbands'])[0]
        nets['hilbert_padding'] = tf.zeros([num_batch,num_filts,int(SIGNAL_SIZE/2)-1], tf.complex64) 
        nets['envelopes_freq'] = tf.concat([nets['subbands'],nets['hilbert_padding']],2,name='env_freq')

    # fft of the envelopes.
    nets['envelopes_time'] = tf.ifft(nets['envelopes_freq'],name='ifft_envelopes')

    if not rFFT: # TODO: was this a bug in pycochleagram where the pad factor doesn't actually work? 
        if pad_factor is not None:
            nets['envelopes_time'] = nets['envelopes_time'][:,:,:SIGNAL_SIZE]

    return nets

def abs_envelopes(nets, SMOOTH_ABS):
    """
    Absolute value of the envelopes (and expand to one channel), analytic hilbert signal
    
    Parameters
    ----------
    nets : dictionary
        dictionary containing the cochleagram graph. Downsampling will be applied to 'envelopes_time'
    SMOOTH_ABS : Boolean
        If True, uses a smoother version of the absolute value for the hilbert transform sqrt(10^-3 + real(env) + imag(env))

    Returns
    -------
    nets : dictionary
        dictionary containing the updated cochleagram graph
    """

    if SMOOTH_ABS:
        nets['envelopes_abs'] = tf.sqrt(1e-10 + tf.square(tf.real(nets['envelopes_time'])) + tf.square(tf.imag(nets['envelopes_time'])))
    else:
        nets['envelopes_abs'] = tf.abs(nets['envelopes_time'], name='complex_abs_envelopes')
    nets['envelopes_abs'] = tf.expand_dims(nets['envelopes_abs'],3, name='exd_abs_real_envelopes')
    return nets

def downsample_and_rectify(nets, SR, ENV_SR, WINDOW_SIZE, pycoch_downsamp):
    """
    Downsamples the cochleagram and then performs rectification on the output (in case the downsampling results in small negative numbers)

    Parameters
    ----------
    nets : dictionary 
        dictionary containing the cochleagram graph. Downsampling will be applied to 'envelopes_abs'
    SR : int
        raw sampling rate of the audio signal
    ENV_SR : int
        end sampling rate of the envelopes
    WINDOW_SIZE : int
        the size of the downsampling window (should be large enough to go to zero on the edges).
    pycoch_downsamp : Boolean
        if true, uses a slightly different downsampling function

    Returns
    -------
    nets : dictionary
        dictionary containing parts of the cochleagram graph with added nodes for the downsampled subbands

    """
    # The stride for the downsample, works fine if it is an integer.
    assert (SR % ENV_SR == 0), "SR %d is not evenly divisible by ENV_SR %d, only integer downsampling supported"
    DOWNSAMPLE = SR/ENV_SR
    if not ENV_SR == SR:
        # make the downsample tensor
        nets['downsample_filt_tensor'] = make_downsample_filt_tensor(SR, ENV_SR, WINDOW_SIZE, pycoch_downsamp=pycoch_downsamp)
        nets['cochleagram_preRELU']  = tf.nn.conv2d(nets['envelopes_abs'], nets['downsample_filt_tensor'], [1, 1, DOWNSAMPLE, 1], 'SAME',name='conv2d_cochleagram_raw')
    else:
        nets['cochleagram_preRELU'] = nets['envelopes_abs']
    nets['cochleagram_no_compression'] = tf.nn.relu(nets['cochleagram_preRELU'], name='coch_no_compression')

    return nets

def include_compression(nets, compression='clipped_point3', input_node_name='cochleagram_no_compression', output_node_name='cochleagram', linear_params=None, custom_compression_op=None):
    """
    Choose compression operation to use and adds appropriate nodes to nets

    Parameters
    ----------
    nets : dictionary
        dictionary containing parts of the cochleagram graph. Compression will be applied to input_node_name
    compression : string
        type of compression to perform
    input_node_name : string
        name in nets to apply the compression
    output_node_name : string
        name in nets that will be used for the following operation (default is cochleagram, but if returning subbands than it can be chaged)
    linear_params : list of floats
        used for the linear compression operation, [m, b] where the output of the compression is y=mx+b. m and b can be vectors of shape [1,num_filts,1] to apply different values to each frequency channel.
    custom_compression_op : None or tensorflow partial function
        if specified as a function, applies the tensorflow function as a custom compression operation. Should take the input node and 'name' as the arguments

    Returns
    -------
    nets : dictionary
        dictionary containing parts of the cochleagram graph with added nodes for the compressed cochleagram 

    """
    # 0.3 power compression, "human-like"
    if compression=='point3':
        nets[output_node_name] = tf.pow(nets[input_node_name],0.3, name=output_node_name)
    # 0.3 powercompression, with gradient clipping
    elif (compression=='clipped_point3'):
        nets[output_node_name] = tf.identity(clipped_power_compression(nets[input_node_name]),name=output_node_name) 
    # No compression
    elif compression=='none':
        nets[output_node_name] = nets[input_node_name]
    # dB scale the cochleagrams 
    elif compression=='dB': # NOTE: this compression does not work well for the backwards pass, results in nans
        nets[output_node_name + '_noclipped'] = 20 * tflog10(nets[input_node_name])/tf.reduce_max(nets[input_node_name])
        nets[output_node_name] = tf.maximum(nets[output_node_name + '_noclipped'], -60)
    # scale and offset the cochleagrams
    elif compression=='linear':
        assert (type(linear_params)==list) and len(linear_params)==2, "Specifying linear compression but not specifying the compression parameters in linear_params=[m, b]"
        nets[output_node_name] = linear_params[0]*nets[input_node_name] + linear_params[1]
    # provided a custom compression op in tensorflow
    elif compression=='custom':
        assert (custom_compression_op is not None), "Must specify custom compression as a tensorflow partial function."
        nets[output_node_name] = custom_compression_op(nets[input_node_name], name=output_node_name)
    else:
        raise ValueError('Compression %s is not supported'%compression)

    return nets

def make_step_tensor(freq_signal):
    """
    Make step tensor for calcaulting the anlyatic envelopes.

    Parameters
    __________
    freq_signal : array
        numpy array containing the frequenies of the audio signal (as calculated by np.fft.fftfreqs).

    Returns
    -------
    step_tensor : tensorflow tensor
        tensorflow tensor with dimensions [0 len(freq_signal) 0 0] as a step function where frequencies > 0 are 1 and frequencies < 0 are 0.
    """
    step_func = (freq_signal>=0).astype(np.int)*2 # wikipedia says that this should be 2x the original.
    step_func[freq_signal==0] = 0 # https://en.wikipedia.org/wiki/Analytic_signal
    step_tensor = tf.constant(step_func, dtype=tf.complex64)
    step_tensor = tf.expand_dims(step_tensor, 0)
    step_tensor = tf.expand_dims(step_tensor, 1)
    return step_tensor

def make_filts_tensor(SIGNAL_SIZE, SR, LOW_LIM, HIGH_LIM, N, SAMPLE_FACTOR, use_rFFT=False, pad_factor=None, custom_filts=None, erb_filter_kwargs={}):
    """
    Use pycochleagram to make the filters using the specified prameters (make_erb_cos_filters_nx). Then input them into a tensorflow tensor to be used in the tensorflow cochleagram graph.

    Parameters
    ----------
    SIGNAL_SIZE: int
        length of the audio signal to convert, and the size of cochleagram filters to make.
    SR : int
        raw sampling rate in Hz for the audio.
    LOW_LIM : int
        Lower frequency limits for the filters.
    HIGH_LIM : int
        Higher frequency limits for the filters.
    N : int
        Number of filters to uniquely span the frequency space
    SAMPLE_FACTOR : int
        number of times to overcomplete the filters.
    use_rFFT : Boolean
        if True, the only returns the first half of the filters, corresponding to the positive component. 
    custom_filts : None, or numpy array
        if not None, a numpy array containing the filters to use for the cochleagram generation. If none, uses erb.make_erb_cos_filters from pycochleagram to construct the filterbank. If using rFFT, should contain th full filters, shape [SIGNAL_SIZE, NUMBER_OF_FILTERS]
    erb_filter_kwargs : dictionary 
        contains additional arguments with filter parameters to use with erb.make_erb_cos_filters

    Returns
    -------
    filts_tensor : tensorflow tensor, complex
        tensorflow tensor with dimensions [0 SIGNAL_SIZE NUMBER_OF_FILTERS] that includes the erb filters created from make_erb_cos_filters_nx in pycochleagram
    """
    if pad_factor:
        padding_size = (pad_factor-1)*SIGNAL_SIZE
    else:
        padding_size=None

    if custom_filts is None: 
        # make the filters using pycochleagram
        filts, hz_cutoffs, freqs = erb.make_erb_cos_filters_nx(SIGNAL_SIZE, SR, N, LOW_LIM, HIGH_LIM, SAMPLE_FACTOR, padding_size=padding_size, **erb_filter_kwargs)
    else: 
        assert custom_filts.shape[1] == SIGNAL_SIZE, "CUSTOM FILTER SHAPE DOES NOT MATCH THE INPUT AUDIO SHAPE"
        filts = custom_filts

    if not use_rFFT: 
        filts_tensor = tf.constant(filts, tf.complex64)
    else:
        filts_tensor = tf.constant(filts[:,0:(int(SIGNAL_SIZE/2)+1)], tf.complex64)

    filts_tensor = tf.expand_dims(filts_tensor, 0)

    return filts_tensor


def make_downsample_filt_tensor(SR=16000, ENV_SR=200, WINDOW_SIZE=1001, pycoch_downsamp=False):
    """
    Make the sinc filter that will be used to downsample the cochleagram

    Parameters
    ----------
    SR : int
        raw sampling rate of the audio signal
    ENV_SR : int
        end sampling rate of the envelopes
    WINDOW_SIZE : int
        the size of the downsampling window (should be large enough to go to zero on the edges).
    pycoch_downsamp : Boolean
        if true, uses a slightly different downsampling function

    Returns
    -------
    downsample_filt_tensor : tensorflow tensor, tf.float32
        a tensor of shape [0 WINDOW_SIZE 0 0] the sinc windows with a kaiser lowpass filter that is applied while downsampling the cochleagram

    """
    DOWNSAMPLE = SR/ENV_SR
    if not pycoch_downsamp: 
        downsample_filter_times = np.arange(-WINDOW_SIZE/2,int(WINDOW_SIZE/2))
        downsample_filter_response_orig = np.sinc(downsample_filter_times/DOWNSAMPLE)/DOWNSAMPLE
        downsample_filter_window = signal.kaiser(WINDOW_SIZE, 5)
        downsample_filter_response = downsample_filter_window * downsample_filter_response_orig
    else: 
        max_rate = DOWNSAMPLE
        f_c = 1. / max_rate  # cutoff of FIR filter (rel. to Nyquist)
        half_len = 10 * max_rate  # reasonable cutoff for our sinc-like function
        if max_rate!=1:    
            downsample_filter_response = signal.firwin(2 * half_len + 1, f_c, window=('kaiser', 5.0))
        else:
            downsample_filter_response = zeros(2 * half_len + 1)
            downsample_filter_response[half_len + 1] = 1
            
    downsample_filt_tensor = tf.constant(downsample_filter_response, tf.float32)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 0)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 2)
    downsample_filt_tensor = tf.expand_dims(downsample_filt_tensor, 3)

    return downsample_filt_tensor


def reshape_coch_kell_2018(nets):
    """
    Wrapper to reshape the cochleagram to 256x256 similar to that used in kell2018.
    Note that this function relies on tf.image.resize_images which can have unexpected behavior... use with caution.

    nets : dictionary
        dictionary containing parts of the cochleagram graph. should already contain cochleagram
    """
    print('### WARNING: tf.image.resize_images is not trusted, use caution ###')
    nets['min_cochleagram'] = tf.reduce_min(nets['cochleagram'])
    nets['max_cochleagram'] = tf.reduce_max(nets['cochleagram'])
    # it is possible that this scaling is going to mess up the gradients for the waveform generation
    nets['scaled_cochleagram'] = 255*(1-((nets['max_cochleagram']-nets['cochleagram'])/(nets['max_cochleagram']-nets['min_cochleagram'])))
    nets['reshaped_cochleagram'] = tf.image.resize_images(nets['scaled_cochleagram'],[256,256], align_corners=False, preserve_aspect_ratio=False)
    return nets, 'reshaped_cochleagram'

