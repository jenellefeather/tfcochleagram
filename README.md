# tfcochleagram 

Tensorflow wrappers to integrate cochleagram generation (https://github.com/mcdermottLab/pycochleagram) in tensorflow, allowing for gradient computations on the cochleagram generation graph. Cochleagrams are a variation on spectrograms but with filter shapes and widths motivated by human perception. Default arguments use half cosine filters at erb spacing. Custom filters can alternatively be provided. After initial (bandpass) filtering, the signals are envelope extracted, compressed, and downsampled to construct the cochleagram representation. 

## 🚨🚨🚨Version Warning🚨🚨🚨
This project was written in tensorflow v1.13 and is no longer being actively updated for new versions (with no guarantees to work with tensorflow v2+).

A newer repository for cochleagram generation in pytorch is located here: https://github.com/jenellefeather/chcochleagram

## Getting Started

Cochleagram generation code is in `tfcochleagram.py`. Call for generating cochleagrams is of the form: 
    
```
tfcochleagram.cochleagram_graph(nets, SR)
```

where `nets` is a dictionary which must contain the key `input_signal` with the input waveform and SR is the waveform sampling rate (used for filter construction). Other options are documented in the tfcochleagram.cochleagram_graph docstring. 


### Prerequisites
```
pycochleagram: https://github.com/mcdermottLab/pycochleagram
tensorflow (tested on v1.13)
```

### Demo of cochleagram generation: `tfcochleagram demo.ipynb`
Generates cochleagrams and plots the filter response within a tf.Session for an example sound. Includes demo of cochleagrams using parameters similar to [1]. Default erb filter arguments are for cochleagrams with higher time and frequency resolution, similar to those used in [2].

### Basic demonstration of cochleagram inversion via gradient descent: `InversionDemo.ipynb` 
Generates cochleagrams within a tf.Session and inverts them by minimizing the squared error between a cochleagram from a noise signal and a cochleagram from a demo sound. The inversion procedure is initialized with pink noise, however sound specific initialization (ie power matching the sound) may lead to faster inversions. 

The Demo uses lbfgs to perform the optimization, but similar quality can be obtained with first order optimizers such as Adam. Cochleagrams can also be inverted using a griffin-lim like procedure documented in pycochleagram. 

## Authors
* **Jenelle Feather** (https://github.com/jfeather)

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
* Ray Gonzalez
* Mark Saddler
* Alex Durango
* Josh McDermott
* McDermott Lab (https://github.com/mcdermottLab)

## Citation
This code was heavily used for the following paper. If you use this code, we recommend including the following citation: 

```
@inproceedings{feather2019metamers,
  title={Metamers of neural networks reveal divergence from human perceptual systems},
  author={Feather, Jenelle and Durango, Alex and Gonzalez, Ray and McDermott, Josh},
  booktitle={Advances in Neural Information Processing Systems},
  pages={10078--10089},
  year={2019}
}
```

PDF: https://papers.nips.cc/paper/9198-metamers-of-neural-networks-reveal-divergence-from-human-perceptual-systems.pdf

## References
[1] McDermott J. and Simoncelli E. Sound Texture Perception via Statistics of the Auditory Periphery: Evidence from Sound Synthesis. Neuron (2011). 

[2] Feather J. and McDermott J. Auditory texture synthesis from task-optimized convolutional neural networks. Conference on Cognitive Computational Neuroscience (2018). 
