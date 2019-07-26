# vae_oversampler

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

vae_oversampler provides an API similar to imblearn to oversample a minority class of a dataset. Under the hood it uses keras to build a variational autoencoder that learns the underlying data probability distribution and then samples from that distribution to generate synthetic minority examples.


### Tech

vae_oversampler uses a number of open source projects to work properly:

* [keras] - for deep learning- to build the variational autoencoder
* [sklearn] - primarily to standard scale your data (optional)
* [numpy] - numerical methods

And of course vae_oversampler itself is open source with a [public repository][vae_oversampler] on GitHub.

### Installation

vae_oversampler requires keras to run. 

Install the dependencies and install using pip

```sh
$ pip install vae_oversampler
```

### Todos

 - Write Tests
 - Comply with PEP8
 - Better error handling
 - Add more options for how many samples to generate

License
----

MIT



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [vae_oversampler]: <https://github.com/dyanni3/vae_oversampler>
