# ONNet

**ONNet** is an open-source Python/C++ package for the optical neural networks, which includes the following types of networks:

- #### Diffractive deep network

- #### Diffractive deep neural network with multiple frequency-channels

  Each layer have multiple frequency-channels (optical distributions at different frequency). These channels are merged at the output plane with weighting coefficient. [1]

- #### Diffractive network with multiple binary output plane

  ![](./ONNet_wavelet.png)

Optical neural network(ONN) is a novel machine learning framework on the physical principles of optics, which is still in its infancy and shows great potential. ONN tries to find optimal modulation parameters to change the phase, amplitude or other physical variable of optical wave propagation. So in the final output plane, the optical distribution has special pattern which is the indicator of object’s class or value. ONN opens new doors for the machine learning.




## Citation

Please use the following bibtex entry:
```
[1] Chen, Yingshi, et al."An optical diffractive deep neural network with multiple frequency-channels." arXiv preprint arXiv:1912.10730 (2019).
```

## Future work

- More testing datasets 

  ​	Cifar, ImageNet ......

- More models.

  ​	Wavefront Matching Method

  ​	Express network	

- More papers.


## License

[MIT](./LICENSE)

## Authors

`ONNet` was written by Yingshi Chen(gsp.cys@gmail.com).

QQ group: 1001583663