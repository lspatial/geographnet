# -*- coding: utf-8 -*-
""" Library of Geographic Graph Hybrid Network (geographnet)

The python library of geographic graph hybrid network with attention layers (geographnet).
Current version just supports the PyTorch package of deep geometric learning and
will extend to the others in the future. This package is for the paper,
`"Geographic Graph Hybrid Network for Robust Inversion of Particular Matters"` to be published
in Remote Sensing

geographnet requires installation of PyTorch with support of  PyG (PyTorch Geometric).
Also Pandas and Numpy should be installed.

Major modules:
    model: include the major functions.
           GEOGCon: Function of geographic graph convolution.
           knnd_geograph: function of knn to support the output of spatial or spatiotemporal weights.
           knngeo: function to retrieve the nearest neighbors with the distance values.
           GeoGraphPNet: Geographic Graph Hybrid Network for prediction of PM2.5 and PM10.
            (multilevel geographic graph convolutions plus full residual deep network).
           DataSamplingDSited: Using distance-weighted kdd to sample the data to get the network
            topology and the corresponding sample data.
           WNeighborSampler: Function of using distance-weighted kdd to obtain the mini-batch data
                    to train and test geographic graph hybrid network.
           rsquared and rmse metrics functions.
    helper functions: training and testing functions for your reference,
            train: Training function of geographic graph hybrid network for PM2.5 and PM10.
            test: Testing function of the trained geographic graph hybrid network for PM2.5 and PM10.

Github source: https://github.com/lspatial/geographnet for the sample data and codes.
Author: Lianfa Li
Date: 2021-09-11

"""

#import pkgutil
#__path__ = pkgutil.extend_path(__path__, __name__)

from geographnet.traintest_pm import train
from geographnet.traintest_pm import test
from geographnet.model.geographpnet import GeoGraphPNet

