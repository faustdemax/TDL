# TDL
**CentraleSupélec Theory of Deep Learning final project**


This project investigates the scaling of output function fluctuations in fully connected neural networks and explores ensemble averaging as an alternative to increasing network width. It is based on the research paper:

_Disentangling Feature and Lazy Training in Deep Neural Networks_ – Geiger et al.

Theoretical predictions suggest that fluctuations in neural network outputs follow the scaling law:

$$ \delta F  \sim h^{-1/2}$$

whereh $h$ is the network width and $F$ the scaled output function. We conduct experiments to empirically verify this result and assess ensemble averaging as a computationally efficient alternative to width increase.
