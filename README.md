# Dynamic Mode Decomposition for Multi-scale Analysis of Crowd Simulation

This work presents an operator-informed approach to analyze patterns of crowd density
during evacuation scenarios using Dynamic Mode Decomposition (DMD) and its variants
(Extended Dynamic Mode Decomposition - EDMD) to approximate the Koopman operator.

The study aims to gain valuable insight into pedestrian behavior and optimize evacuation
strategies using mesoscopic data such as crowd density. The utilization of time delay
embedding (TDE) to construct a richer dataset from the mesoscopic data allows more accurate
approximations of the Koopman operator.

The research involves the collection of crowd behavior data of position and speed through
simulations of bottleneck scenarios within Vadere software. The position and speed data
are further mapped to a mesoscopic representation of crowd density in triangular meshes.
This cellular automata pattern or triangular meshes are constructed in the topography of the
scenario itself during simulation. This dataset serves as the foundation for constructing state
space representations, where crowd density is defined as an explicit discrete time-invariant
parameter.

The triangular meshes are chosen due to ease of quantification and computation. By
employing DMD techniques, the data is decomposed into modes, enabling the computation
and prediction of key macroscopic parameters such as the number of pedestrians evacuated
from an area and further providing reconstruction of new test cases. The results showcase
the application and efficacy of DMD and its variants (EDMD) in capturing the underlying
dynamics of crowd movement, particularly in bottleneck scenarios. This includes optimizing
the model by performing hyperparameter tuning of attributes involved in TDE and DMD. The
computed outputs offer valuable insights into the spatio-temporal evolution of crowd density,
aiding in the identification of critical congestion points and optimal evacuation routes.

The work demonstrates the applicability of DMD and EDMD as a useful tool for analyzing
complex crowd dynamics and predicting the evolution of crowd density patterns. The results
provide the importance and benefits of considering crowd density in triangular meshes or
grids within a state space framework for a better understanding of pedestrian flow and crowd
management in bottleneck scenarios.
