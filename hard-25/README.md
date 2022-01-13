Handling Uncertainty for Self-adaptation in Dynamic Environments: An Ontology-based Model

Ubiquitous and pervasive systems interact with each other and perform actions favoring the emergence of a global desired behavior. These systems must be self-adaptive, which means adapting themselves to dynamic environments. Self-adaptive systems depend on models of themselves and their environment, learned by observing the environment's states and consequences of actions, to decide whether and how to adapt, but these states are often affected by the uncertainty that can arise from entities' unpredictable characteristics and the changes they cause in their dynamic environment. 

Reducing the uncertainty results in a better characterization of the current and future states of the system and the environment, which supports making better adaptation decisions. Various techniques have been proposed to handle uncertainty including evolutionary, machine learning, and genetic algorithms. However, they depend on the availability of large amounts of data and when the number of adaptation decisions increases and on the fly decision-making is required, they cannot perform well. If such a problem is not addressed, uncertainty will result in inconsistencies in decisions and unexpected system behavior.

To address this problem, we propose an Ontology-based unCertainty handling model (OnCertain), which enables the system to augment its observation and reason about the unanticipated situations using prior ontological knowledge. Our model is evaluated in an adaptive traffic signal control system and an edge computing environment. The results show that the OnCertain model can improve the RL systems' observation and consequently their performance in such environments. 

This repository provides the implementation of the Ontology-based unCertainty handling model (OnCertain). If you use this code please cite the paper using the following bibtex:

