Handling Uncertainty in Self-adaptive Systems: An Ontology-based Reinforcement Learning Model

Ubiquitous and pervasive systems interact with each other and perform actions favoring the emergence of a global desired behavior. These systems' self-adaptability means adapting to dynamic, noisy, and partially unknown environments. The difficulty in predicting which environmental conditions systems will encounter and imperfect information about their environment cause uncertainty in adaptation. Self-adaptive systems can cope with such unpredictable conditions by using adaptive modeling mechanisms to select their default actions, exploiting machine learning and Reinforcement Learning (RL) algorithms to learn new actions, or using a default fail-safe action in an over-conservative way. However, the modelling mechanisms do not specify how rare events should be considered. Also, exploring the huge number of possible adaptation actions may compromise the system's on the fly decision-making. If such a problem is not addressed, uncertainty will result in inconsistencies in decisions and unexpected system behavior.

To address this problem, we propose an Ontology-based unCertainty handling model (OnCertain), which enables the RL-based system to augment its observation and reason about the unanticipated situations using prior ontological knowledge. Our model is evaluated in a traffic signal control system and an edge computing environment. The results show that the OnCertain model can improve the RL-based systems' observation and, consequently, their performance in such environments.

This repository provides the implementation of the Ontology-based unCertainty handling model (OnCertain). 

References

Baseline Code

https://github.com/davidtw0320/Resources-Allocation-in-The-Edge-Computing-Environment-Using-Reinforcement-Learning

Mobility Data

Mongnam Han, Youngseok Lee, Sue B. Moon, Keon Jang, Dooyoung Lee, CRAWDAD dataset kaist/wibro (v. 2008‑06‑04), downloaded from https://crawdad.org/kaist/wibro/20080604, https://doi.org/10.15783/C72S3B, Jun 2008.

