# Robustness to Augmentations as a Generalization Metric
Runner Up Solution for the NeurIPS 2020 Competition - [Predicting Generalization in Deep Learning](https://sites.google.com/view/pgdl2020/home)

# Abstract
Generalization is the ability of a model to predict on unseen domains and is a fundamental task in machine learning. Several generalization bounds, both theoretical and empirical have been proposed but they do not provide tight bounds. In this work, we propose a simple yet effective method to predict the generalization performance of a model by using the concept that models that are robust to augmentations are more generalizable than those which are not. We experiment with several augmentations and composition of augmentations to check the generalization capacity of a model. We also provide a detailed motivation behind the proposed method. The proposed generalization metric is calculated based on the change in the model's output after augmenting the input.
The proposed method was the first runner up solution for the competition "Predicting Generalization in Deep Learning".

# Prerequisites:
- Python 3.6.6
- Tensorflow 2.2
- pandas
- pyyaml
- scikit-learn

# How to Run
```
python ingestion_program/ingestion.py PATH_TO_DATA runner_up_solution
```
Please refer to the starting kit provided by [PGDL Competition](https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit) for more instructions on how to run the code.

```complexity.py``` contains the proposed solution.\
For details about the datasets and the competition, please refer to [this repository](https://github.com/google-research/google-research/tree/master/pgdl).
# BibTeX
For more details please refer to our [paper](https://arxiv.org/abs/2101.06459).
```
       @article{aithal2021robustness,
        title={Robustness to Augmentations as a Generalization metric.},
        author={Aithal, Sumukh and Kashyap, Dhruva and Subramanyam, Natarajan},
        journal={arXiv preprint arXiv:2101.06459},
        year={2021}
        }
```

