
# DCAQ

Code for our paper ”Dual-Centroid Alignment with Confidence-Aware Query Optimization for Transductive Few-Shot Learning“. 

In this paper, we propose a simple and effective method to alleviate this problem, dubbed Dual-Centroid Alignment with Confidence-Aware Query Optimization (DCAQ) . First, the centroids of the support set and query set are computed independently to remove bias and align the distributions using centroid information. Second, high-confidence query samples are selected to refine prototype and centroid computation within an iterative refinement loop. 


# Preprocessing:
For illustration, we use the publicly available algorithm with extracted features.
We use the same backbone network and training strategies as 'S2M2' and 'Baseline'. Please refer to [link](https://github.com/nupurkmr9/S2M2_fewshot) and [link](https://github.com/wyharveychen/CloserLookFewShot) for the backbone training.

Create a  new cache/ file  in the main file

# Evaluate

To show the accuracy of the baseline,

- Run:

```
python evaluate.py --cls cosine --n_shot 1
```

1-shot:  64.82 $\pm​$  0.20

5-shot:  83.88 $\pm​$ 0.13


### The proposed DCAQ

To show the accuracy of the proposed DCAQ,

- Run:

```
python evaluate.py --cls dcaq --n_shot 1
```

1-shot:  81.16 $\pm​$  0.25

5-shot:  87.85 $\pm​$ 0.12



## References 

This code is based on the implementations of  [TCPR](https://github.com/KikimorMay/FSL-TCBR)


