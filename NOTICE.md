# NOTICE

## Third-Party Code Attribution

Parts of the data-loading and batch-combination logic in:

```text
code/matlab/preprocessing/preprocessing.m
```

were adapted from the `LoadData.m` workflow made available in the repository associated with:

> Severson, K. A., Attia, P. M., Jin, N., Perkins, N., Jiang, B., Yang, Z., Chen, M. H., Aykol, M., Herring, P. K., Fraggedakis, D., Bazant, M. Z., Harris, S. J., Chueh, W. C., and Braatz, R. D. Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 2019, 4, 383–391. https://doi.org/10.1038/s41560-019-0356-8

Original workflow source:

https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation/blob/master/LoadData.m

The present repository adapts this logic to support the preprocessing, cleaning, alignment, and subsequent state-of-charge estimation workflow described in the associated article.

## Dataset Attribution

The raw battery data used by this workflow are obtained separately from the MATR data repository and are not redistributed in this repository:

https://data.matr.io/1/projects/5c48dd2bc625d700019f3204

Users of the data must comply with the original dataset terms and citation requirements.

## License Scope

The MIT License included in this repository applies only to original code and documentation developed for this repository, unless otherwise stated.

External datasets and third-party code are not relicensed by this repository and remain subject to their respective original terms, licenses, and attribution requirements.
