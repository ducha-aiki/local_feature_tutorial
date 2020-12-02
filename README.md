# Hands-on wide baseline tutorial
> Summary description here.


We will create the wide baseline stereo mather and try it on various images with various features. There is also a a (naive) example of the spatial verification together with image retrieval. We will not build the components from scratch, instead will be using a ready packages, like [kornia](https://github.com/kornia/kornia), [pydegensac](https://pypi.org/project/pydegensac/) and [OpenCV](https://github.com/opencv/opencv)

## Install

`pip install local_feature_tutorial`

## How to use

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from local_feature_tutorial.wbs import *
from local_feature_tutorial.visualization import *
from local_feature_tutorial.io import *
from local_feature_tutorial.datasets import *
import cv2

hard_images_to_match = 'http://cmp.felk.cvut.cz/~mishkdmy/wbs_illum.tgz'
fname = download_file(hard_images_to_match)


untar_to(fname, 'data/wbs')

wbs_img_fnames = get_all_images_in_subdirs('data/wbs')
print (wbs_img_fnames)

visualize_grid(wbs_img_fnames, (8,8))
```

    ['data/wbs/chimera_01.png', 'data/wbs/chimera_02.png', 'data/wbs/dh_01.png', 'data/wbs/dh_02.png', 'data/wbs/doll_theater1.jpeg', 'data/wbs/doll_theater2.jpeg', 'data/wbs/doll_theater3.jpeg', 'data/wbs/kn_church-2.jpg', 'data/wbs/kn_church-8.jpg', 'data/wbs/ministry_01.png', 'data/wbs/ministry_02.png', 'data/wbs/ministry_03.png', 'data/wbs/purkine-2.jpg', 'data/wbs/purkine-4.jpg']



![png](docs/images/output_4_1.png)


```python
sift_hardnet_matcher = TwoViewMatcher(detector=cv2.SIFT_create(8000), descriptor=HardNetDesc(),
                              matcher=SNNMMatcher(0.9), geom_verif=degensac_Verifier(0.5))

res = sift_hardnet_matcher.verify(wbs_img_fnames[7], wbs_img_fnames[8])
print (res.keys())
draw_matches(res['match_kpts1'], res['match_kpts2'],
                wbs_img_fnames[7], wbs_img_fnames[8], color=(0,255,0), R=10)
```

    dict_keys(['match_kpts1', 'match_kpts2', 'F', 'num_inl'])



![png](docs/images/output_5_1.png)

