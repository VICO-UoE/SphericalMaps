# Improving Semantic Correspondences with Viewpoint-Guided Spherical Maps (CVPR 2024)

### Installation
To create the training environment, run 
```
conda env create -f sph.yaml
```

### Training

Specify the dataset and log path in `config/conf_file.yaml` then run:

```python
python train.py --config config/conf_file.yaml
```
### Evaluation

To ensure exact comparison, our evaluation is based on that of [sd-dino](https://github.com/Junyi42/sd-dino).
First, create a conda environment following the sd-dino instructions, then run
```
python pck_spair_pascal_sphere.py --SAMPLE 0
```

Additional arguments that we introduced for our method:
* `--SPH` to perform fused evaluation with a pretrained sphere mapper
* `--KAP` to compute Keypoint Average precision instead of PCK
* `--DATA_PATH` to specify the path to the evaluation set
* `--SPH_CKPT_PATH` to specify the path to a spherical mapper checkpoint


### Similar works and acknoledgements

* [sd-dino](https://github.com/Junyi42/sd-dino) investigates unsupervised correspondences emerging from recent deep models that inspired this work
* [geoaware-sc](https://github.com/Junyi42/geoaware-sc) is a followup paper that comes to the same conclusion about geometry-related issues and proposes to fix them at test-time



### Citing
If you find our work useful, please cite:
```BiBTeX
@InProceedings{mariotti2024improving,
    author    = {Mariotti, Octave and Mac Aodha, Oisin and Bilen, Hakan},
    title     = {Improving Semantic Correspondence with Viewpoint-Guided Spherical Maps},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {19521-19530}
}
```



