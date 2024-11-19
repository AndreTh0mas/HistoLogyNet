# Stain Normalization and Augmentation, Group: P17-A

Histology images can have variations in staining styles, which can confuse deep learning models. RandStainNA addresses this by combining two techniques:

Stain Normalization (SN): Reduces these variations by transforming the image colors to a standard range.
Stain Augmentation (SA): Artificially creates new images with different stain variations to increase the variety of data the model sees during training.
By combining these techniques, RandStainNA aims to train models that are less sensitive to stain variations and can perform better on unseen data.

## Key benifits

Better Performance: RandStainNA has been shown to outperform approaches that only use SN or SA in tasks like tissue classification and nuclei segmentation. More Robust Models: Models trained with RandStainNA are expected to be more adaptable to new datasets with different staining styles, improving their generalizabilit

## How to use

Install the required dependencies. Then:

```bash
  cd preprocessing
```

In the `train` folder in this directory, add styles which is in your original data. Each style should signify each type of stain augmentation. Example have been given from the images provided to us.

Run:

```bash
python ./datasets_statistics.py
```

This will create a `.yaml` file in the `./output` directory.
Store your images which you want to augment in the `./data/original` folder. Then go to root directory and run:

```bash
python ./main.py
## OR
python ./Reinhard_stain_norm_only.py
## OR
python ./stain_augmentation_only.py
```

as required. Each cammand will create the normalized or augmented images in the `./data` directory as either `data/augmented` or `data/stain_normalization/` or `data/stain_augmentation` depending on the cammand you use.

## Description

`main.py` is for RandStainNA which creates images in `./data/augmented` directory, `Reinhard_stain_norm_only.py` is for StainNormalization only which creates images in `./data/Stain_normalization` directory, and `stain_augmentation_only.py` is for Augmention only which creates images in `./data/stain_augmentation` directory
