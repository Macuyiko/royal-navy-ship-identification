
@echo off
for %%x in (
        basic_grey
        convmixer
        convnexts
        convnextm
        convnextl
        efficientnetb0
        efficientnetv2b0
        efficientnetv2s
        efficientnetv2m
        resnet50
        resnet50v2
       ) do (
        python scripts/train-model.py %%x 0 0 0
        python scripts/train-model.py %%x 0 0 1
        python scripts/train-model.py %%x 0 1 0
        python scripts/train-model.py %%x 1 0 0
        python scripts/train-model.py %%x 1 0 1
        python scripts/train-model.py %%x 1 1 0
        python scripts/train-model.py %%x 1 1 1
       )
