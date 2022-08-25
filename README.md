# Vectorization

**News: Our codes has been merged into ULTRA framework (https://github.com/ULTR-Community/ULTRA).**

This project is based on ULTRA (https://github.com/ULTR-Community/ULTRA). Please see its document for more details.

Our main code is in `ultra/learning_algorithm/vectorization.py`.

## load dataset

Please download datasets first, then run the following command:

```bash
bash ./example/Yahoo/offline_exp_pipeline.sh
bash ./example/Istella-S/offline_exp_pipeline.sh
```

## run experiment

You can modify "dimension=xxx" in the file of ./config/xxx.json, to modify the dimension of `Vectorization`

- Real click on Yahoo!
```
python3 main.py \
    --max_train_iteration=15000 \
    --data_dir=./Yahoo_letor/tmp_data/ \
    --model_dir=./tmp/model/ \
    --output_dir=./tmp/model/output/ \
    --setting_file=./config/vector_tiangong.json
```

- Trust bias on Yahoo!
```
python3 main.py \
    --max_train_iteration=15000 \
    --data_dir=./Yahoo_letor/tmp_data/ \
    --model_dir=./tmp/model/ \
    --output_dir=./tmp/model/output/ \
    --setting_file=./config/vector_trust.json
```

- Real click on Istella-S
```
python3 main.py \
    --max_train_iteration=30000 \
    --steps_per_checkpoint=200 \
    --data_dir=./istella-s-letor/tmp_data/ \
    --model_dir=./tmp/model/ \
    --output_dir=./tmp/model/output/ \
    --setting_file=./config/vector_tiangong.json 
```

- Trust bias on Istella-S
```
python3 main.py \
    --max_train_iteration=30000 \
    --steps_per_checkpoint=200 \
    --data_dir=./istella-s-letor/tmp_data/ \
    --model_dir=./tmp/model/ \
    --output_dir=./tmp/model/output/ \
    --setting_file=./config/vector_trust.json
```

## Citation

Please consider citing the following paper when using our code for your application.

```bibtex
@inproceedings{chen2022scalar,
  title={Scalar is Not Enough: Vectorization-based Unbiased Learning to Rank},
  author={Mouxiang Chen and Chenghao Liu and Zemin Liu and Jianling Sun},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2022}
}
```

