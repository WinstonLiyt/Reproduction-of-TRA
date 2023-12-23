## Analysis and reproduction of paper: Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport
### Running
* Running from code with self-defined parameters

    Setting different parameters is also allowed. See codes in `example.py`:

    ```
    python example.py --config_file configs/config_alstm.yaml
    ```

Here we trained TRA on a pretrained backbone model. Therefore we run `*_init.yaml` before TRA's scripts.

---
### Results

After running the scripts, you can find result files in path `./output`:

* `info.json` - config settings and result metrics.
* `log.csv` - running logs.
* `model.bin` - the model parameter dictionary.
* `pred.pkl` - the prediction scores and output for inference.
---
### Cite
If you find the work useful in your research, please cite:

```
@inproceedings{HengxuKDD2021,
 author = {Hengxu Lin and Dong Zhou and Weiqing Liu and Jiang Bian},
 title = {Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport},
 booktitle = {Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
 series = {KDD '21},
 year = {2021},
 publisher = {ACM},
}

@article{yang2020qlib,
  title={Qlib: An AI-oriented Quantitative Investment Platform},
  author={Yang, Xiao and Liu, Weiqing and Zhou, Dong and Bian, Jiang and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2009.11189},
  year={2020}
}
```
