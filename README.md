# Rethinking the Bias of Foundation Model under Long-tailed Distribution

### Abstract 
Long-tailed learning has garnered increasing attention due to its practical significance. Among the various approaches, the fine-tuning paradigm has gained considerable interest with the advent of foundation models. However, most existing methods primarily focus on leveraging knowledge from these models, overlooking the inherent biases introduced by the imbalanced training data they rely on. In this paper, we examine how such imbalances from pre-training affect long-tailed downstream tasks. Specifically, we find the imbalance biases inherited in foundation models on downstream tasks as parameter imbalance and data imbalance. During fine-tuning, we observe that parameter imbalance plays a more critical role, while data imbalance can be mitigated using existing re-balancing strategies. Moreover, we find that parameter imbalance cannot be effectively addressed by current re-balancing techniques, such as adjusting the logits, during training, unlike data imbalance. To tackle both imbalances simultaneously, we build our method on causal learning and view the incomplete semantic factor as the confounder, which brings spurious correlations between input samples and labels. To resolve the negative effects of this, we propose a novel backdoor adjustment method that learns the true causal effect between input samples and labels, rather than merely fitting the correlations in the data. Notably, we achieve an average performance increase of about $1.67\%$ on each dataset.


## Requirements
Please refer to [LIFT](https://github.com/shijxcs/LIFT/blob/main/)

The  version of OpenCLIP is ```huggingface/hub/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/open_clip_pytorch_model.bin```


## Quick start on ImageNet-LT
### Stage1 

```bash
# fine tuning based on CLIP, OpenCLIP, and MetaCLIP
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True select open
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True select meta
```

### Stage2

```bash
# Backdoor adjustment
python main.py -d imagenet_lt -m clip_vit_b16 adaptformer True test_only True clip_model_path $CLIP_model_finetune_dir metaclip_model_path $MetaCLIP_model_finetune_dir openclip_model_path $OpenCLIP_model_finetune_dir 
```

## Acknowledgment

We thank the authors for the following repositories for code reference: [[LIFT]](https://github.com/shijxcs/LIFT/blob/main/).


## Citation

If you find this repo useful for your work, please cite as:

```bibtex
@article{chen2025rethinking,
  title={Rethinking the Bias of Foundation Model under Long-tailed Distribution},
  author={Chen, Jiahao and Qin, Bin and Li, Jiangmeng and Chen, Hao and Su, Bing},
  journal={arXiv preprint arXiv:2501.15955},
  year={2025}
}

@article{chen2025rethinking,
  title={Rethinking the Bias of Foundation Model under Long-tailed Distribution},
  author={Chen, Jiahao and Qin, Bin and Li, Jiangmeng and Chen, Hao and Su, Bing},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```