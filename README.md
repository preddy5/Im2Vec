# Im2Vec: Synthesizing Vector Graphics without Vector Supervision

Vector graphics are widely used to represent fonts, logos, digital artworks, and graphic designs. But, while a vast body of work has focused on generative algorithms for raster images, only a handful of options exists for vector graphics. One can always rasterize the input graphic and resort to image-based generative approaches, but this negates the advantages of the vector representation. The current alternative is to use specialized models that require explicit supervision on the vector graphics representation at training time. This is not ideal because large-scale high quality vector-graphics datasets are difficult to obtain. Furthermore, the vector representation for a given design is not unique, so models that supervise on the vector representation are unnecessarily constrained. Instead, we propose a new neural network that can generate complex vector graphics with varying topologies, and only requires indirect supervision from readily-available raster training images (i.e., with no vector counterparts). To enable this, we use a differentiable rasterization pipeline that renders the generated vector shapes and composites them together onto a raster canvas. We demonstrate our method on a range of datasets, and provide comparison with state-of-the-art SVG-VAE and DeepSVG, both of which require explicit vector graphics supervision. Finally, we also demonstrate our approach on the MNIST dataset, for which no groundtruth vector representation is available.

Website: http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/

<img src="http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/paper_docs/teaser.png">


# Usage

Training

`CUDA_VISIBLE_DEVICES=1 python run.py -c configs/emoji.yaml`

Inference

```
cd ./logs/VectorVAEnLayers/version_110
wget  http://geometry.cs.ucl.ac.uk/projects/2021/im2vec/paper_docs/epoch=667.ckpt
CUDA_VISIBLE_DEVICES=1 python eval_local.py -c configs/emoji.yaml
```


*Note that I have an example of the training in the logs directory. The logs directory run is only for the sake of showing what to expect if everything is working.*


## Citation
```
@article{reddy2021im2vec,
  title={Im2Vec: Synthesizing Vector Graphics without Vector Supervision},
  author={Reddy, Pradyumna and Gharbi, Michael and Lukac, Michal and Mitra, Niloy J},
  journal={arXiv preprint arXiv:2102.02798},
  year={2021}
}
```
