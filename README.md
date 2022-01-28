# DL-DIY Project on CLIP

In this project, I used the pre-trained CLIP model on four different datasets (DomainNet, Paris Buildings, Oxford Buildings, and Wikiart) in a zero-shot setting and test out its performance.
- `preprocess.ipynb` contains the code to download and preprocess all the four datasets.
- `clip.ipynb` contains the code to test CLIP on the four datasets.
- `dall-e_vae.ipynb` contains the code to train a VAE on the `quickdraw` images of `DomainNet` based on the encoder and decoder architectures used in [DALL-E](https://github.com/openai/DALL-E).
- `encoder.py`, `decoder.py`, `utils.py` are adapted from [the DALL-E repo](https://github.com/openai/DALL-E/tree/master/dall_e).

-----------------

# CLIP

[[Blog]](https://openai.com/blog/clip/) [[Paper]](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf) [[Model Card]](model-card.md) [[Colab]](https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb)

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. We found CLIP matches the performance of the original ResNet50 on ImageNet “zero-shot” without using any of the original 1.28M labeled examples, overcoming several major challenges in computer vision.



## Approach

![CLIP](https://github.com/openai/CLIP/blob/main/CLIP.png)

-----------------

# [[DomainNet Dataset]](http://ai.bu.edu/M3SDA/)

- Data on six domains: clipart, infograph, painting, quickdraw, real, sketch.
- Introduced in *Moment Matching for Multi-Source Domain Adaptation*, CVPR 2019.

![examples](http://ai.bu.edu/M3SDA/imgs/data_examples.png)
![statistics](http://ai.bu.edu/M3SDA/imgs/statistics.png)

## Accuracy
|    Domain    |        Benchmark        |        CLIP Zero-Shot        |        CLIP Linear Probe        |
|:--------------:|:--------------:|:--------------:|:--------------:|
| clipart | 58.6% | 66.50% | 78.14% |
| infograph | 26.0% | 43.37% | 50.41% |
| painting | 52.3% | 59.74% | 74.54% |
| quickdraw | 16.2% | 14.15% | 59.60% |
| real | 62.7% | 80.72% | 87.09% |
| sketch | 49.6% | 49.97% | 69.85% |

-----------------

# Buildings Datasets ([Paris](https://www.robots.ox.ac.uk/~vgg/data/parisbuildings/), [Oxford](https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/))

## Paris Buildings
- 6412 images collected from Flickr for 11 different Paris landmarks + general views in Paris.

### Accuracy
|    Class    |        CLIP Zero-Shot        |        CLIP Linear Probe        |
|:--------------:|:--------------:|:--------------:|
| overall | 50.08% | 79.61% |
| La Defense | 8% | 80% |
| Eiffel Tower | 78% | 4% |
| general | 2% | 76% |
| Hotel des Invalides | 69% | 77% |
| Louvre | 92% | 90 % |
| Moulin Rouge | 87% | 87% |
| Musee d'Orsay | 76% | 86% |
| Notre Dame | 66% | 85% |
| Pantheon | 26% | 83% |
| Pompidou | 77% | 78% |
| Sacre Coeur | 69% | 75% |
| Arc de Triomphe | 79% | 84% |

## Oxford Buildings
- 5062 images collected from Flickr for 16 different Paris landmarks + general views in Oxford.

### Accuracy
|    Class    |        CLIP Zero-Shot        |        CLIP Linear Probe        |
|:--------------:|:--------------:|:--------------:|
| overall | 18.19% | 57.47% |
| All Souls | 0% | 50% |
| Ashmolean | 56% | 82% |
| Balliol | 29% | 38% |
| Bodleian | 75% | 48% |
| Christ Church | 7% | 54 % |
| Cornmarket | 52% | 20% |
| Hertford | 2% | 32% |
| Jesus | 4% | 35% |
| Keble | 44% | 45% |
| Magdalen | 10% | 60% |
| New | 24% | 48% |
| Oriel | 10% | 25% |
| general | 12% | 66 % |
| Pitt Rivers | 67% | 77% |
| Radcliffe Camera | 2% | 76% |
| Trinity | 0% | 47% |
| Worcester | 4% | 33% |

-----------------

# [[Wikiart Dataset]](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

- 52727 paintings from 195 different artists, each labeled with style, genre, and artist.

## Accuracy
|    Class    |        CLIP Zero-Shot        |        CLIP Linear Probe        |
|:--------------:|:--------------:|:--------------:|
| overall | 38.69% | 81.36% |
| Abstract Expressionism | 54% | 82% |
| Action painting | 3% | 20% |
| Analytical Cubism | 98% | 80% |
| Baroque | 85% | 76% |
| Color Field Painting | 27% | 79% |
| Fauvism | 80% | 65% |
| High Renaissance | 63% | 83% |
| Impressionism | 52% | 85% |
| Naive Art Primitivism | 39% | 82% |
| New Realism | 21% | 40% |
| Pop Art | 30% | 74% |
| Realism | 8% | 80% |
| Synthetic Cubism | 1% | 88% |

-----------------
# Conditional Generation on Quickdraw images

I attempted to train a VAE on the Quickdraw images of `DomainNet`, using the encoder and decoder architectures of [DALL-E](https://github.com/openai/DALL-E). Unfortunately, I kept running into a `RuntimeError` saying `CUDA out of memory`.
