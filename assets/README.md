# Figures

Image sources of the figures in the main README. Both figures draw the answers of SAM 3 and UNVEIL over the images.

## `teaser.webp`

One test image per prompt.

| Prompt | Test set | Image |
| --- | --- | --- |
| salient object | DUTS-TE | `sun_avjotqyblnswxoio` |
| camouflaged object | NC4K | `3451` |
| edges | BSDS500 test | `79073` |
| skeleton | SK-LARGE test | `12200fc1de7313f132a59c99a84cc9d2` |
| transparent object | Trans10K test | `1180` |
| mirror | PMD test | `000000133912` |
| shadow | ISTD test | `125-4` |

Two of these images come from COCO, under CC BY 2.0:

- skeleton: a crop of COCO image 421109, "[Paltinis](https://www.flickr.com/photos/bortescristian/6144513807/)" by
  Cristian Bortes.
- mirror: COCO image 133912
  ([Flickr original](http://farm3.staticflickr.com/2279/2350359108_484c82a130_z.jpg)).

The other images are under the terms of their test sets ([datasets/README.md](../datasets/README.md)).

## `one_image.png`

COCO image 80554, "[Wii Bowling](https://www.flickr.com/photos/stevewhite/2621941710/)" by echobase_2000,
CC BY-NC-SA 2.0.
