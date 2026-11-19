# Example images

| File | Source | Licence |
| --- | --- | --- |
| `coco_000000228144.jpg` | COCO 2017 val, image 228144 ([Flickr original](http://farm4.staticflickr.com/3800/9678757658_774e298ba2_z.jpg)) | CC BY 2.0 |

The image is not part of the training data. Try it with

```bash
python scripts/predict.py assets/examples/coco_000000228144.jpg --prompts mirror edges "salient object" person --overlay --out outputs/example
```
