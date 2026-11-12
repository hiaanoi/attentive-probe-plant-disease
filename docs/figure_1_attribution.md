# Figure 1 photograph and attention output

Photograph from PlantDoc (CC BY 4.0), centre-cropped as model input.

Source: `test/Tomato Early blight leaf/Shoemaker_7068.JPG.jpg`, in the
[PlantDoc classification repository](https://github.com/pratikkayal/PlantDoc-Dataset),
commit `5467f6012d78d1c446145d5f582da6096f852ae8`; dataset publication Singh et al. (2020).
License: https://creativecommons.org/licenses/by/4.0/ . Attribution follows the
repository declaration; it is not an independent audit of each photograph's
original upstream copyright chain.

Changes: the model-input resize and 518-pixel centre crop; a diagram uses the
same photograph and a version dimmed according to five-seed mean softmax
attention. The predictions correspond to the saved official PlantDoc classifiers.
The attentive model corrects the within-host error in all five seeds
101, 202, 303, 404 and 505. The attention array is a saved visualisation output,
not trainable model parameters. No new model forward generated it for this release.

The supplied author's Figure 1 PPTX was exported to PDF. The SVG was derived
from that PDF. The standalone public image and editable PPTX stay outside this
repository; source code allows a reader to rebuild after acquiring the image.
