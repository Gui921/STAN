# Spatial Temporal Attention Network for the HARIA Project
![Spatial Attention Map](/images/spatial_attention.png)

## Architecture:
To fine-tune the R2plus1D model, we've applied spatial and temporal attention before feeding the processed images to the model. This way, we can "teach" the model to look on the most significant parts of the image.

In the end, we return the logits that make the action decision, the spatial attention map and the temporal attention map

![STAN Model Architecture](/images/STAN_architecture.png)

## How to run:

This framework has the following arguments:

`--train_mode`: Set the model to training mode. It will train using the provided dataset'

> Default : False
>
> Type : bool


`--train_name`: Name given to the checkpoint folder in training

> Type: str

`--checkpoint`: Model checkpoint to do the inference.

> Type: str

`--input_video`: Input video for the inference

> Type: str

### Examples:
For training:

`python main.py --train_mode True --train_name training_test`

For inference:

`python main.py --checkpoint chekpoint.pt --input_video video_test.mp4`

## Output Example:

### Spatial Attention:

![Spatial Attention Map](/images/spatial_attention.png)

### Temporal Attention:

![Temporal Attention Map](/images/temporal_attention.png)

## Contact:
Guilherme Ribeiro - fc53699@alunos.fc.ul.pt - [LinkedIn](https://www.linkedin.com/in/guilherme-ribeiro0328/) - [GitHub](https://github.com/Gui921)
