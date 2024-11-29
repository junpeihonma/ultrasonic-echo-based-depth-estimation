# Indoor Depth Estimation Using Ultrasonic Echoes

Junpei Honma, Akisato Kimura, Go Irie

<p align="center"><img width="600" alt="top_image" src="https://github.com/user-attachments/assets/555baa15-8845-4f37-a0ca-c78b9c324f94"></p>


[Paper](https://arxiv.org/pdf/2409.03336)


## Requirements
The code is tested with
``` 
- Python 3.6 
- PyTorch 1.6.0
- Numpy 1.19.5
```

## Dataset

**TUS-Echo** can be obatined from [here](https://github.com/junpeihonma/TUS-Echo). 

**Replica-VisualEchoes** can be obatined from [here](https://github.com/facebookresearch/VisualEchoes). We have used the 128x128 image resolution for our experiment. 


## Training

To train the model, first download the pre-trained material net from above link. 
```
python train.py \
--validation_on \
--dataset mp3d \
--img_path path_to_img_folder \
--metadatapath path_to_metadata \
--audio_path path_to_audio_folder \
--checkpoints_dir path_to_save_checkpoints \
--init_material_weight path_to_pre-trained_material_net
```
## Evaluation 

To evaluate the method using the pre-trained model, download the models for the corresponding dataset and the dataset.
- Evalution for Replica dataset
```
python test.py \
--img_path path_to_img_folder \
--audio_path path_to_audio_data \
--checkpoints_dir path_to_the_pretrained_model \
--dataset replica
```
- Evaluation for Matterport3D dataset
```
python test.py \
--img_path path_to_img_folder \
--audio_path path_to_audio_data \
--checkpoints_dir path_to_the_pretrained_model \
--dataset mp3d
```

## Acknowledgements
This work was partially supported by JSPS KAKENHI Grant Number 23K11154.

