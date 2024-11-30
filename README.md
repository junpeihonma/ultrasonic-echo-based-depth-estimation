# Multi-Task Learning for Ultrasonic Echo-Based Depth Estimation with Audible Frequency Recovery

Junpei Honma, Akisato Kimura, Go Irie

Echo-based depth estimation has been explored as a promising alternative solution.
We explore depth estimation based on ultrasonic echoes, which has scarcely been explored so far. 
The key idea of our method is to learn a depth estimation model that can exploit useful, but missing information in the audible frequency band.
we propose a multi-task learning approach that requires the model to estimate the depth map from the ultrasound echoes while simultaneously restoring the spectra in the audible frequency range.

[[Paper]](https://arxiv.org/pdf/2409.03336)

<p align="center"><img width="750" alt="top_image" src="https://github.com/user-attachments/assets/555baa15-8845-4f37-a0ca-c78b9c324f94"></p>

## Requirements
The code is tested with
``` 
- Python 3.8.10
- PyTorch 1.12.0
- Numpy 1.22.3
- librosa 0.9.2
- SoundFile 0.10.3
- torchaudio 0.12.0
```

## Dataset

**TUS-Echo** can be obatined from [here](https://github.com/junpeihonma/TUS-Echo). 

## Training

To train the model, first download the pre-trained material net from above link. 
```
python train.py \
--dataset TUS-Echo \
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

