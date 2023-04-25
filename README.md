# TransformerVAE
The official repository for ICASSP 2020 Transformer VAE paper Transformer VAE: A Hierarchical Model for Structure-aware and Interpretable Music Representation Learning.

## Demo

See here for demo: https://drive.google.com/drive/folders/1Su-8qrK__28mAesSCJdjo6QZf9zEgIx6

## Model inference & Pre-trained weights

You can also generate the demo by yourself using pre-trained models

1. Download pre-trained model from https://drive.google.com/drive/folders/17H32cQC2SPpajIvUqXaLIWIhx_QKGO_H and put the model file at ``cache_data\transformer_sequential_vae_no_chord_v2.1_m111_3_layer_kl1.000000_s0.sdict``.
2. Install the dependencies for the repo by ``pip3 install -r requirements.txt`` (preferably in a virtual environment)
3. Run ``python3 transformer_sequential_vae_interp.py``, and the program should generate a new folder ``output`` with audio samples in it.

Each generated piece in the folder ``output/transformer_sequential/{model_name}/swap_first/{song_1_name}-{song_2_name}.mid`` has the following format:

* 0:00 - 0:12 Original song 1 (8 bars)
* 0:12 - 0:24 Original song 2 (8 bars)
* 0:24 - 0:36 Break
* 0:36 - 0:48 Reconstructed song 1 (8 bars)
* 0:48 - 1:00 Reconstructed song 2 (8 bars)
* 1:00 - 1:12 Break
* 1:12 - 1:24 Generated new song 1 (with latent code from song 2 first, then song 1)
* 1:24 - 1:36 Generated new song 2 (with latent code from song 1 first, then song 2)

## Retrain the model

1. You need to acquire the dataset file ``hooktheory_gen_update_4`` to reproduce. Currently, you need to contact the first author to get access to the dataset.
2. Change the code of path of ``FramedRAMDataStorage('E:/dataset/hooktheory_gen_update_4')`` in ``transformer_sequential_vae.py`` to the path that you put your dataset file in.
3. Run ``python3 transformer_sequential_vae.py 0`` to train the model and use data fold 0 as the validation fold.
