# speech-separation-tf

A single-path time-domain monaural speech-separation model solves the cocktail party problem. The model was trained on the Libri2mix dataset, which also includes WHAM noise. Our model effectively suppresses the noise and directly decodes the mixture into sources. The model was trained with SI-SNR loss function and SI-SNRi is the performance metric.
