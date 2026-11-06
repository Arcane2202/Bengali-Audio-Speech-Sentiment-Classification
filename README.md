# Bengali Speech Sentiment Analysis using Machine Learning Models: A Comparative Study

Code and processed data for our paper published at the 2024 IEEE International Conference on Artificial Intelligence in Engineering and Technology (IICAIET), pages 254 to 259.

Paper: https://doi.org/10.1109/IICAIET62352.2024.10730340

## About

Most sentiment analysis work on Bangla has been done on text, even though Bangla is the fifth most spoken language in the world. In this project we classify emotion directly from Bengali speech audio. We merged three public Bangla speech emotion datasets and compared three kinds of models on the same data:

1. Classical ML models (KNN, Random Forest, Gradient Boosting, AdaBoost, Decision Tree) trained on numerical audio features
2. Sequence models (LSTM and Bi-LSTM) trained on the same numerical features
3. ImageNet pretrained CNNs (ResNet50, ResNet101, ResNet152, VGG16, VGG19, DenseNet201, GoogleNet) fine tuned on log Mel spectrogram images

We also combined the models in each group using voting ensembles.

The seven classes are Angry, Disgust, Fear, Happy, Neutral, Sad and Surprise.

## Main results

| Approach | Model | Accuracy |
|---|---|---|
| Single CNN on spectrograms | DenseNet201 | 94.00% |
| Single ML model on all numerical features | Gradient Boosting | 90.84% |
| Single sequence model on all numerical features | Bi-LSTM | 80.29% |
| Ensemble of all 7 CNNs (hard and soft voting) | | 95.00% |
| Ensemble of ML and sequence models (hard voting) | | 94.99% |

CNNs on spectrograms gave the best single model results. Ensembling pushed both the spectrogram side and the numerical feature side to around 95%.

## Dataset

We used SUBESCO, BanglaSER and KBES. Classes were matched across the three datasets and merged. KBES has low and high intensity versions of each emotion (except Neutral), and we merged those into one class per emotion.

| Sentiment | SUBESCO | BanglaSER | KBES | Merged |
|---|---:|---:|---:|---:|
| Happy | 1000 | 306 | 200 | 1506 |
| Surprise | 1000 | 306 | 0 | 1306 |
| Angry | 1000 | 306 | 200 | 1506 |
| Sad | 1000 | 306 | 200 | 1506 |
| Fear | 1000 | 0 | 0 | 1000 |
| Disgust | 1000 | 0 | 200 | 1200 |
| Neutral | 1000 | 243 | 100 | 1343 |
| Total | 7000 | 1467 | 900 | 9367 |

Dataset sources:

* SUBESCO: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0250173
* BanglaSER: https://www.sciencedirect.com/science/article/pii/S235234092200302X
* KBES: https://www.sciencedirect.com/science/article/pii/S2352340923008107

We set aside 10% of the merged data as the test set first. Only the remaining training data was augmented, which doubled its size. These are the counts in this repo:

| Sentiment | Train (augmented) | Test |
|---|---:|---:|
| Angry | 2710 | 151 |
| Disgust | 2160 | 120 |
| Fear | 1800 | 100 |
| Happy | 2710 | 151 |
| Neutral | 2416 | 135 |
| Sad | 2710 | 151 |
| Surprise | 2350 | 131 |
| Total | 16856 | 939 |

Note: Table II in the paper lists 2466 train and 110 test samples for Neutral. The numbers above are what is actually in the repo.

The raw audio files are not included here. If you need them, please download them from the original dataset authors. This repo only has the extracted features and the spectrogram images.

## Method

### Pipeline

1. Merge SUBESCO, BanglaSER and KBES into one dataset
2. Split off 10% as the test set
3. Augment the training audio
4. Extract numerical features into CSV files and generate log Mel spectrograms as PNG images
5. Train ML and sequence models on the CSV features, and CNNs on the spectrograms
6. Combine the models with voting ensembles

### Augmentation

All three datasets were recorded in quiet, controlled settings, which is not very close to real life. So we augmented each training clip with Librosa using:

* Pitch shifting with a factor of about 0.7, plus a random variation between -0.1 and 0.1
* Time stretching with a factor of 0.8, plus a random variation between -0.1 and 0.1
* Gaussian noise at 0 to 0.5% of the clip's maximum amplitude

### Features

Numerical features extracted with Librosa (in `Dataset/Feature CSV/`):

* 40 MFCCs (`mfcc0` to `mfcc39`)
* Zero Crossing Rate (`zero_crossing_rate`)
* Chroma (`chroma`)
* Root Mean Square energy (`rms`)

`spectral_bandwidth` is also in the CSVs but we did not use it in any experiment.

Log Mel spectrograms were generated with 128 Mel bins, a Hann window of 2048 samples and a hop length of 512. They are saved as 1000x400 PNG files in `Dataset/Train_Spects/` and `Dataset/Test_Spects/`.

### Class imbalance and scaling

Fear only exists in SUBESCO, so the classes are not balanced. We used SMOTE to oversample the smaller classes in the numerical training data. StandardScaler is applied to the features before they go into the sequence models.

### ML models

Decision Tree, Random Forest, AdaBoost, Gradient Boosting and KNN (k = 5) from scikit-learn. Each one was trained on six feature sets: MFCC only, ZCR only, RMS only, Chroma only, ZCR + RMS + Chroma, and all features together.

### Sequence models

Built with Keras. Each feature vector is reshaped into a single time step.

* LSTM: three stacked LSTM layers with 100, 50 and 20 units, dropout between them, softmax output
* Bi-LSTM: two bidirectional LSTM layers with 100 and 50 units, dropout, softmax output

Both use the Adam optimizer and categorical cross entropy, trained for 120 epochs with batch size 128 and a 20% validation split.

### CNNs

Built with PyTorch and torchvision, starting from ImageNet weights. We replaced the original classifier of each network with five residual blocks followed by a linear output layer for the 7 classes. Each block is Linear, BatchNorm1d, ReLU and Dropout with a skip connection. The block widths are 512, 256, 128, 64 and 32, and the dropout goes from 0.1 up to 0.5. For the VGG models we added adaptive average pooling (7x7) and a flatten layer before the blocks.

Training details:

* Images are resized so the shorter side is 224 px and normalized with the ImageNet mean and std
* The training spectrograms are split 80/20 into train and validation
* SGD with learning rate 0.001 and momentum 0.9, StepLR with step size 7 and gamma 0.1, cross entropy loss
* 10 to 12 epochs depending on the model
* The checkpoint with the best validation accuracy is saved and later tested on `Test_Spects`

### Ensembles

For the ML and sequence models we used hard voting (majority vote on predicted labels) and soft voting (average of predicted probabilities) over DT, RF, AdaBoost, GB, KNN, LSTM and Bi-LSTM.

For the CNNs, `DL Models/Ensembled_Custom.ipynb` loads all seven trained models, adds up their outputs and picks the class with the highest score.

## Repository structure

```
Dataset/
    Feature CSV/
        Train_Data.csv      16,856 rows, augmented training features
        Test_Data.csv       939 rows, test features
    Train_Spects/<Class>/   16,856 spectrogram images
    Test_Spects/<Class>/    939 spectrogram images

ML Models/                  ML models, LSTM, Bi-LSTM and voting ensembles
    Final_Ml_Models_MFCC.ipynb
    Final_Ml_Models_ZCR.ipynb
    Final_Ml_Models_RMS.ipynb
    Final_Ml_Models_CR.ipynb             (Chroma)
    Final_Ml_Models_CR_ZCR_RMS.ipynb     (Chroma + ZCR + RMS)
    Final_Ml_Models_ALL_Features.ipynb

DL Models/                  CNN training on spectrograms, one notebook per model
    DenseNet201_Custom.ipynb
    GoogleNet_Custom.ipynb
    ResNet50_Custom.ipynb
    ResNet101_Custom.ipynb
    ResNet152_Custom.ipynb
    VGG16_Custom.ipynb
    VGG19_Custom.ipynb
    Ensembled_Custom.ipynb               (ensemble of all 7 CNNs)
```

The CSV files have these columns: `mfccs` (the full MFCC array stored as a string), `zero_crossing_rate`, `spectral_bandwidth`, `chroma`, `rms`, `class_label`, `file_name`, and `mfcc0` to `mfcc39`. File names follow the naming used by each original dataset.

## How to run

Clone the repo (it is fairly large because of the ~17.8k images):

```bash
git clone https://github.com/Arcane2202/CSE4238_Group5_Project.git
cd CSE4238_Group5_Project
```

Install the packages. We used Python 3.9 or newer.

```bash
# for the ML and sequence model notebooks
pip install numpy pandas scikit-learn imbalanced-learn tensorflow seaborn matplotlib joblib

# for the CNN notebooks
pip install torch torchvision opencv-python pillow tqdm
```

A GPU with CUDA is highly recommended for the CNN notebooks.

### Fix the paths first

The notebooks still have the file paths from our own machines, so you will need to change them:

| Notebooks | Path in the notebook | Change to |
|---|---|---|
| `ML Models/*` (made in Google Colab) | `/content/drive/MyDrive/Softcom/Train_Data.csv` and `Test_Data.csv` | `Dataset/Feature CSV/Train_Data.csv` and `Dataset/Feature CSV/Test_Data.csv` |
| `DL Models/*` | `D:/Program Files (x86)/Softcom/Train_Spects/*` and `Test_Spects/*` | `Dataset/Train_Spects/*` and `Dataset/Test_Spects/*` |
| `DL Models/*` | `D:/Program Files (x86)/Softcom/<model>.pth` | wherever you want to save the model weights |

If you are not using Colab, also remove the `drive.mount(...)` cell from the ML notebooks.

### Running the notebooks

* ML and sequence models: open any notebook in `ML Models/` and run all cells. It trains the five ML models plus LSTM and Bi-LSTM, prints accuracy, precision, recall and F1 with confusion matrices, and then runs the hard and soft voting ensembles.
* CNNs: run each model notebook in `DL Models/` to train it and save its weights. After all seven are trained, run `Ensembled_Custom.ipynb` to test the ensemble.

Trained weights are not included in the repo, so you need to train the CNNs before running the ensemble notebook. The scripts we used for augmentation, feature extraction and spectrogram generation are also not included, but their outputs are in `Dataset/`.

## Detailed results

All results are on the test set. Precision, recall and F1 are weighted averages. These numbers come from the paper.

### CNNs (log Mel spectrograms)

| Model | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| ResNet50 | 90% | 90% | 89% | 90% |
| ResNet101 | 90% | 90% | 90% | 90% |
| ResNet152 | 92% | 92% | 92% | 92% |
| VGG16 | 90% | 90% | 90% | 90% |
| VGG19 | 91% | 91% | 91% | 91% |
| DenseNet201 | 94% | 94% | 94% | 94% |
| GoogleNet | 86% | 86% | 86% | 86% |

### ML and sequence models (accuracy per feature set)

| Features | KNN | Random Forest | Gradient Boost | AdaBoost | Decision Tree | Bi-LSTM | LSTM |
|---|---:|---:|---:|---:|---:|---:|---:|
| MFCC | 55.27% | 80.83% | 73.69% | 24.49% | 73.16% | 77.95% | 79.76% |
| ZCR | 14.32% | 14.44% | 16.71% | 13.41% | 16.08% | 17.57% | 17.25% |
| RMS | 18.21% | 17.57% | 20.44% | 17.41% | 17.15% | 20.23% | 21.19% |
| Chroma | 14.90% | 14.90% | 20.76% | 15.01% | 15.76% | 20.55% | 21.61% |
| ZCR + RMS + Chroma | 28.22% | 33.65% | 32.58% | 17.25% | 35.67% | 31.73% | 33.01% |
| All features | 71.13% | 87.00% | 90.84% | 71.14% | 72.94% | 80.29% | 79.44% |

Using all features together worked best. MFCC was the most useful single feature by a big margin. ZCR, RMS and Chroma on their own performed close to random guessing (around 14% for 7 classes). Precision, recall and F1 for every model are in Table III of the paper.

### Ensembles

| Models | Features | Voting | Accuracy | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| ML + Sequence | All features | Hard | 94.99% | 95.52% | 94.99% | 94.93% |
| ML + Sequence | All features | Soft | 91.58% | 91.84% | 91.58% | 91.36% |
| ML + Sequence | MFCC | Hard | 83.91% | 84.28% | 83.91% | 83.71% |
| ML + Sequence | MFCC | Soft | 82.42% | 82.95% | 82.42% | 81.69% |
| CNNs | Log Mel spectrogram | Hard | 95.00% | 95.00% | 95.00% | 95.00% |
| CNNs | Log Mel spectrogram | Soft | 95.00% | 95.00% | 95.00% | 95.00% |

## Limitations and future work

All three datasets were recorded in controlled environments. Our augmentation adds some noise and variation, but we did not test on real world recordings with background noise, different dialects or different speaking styles. Some things we would like to try next:

* Test more CNN architectures, including our own designs
* Build models that hold up on noisy, real world Bangla speech
* Try other audio features to see which ones suit this task best

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{shams2024bengali,
  title     = {Bengali Speech Sentiment Analysis using Machine Learning Models: A Comparative Study},
  author    = {Shams, Mohammad Tanveer and Hasan, Md. Akib and Chowdhury, Animesh Das and
               Lamia, Tabassum Jahan and Chowdhury, Md. Reasad Zaman and Rahman, Mohammad Marufur},
  booktitle = {2024 IEEE International Conference on Artificial Intelligence in Engineering and Technology (IICAIET)},
  pages     = {254--259},
  year      = {2024},
  publisher = {IEEE},
  doi       = {10.1109/IICAIET62352.2024.10730340}
}
```

Please also cite SUBESCO, BanglaSER and KBES if you use the data.

## Authors

Department of CSE, Ahsanullah University of Science and Technology (AUST), Dhaka, Bangladesh

* [Mohammad Tanveer Shams](https://github.com/Arcane2202)
* [Md. Akib Hasan](https://github.com/MDAkibHasan)
* [Animesh Das Chowdhury](https://github.com/T1RTHO)
* [Tabassum Jahan Lamia](https://github.com/TabassumJahanLamia)

Supervisor:

* Md. Reasad Zaman Chowdhury (corresponding author)

## Acknowledgments

Thanks to our supervisors, Md. Reasad Zaman Chowdhury, for his guidance throughout this project. We are also grateful to the creators of SUBESCO, BanglaSER and KBES for making their datasets public.
