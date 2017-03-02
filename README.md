# Jingju Singing Phrase Matching
The code in this repo aims to help reproduce the results in the work:
>Matching singing phrase audio to score by combining phonetic and duration information

The objective of this research task to find the corresponding score for its singing query audio. By pre-segmenting both the singing audios and the music scores into the phrase units, we restrict this research to the "matching" scope. The matched scores could facilitate several lower-level MIR tasks, such as the score-informed automatic syllable or phoneme segmentation for singing voice.

The related code only situated in **phoneticSimilarity** folder. Other folders are used to test other matching methods.

## Steps to reproduce the experiment results
1. Clone this repository
2. Download Jingju a capella singing dataset from http://doi.org/10.5281/zenodo.344932
3. Change `dataset_path` variable in `general/filePath.py` to locate the above dataset
4. Compile cython viterbi decoding code: go into `CythonModule`, in terminal type
```
python setup.py build_ext --inplace
```
5. Install dependencies (see below)
6. Choose `class_name` in `general/filePath.py` to `'danAll'` or `'laosheng'` to experiment on either dan or laosheng role-type
7. Choose `am ` in `general/parameters.py` to `'gmm'` or `'cnn'` to select acoustic model
8. Run `python runHMM.py` to produce the experiment results for HMM and post-processor duration modelling matching
9. Run `python runHSMM.py` to produce the experiment results for HSMM duration modelling matching
10. The details instructions are written in these two files above

## Steps to train GMM, CNN acoustic models
1. Do steps 1, 2, 3 in **Steps to reproduce the experiment results**
2. To train GMM models, run `python acousticModelTraining.py`
3. To train CNN acoustic models, please follow the instructions in https://github.com/ronggong/EUSIPCO2017
4. After training CNN models, put them cnnModels folders. Pre-trained models are already included.

## Dependencies
`numpy scipy matplotlib essentia vamp scikit-learn pinyin cython keras theano unicodecsv`

## License
Affero GNU General Public License version 3
