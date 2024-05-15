# DLNLP_assignment_24-SN23077662

Deep Learning for Natural Language Processing (ELEC0141) 2024 Final Assignment

### Description
This project provides solutions to the Kaggle competition: "Sentiment Analysis on Movie Reviews" [[Kaggle Competition](https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews)].   
The project aims to classify text data using multiple models (Naive Bayes, LSTM, and BERT). It includes steps for data loading, preprocessing, model training, and performance evaluation.

The project employs three models:
1. **Naive Bayes**: Uses TF-IDF features for classification.
2. **LSTM**: A deep learning model for handling sequential data.
3. **BERT**: A powerful pre-trained language model for text classification tasks.
 
| Model              | Test Accuracy |
|--------------------|---------------|
| Naive Bayes        | 0.58          |
| LSTM               | 0.65          |
| Fine-tuning BERT   | 0.69          |

### Python Libraries Required

- numpy
- scikit-learn
- tensorflow
- keras
- pandas
- seaborn
- spacy
- transformers
- matplotlib

### Program Structure
-- AMLS_23-24_SN23077662
```
- Datasets
- Images
- Model
  - Bert_weights.h5 (trained bert model weights)
- .gitattributes
- main.py (file to run the whole project)
- src.py
- Bert_FineTuning.ipynb
- environment.yml
- README.md
```

- `Datasets/`: Contains the datasets, you can also download from the Kaggle website.
- `Images/`: Folder to save images of the training process and confusion matrix.
- `Model/`: Contains pre-trained BERT model weights.
- `.gitattributes`: Git attributes file for large files (Git LFS).
- `src.py`: Contains functions for the whole project used by `main.py`.
- `main.py`: The main script for executing the whole project.
- `Bert_FineTuning.ipynb`: Kaggle notebook showing the model training process for fine-tuning BERT.
- `environment.yml`: The Python environment file.
- `README.md`: The README file for the project.

### Program Run Instruction
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Make sure all required libraries are installed in the local environment. To create the environment and activate it, run the following commands:
    ```sh
    conda env create -f environment.yml
    conda activate NLP
    ```

3. Run `main.py` to create, train models, and evaluate them. Change the first parameter `train_bert` to `True` to begin training the BERT model instead of loading pre-trained weights:
    ```sh
    python main.py
    ```

4. Detailed training processes for BERT are shown in the Kaggle notebook (recommended!).

### Before Running
- The program defaults to loading the pre-trained BERT model rather than training from scratch, which may take more than 1 hour on a GPU (P100) provided by Kaggle.
- You can change the first parameter `train_bert` to `True` to begin training BERT instead of loading pre-trained models.
- The Kaggle competition does not provide test labels, so the training data are used and split with a ratio of 0.1 for validation.
- Training the models on different platforms and environments may lead to slight differences in results.
- For GPU requirements, the notebook is created and run on Kaggle based on the competition's Python environment pinned to 2021-12-01. Please ensure to use this environment to avoid conflicts. You could copy a baseline competition code notebook and copy the script into that notebook to use the same environment.

