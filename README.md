# POS Tagging using Recurrent Neural Networks (RNNs)

This project demonstrates the use of Recurrent Neural Networks (RNNs), specifically LSTM, GRU, and Bidirectional LSTM, for Part-of-Speech (POS) tagging. POS tagging is a fundamental task in Natural Language Processing (NLP) that involves assigning grammatical tags (e.g., noun, verb, adjective) to words in a sentence.

## Project Overview

The project is implemented in Python using the following libraries:

* **NLTK:** For accessing POS-tagged corpora.
* **Gensim:** For loading pre-trained word embeddings (Word2Vec).
* **TensorFlow/Keras:** For building and training the RNN models.
* **Scikit-learn:** For data splitting and shuffling.
* **Seaborn and Matplotlib:** For data visualization.


## Data

The project utilizes the following POS-tagged corpora from NLTK:

* **Treebank**
* **Brown**
* **Conll2000**
* **masc_tagged**

These corpora are combined to create a comprehensive dataset for training and evaluation.


## Methodology

The project follows these steps:

1. **Data Preprocessing:**
* Load and combine the POS-tagged corpora.
* Extract input sequences (words) and output sequences (POS tags).
* Encode words and tags using tokenization.
* Pad sequences to a uniform length using Keras' `pad_sequences`.
* Utilize pre-trained Word2Vec embeddings for word representation.
* One-hot encode the output sequences (POS tags).
* Split data into training, validation, and testing sets.


2. **Model Building:**
* Experiment with different RNN architectures:
    * **Vanilla RNN**
    * **LSTM**
    * **GRU**
    * **Bidirectional LSTM**
* Incorporate pre-trained Word2Vec embeddings.
* Train the models using the training data.
* Validate the models using the validation set.


3. **Evaluation:**
* Evaluate the trained models on the testing set.
* Compare the performance of different architectures using accuracy as the metric.

## Usage

**Training the models:**

* Run the code provided in the Colab notebook. This will train all four RNN models.

**Evaluating the models:**
* Run the evaluation code snippets to check the model's performance on test data.
* Adjust the hyperparameters in the notebook to experiment with other models.

## Results

The notebook displays training and validation accuracy for each model.
You can analyze these results to find the best-performing model.

## Conclusion

This project provides a practical example of using RNNs for POS tagging. By experimenting with different architectures and embeddings, you can explore various options for improving model performance.
