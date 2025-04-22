# **Detailed Analysis of Multi-Task NLP Project**

## **Project Overview**

This project implements and evaluates three fundamental Natural Language Processing (NLP) techniques: Named Entity Recognition and Classification (NERC), Sentiment Analysis, and Topic Analysis. The implementation uses both rule-based and machine learning approaches, with detailed performance evaluation and error analysis.

## **Datasets**

The project utilizes three datasets:

1. **my_tweets.json**: Contains 50 tweets with sentiment annotations (positive, negative, neutral) used for training sentiment analysis models
2. **sentiment-topic-test.tsv**: Contains 18 sentences annotated with sentiment labels (positive, negative, neutral) and topic categories (sports, book, movie)  
3. **NER-test.tsv**: Contains tokenized sentences with BIO tagging scheme for named entities (PERSON, ORG, LOC, WORK\_OF\_ART)

## **NLP Techniques Implementation**

### **1\. Sentiment Analysis**

Two different approaches are compared:

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**

* A lexicon and rule-based sentiment analyzer specifically designed for social media text  
* Uses a dictionary of words with associated sentiment scores and applies grammatical and syntactical rules  
* Converts compound sentiment scores to categorical labels using threshold values (±0.05)  
* Achieves reasonable performance without training data
* **Results**: Achieved 55.6% accuracy on the test set with balanced precision and recall across sentiment classes
* **Enhancement**: Implemented custom lexicon additions for domain-specific terms to improve performance

**Machine Learning Approach (SVM)**

* Implements a pipeline with TF-IDF vectorization and Linear SVM classifier  
* Trains on augmented tweet data (54 samples after augmentation) using a combination of original and synonyms
* Uses n-gram features (unigrams, bigrams, trigrams) to capture phrase-level information
* Uses balanced class weights to handle class imbalance
* **Results**: Achieved 22.2% accuracy on the test set, significantly underperforming compared to VADER
* **Challenge**: Limited training data and domain mismatch between training tweets and test sentences

### **2\. Named Entity Recognition and Classification (NERC)**

* Utilizes spaCy's pre-trained English model (en\_core\_web\_sm)  
* Maps spaCy's entity types to the project's tagging scheme  
* Processes sentences token by token to identify entities  
* Evaluates performance using standard classification metrics  
* Handles entity boundaries through BIO (Beginning-Inside-Outside) tagging
* **Results**: Achieved 81% overall accuracy, with strong performance on LOC and PERSON entities
* **Custom Enhancement**: Implemented a custom entity recognition component to improve WORK_OF_ART detection
* **Challenges**: Poor recall for ORG entities (0%) and moderate recall for WORK_OF_ART entities (22%)

### **3\. Topic Analysis**

* Implements a machine learning approach using TF-IDF features and SVM  
* Classifies sentences into three topics: sports, books, and movies  
* Uses n-gram features (unigrams and bigrams) to capture phrase-level information  
* Evaluates classification performance with detailed metrics
* **Results**: Achieved 38.9% cross-validation accuracy on the limited dataset
* **Feature Analysis**: Identified distinctive n-grams for each topic category
* **Challenge**: Limited amount of training data (only 18 samples across 3 categories)

## **Comparative Analysis and Error Analysis**

The project performs detailed comparison between VADER and ML-based sentiment analysis:

* 38.9% of samples were misclassified by both methods
* 38.9% of samples were correctly classified by VADER but misclassified by ML
* 5.6% of samples were correctly classified by ML but misclassified by VADER
* Key error patterns:
  * Neutral sentences were particularly challenging for both classifiers
  * Complex sentences with mixed sentiment signals caused confusion
  * Context-dependent sentiment expressions were difficult to capture

## **Technical Implementation**

* Implemented in Python using standard data science and NLP libraries  
* Uses pandas for data manipulation  
* Employs scikit-learn for machine learning components  
* Utilizes spaCy for named entity recognition  
* Implements NLTK's VADER for lexicon-based sentiment analysis
* Implements data augmentation techniques to increase training data size
* Provides detailed error analysis and comparative assessment

## **Improvement Suggestions**

Based on the observed results, several enhancements could improve performance:

1. **Sentiment Analysis**:
   * Expand VADER lexicon with more domain-specific terms
   * Collect larger, more diverse training data for ML models
   * Implement pre-trained transformer models like BERT for better context understanding
   * Develop hybrid approaches combining rule-based and ML methods

2. **Named Entity Recognition**:
   * Fine-tune spaCy model on domain-specific data
   * Implement more sophisticated custom entity detection rules
   * Try alternative NER models with better performance on literary and creative works

3. **Topic Analysis**:
   * Increase the size of the topic-annotated dataset
   * Implement semi-supervised techniques to leverage unlabeled data
   * Use domain adaptation techniques to improve cross-domain performance

## **Potential Applications**

This multi-task NLP project demonstrates techniques that could be applied in:

* Social media monitoring  
* Customer feedback analysis  
* Content categorization  
* Information extraction from unstructured text  
* News article classification and entity extraction

The project showcases both traditional rule-based approaches and modern machine learning methods, providing a comprehensive view of different NLP paradigms and their relative strengths and weaknesses.
