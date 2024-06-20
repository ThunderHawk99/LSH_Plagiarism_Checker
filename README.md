# Scalable-Analytics

This repository is for the group work of the course Scalable Analytics.

- [Scalable-Analytics](#scalable-analytics)
  - [Members](#members)
  - [Topic](#topic)
      - [Research Questions](#research-questions)
      - [Hypothesis](#hypothesis)
  - [Initializing the project](#initializing-the-project)
  - [Project Structure](#project-structure)
    - [Data Collection](#data-collection)
    - [Pre-processing](#pre-processing)
    - [LSH](#lsh)
    - [Evaluation](#evaluation)
    - [Scalability](#scalability)
    - [Conclusion](#conclusion)

## Members
We are all Master Computer Science students specializing in Data Management & Analytics. 
1. Faroukh Davouzov (2nd year Master)
2. Youri Langhendries (1st year Master)
3. Dana Tabatabaie Irani (2nd year Master)

## Topic

How effective is the use of Locality Sensitive Hashing (LSH) for detecting near-duplicates (plagiarism) in a large dataset of e.g. code and text documents?

#### Research Questions
- RQ1: How effective is LSH in detecting plagiarism within a large dataset of text documents?
- RQ2: Which shingling approach is most suitable for detecting plagiarism?
- RQ3: Which benefits does preprocessing offer when using LSH for plagiarism detection?
- RQ4: How does the performance of an LSH tool change when only a subset of the keywords in sentences is considered?

#### Hypothesis
- H1 : LSH is highly effective in detecting plagiarism within large datasets of text documents
due to its efficient similarity searches.
- H2 : We hypothesise that character-based shingling approaches are more suitable due to their ability to capture general patterns.
- H3: We hypothesise that preprocessed documents significantly improve the accuracy and efficiency of plagiarism detection using LSH by reducing noise and document size.
- H4: We anticipate that focusing on a subset of keywords within sentences will lead to a trade-off between performance and efficiency in LSH tools. However, we hypothesise that this trade-off will be justified by the gains in computational efficiency, making it a worthwhile approach.



## Initializing the project
**Installing `pipenv`**

To run the project, first run the following command to install `pipenv`:
```shell
pip install pipenv
```
**Initializing the virtual environment**

After this you can use the pipenv commands to initialize the virtual environment.
Run the following command to install all the dependencies from the `Pipfile`.
```shell
pipenv install
```
**Entering the virtual environment**

Finally, you can either change your interpreter to the interpreter in the virtual environment by pressing `Ctrl + Shift + P` and selecting the interpreter, or you can run the following command to enter the virtual environment:
```shell
pipenv shell
```
**Exiting the virtual environment**

When using pipenv shell, use the following command to exit the environment:
```shell
exit
```

## Project Structure

1. Data Collection
2. Pre-processing
3. LSH (our)
   1. Shingling
   2. Min-Hashing
   3. LSH
   4. Evaluation
4. Evaluation with other Plagiarism Detection Methods (e.g. [LSH Example](https://github.com/rushyaP/Locality-Sensitive-Hashing-Plagiarism-Detection))
5. Visualisation
6. Scalability
7. Conclusion

Extra:

- Make project scalable ([MapReduce](https://stackoverflow.com/questions/29320943/how-to-implement-lsh-by-mapreduce)).

### Data Collection

- Java code: https://github.com/oscarkarnalim/sourcecodeplagiarismdataset
- Wikipedia pages: https://deepblue.lib.umich.edu/data/concern/data_sets/2801pg45f?locale=en
- Text documents: https://github.com/josecruzado21/plagiarism_detection
- PAN Plagiarism Corpus 2011 (PAN-PC-11): https://zenodo.org/records/3250095

### Pre-processing

1. **Read** in file
2. Depending on **file type**

   1. Text: [Remove stopwords](https://gist.github.com/sebleier/554280), remove punctuation, lowercase, lemmatise
   2. Code: Remove comments, remove whitespace, change variable names to their token types (function, class, ...)

3. Extract **keywords** from files to create a bag of words and reduce the dimensionality of the data
4. Explode file-keywords into multiple file-keyword by **generating paraphrased versions** of these by generating synonyms of words.

> Notes:
>
> 1. Results from (2, 3) need to be saved to prevent re-computation.
> 2. The files in our possession remain unchanged. So we will not apply step 4 to these files. An argument for this is 1. for performance and 2. seems as duplicate work since we are trying to match the Read file with one of these.

Future work
- Look into another algorithm for code files

### LSH

LSH will be performed twice:

1. Once on the keywords extracted from the files.
2. If candidate pairs are found based on the keywords, LSH will be performed on the complete file, and its variants, to verify plagiarism with a higher degree of confidence.
> Notes:
> Files with the same keywords could be seen as plagiarism, even though their context might be completely different. This is why we use the first iteration as a filtering pass with increased performance, as the number of keywords in a file scales slower than the size of a document. This aims to increase the scalability of the model.
> The goal of altering the existing data is to extend our search range for broader plagiarism such as paraphrasing or translations.

### Evaluation
Perform exhaustive evaluation to find optimal combination of hyperparameters. 
### Scalability

- **MapReduce** implementation
  - This can be done by using the [Hadoop](https://hadoop.apache.org/) framework.
  - LSH can be implemented in a MapReduce fashion by using the following steps (specific steps need to be worked out):
    1. **Map**: The input data is split into chunks and each chunk is processed by a different mapper.
    2. **Shuffle**: The output of the mappers is shuffled and sorted by the keys.
    3. **Reduce**: The output of the shuffle phase is processed by the reducers to generate the final output.
- **Multi-Threaded** implementation:
  - The preprocessing is done in parallell with each thread working on a different file.
  - Each LSH run is also done in parallell.

### Conclusion

- **Results** of the LSH model on the different datasets
- **Comparison** with other Plagiarism Detection Methods
- **Comparsion** of the LSH model on various data types (text vs code)
  - Can tweak hyperparameters to be optimised in each case and which one performs best.
