

# DNN Assignment 2
- [DNN Assignment 2](#dnn-assignment-2)
  - [Student Details](#student-details)
  - [Configuration](#configuration)
    - [Problem 1](#problem-1)
    - [Problem 2](#problem-2)
  - [Environment](#environment)
  - [Execution Order](#execution-order)
    - [Run problem1.ipynb](#run-problem1ipynb)
  - [Results](#results)
  - [Runtime](#runtime)
  - [Assumptions](#assumptions)
  - [Known Limitations](#known-limitations)

## Student Details

Name: Nilesh Ranjan Mitra

Student ID: 2026aj05496

## Configuration

### Problem 1

Hidden size (h): 64

Seeds: 42, 123, 999

### Problem 2

Model dimension (d): 128

Heads: 4

Layers: 4

## Environment

Python:3.12.10
PyTorch:
NumPy:
Pandas:

## Execution Order

### Run problem1.ipynb
In order to perform the big 5 task of problem 1, I have decided to spilt out thing so that debugging is easy. Thus following **S**ingle **R**esponsibility and **S**eperation of **C**oncern, the problem1.ipynb files is the orchestrator or diff logics and the integrator of the results. The actual heavy lifting has been segeregated into respective modules and available in the "modules" folder. These modules are kept and loaded relative path ways so that the executor of problem1.ipynb file should not have any extra things to do before executing.
    
2. Run problem2.ipynb

## Results

Location of figures

## Runtime

**HAR:** The data source used for this is from [UCI-Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/humanactivityrecognitionusingsmartphones)

Adding Problem:

Transformer:

## Assumptions

...

## Known Limitations

...