## Naive Bayesian Classification with TF-IDF support

Perform naive Bayesian classification into an arbitrary number of classes on sets of strings.

Copyright (c) 2011. Jake Brukhman. (jbrukh@gmail.com).
All rights reserved.  See the LICENSE file for BSD-style
license.

Forked from github.com/jbrukh/bayesian

Added TF-IDF (term frequency–inverse document frequency) capability.
Gain quite a bit of accurancy !


------------

### Background

See code comments for a refresher on naive Bayesian classifiers.

------------

### Installation

Using the go command:
```shell
go get github.com/jbrukh/bayesian
go install !$
```
------------

### Documentation

See the GoPkgDoc documentation [here](https://godoc.org/github.com/jbrukh/bayesian).

------------

### Features

- Conditional probability and "log-likelihood"-like scoring.
- Underflow detection.
- Simple persistence of classifiers.
- Statistics.

------------

### Example 1 (plain no tf-idf)


To use the classifier, first you must create some classes
and train it:
```go
import . "bayesian"

const (
    Good Class = "Good"
    Bad Class = "Bad"
)

classifier := NewClassifier(Good, Bad)
goodStuff := []string{"tall", "rich", "handsome"}
badStuff  := []string{"poor", "smelly", "ugly"}
classifier.Learn(goodStuff, Good)
classifier.Learn(badStuff,  Bad)
```
Then you can ascertain the scores of each class and
the most likely class your data belongs to:
```go
scores, likely, _ := classifier.LogScores(
                        []string{"tall", "girl"}
                     )
```
Magnitude of the score indicates likelihood. Alternatively (but
with some risk of float underflow), you can obtain actual probabilities:

```go
probs, likely, _ := classifier.ProbScores(
                        []string{"tall", "girl"}
                     )
```

### Example 2 (TF-IDF)
To use the TF-IDF classifier, first you must create some classes
and train it AND you need to call ConvertTermsFreqToTfIdf() AFTER training
and before Classifying methods(LogScore,ProbSafeScore,ProbScore)

```go
import . "bayesian"

const (
    Good Class = "Good"
    Bad Class = "Bad"
)

classifier := NewClassifierTfIdf(Good, Bad) // Extra constructor
goodStuff := []string{"tall", "rich", "handsome"}
badStuff  := []string{"poor", "smelly", "ugly"}
classifier.Learn(goodStuff, Good)
classifier.Learn(badStuff,  Bad)

classifier.ConvertTermsFreqToTfIdf() // IMPORTANT !!
```
Then you can ascertain the scores of each class and
the most likely class your data belongs to:
```go
scores, likely, _ := classifier.LogScores(
                        []string{"tall", "girl"}
                     )
```
Magnitude of the score indicates likelihood. Alternatively (but
with some risk of float underflow), you can obtain actual probabilities:

```go
probs, likely, _ := classifier.ProbScores(
                        []string{"tall", "girl"}
                     )
```
Use wisely.
