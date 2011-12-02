### Naive Bayesian Classification

Perform naive Bayesian classification into an arbitrary number of classes on sets of strings.

Copyright (c) 2011. Jake Brukhman. (jbrukh@gmail.com).
All rights reserved.  See the LICENSE file for BSD-style
license.

------------

#### Background

See code comments for a refresher on naive Bayesian classifiers.

------------

#### Installation

To install, simply:

    $ make install

To test, use:

    $ make test

------------

#### Example

To use the classifier, first you must create some classes
and train it:

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

Then you can ascertain the scores of each class and
the most likely class your data belongs to:

    scores, likely, _ := classifier.LogScores(
                            []string{"tall", "girl"}
                         )

Magnitude of the score indicates likelihood. Alternatively (but
with some risk of float underflow), you can obtain actual probabilities:


    probs, likely, _ := classifier.Probabilities(
                            []string{"tall", "girl"}
                         )

Use wisely.
