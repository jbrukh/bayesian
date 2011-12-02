// A Naive Bayesian Classifier
// Jake Brukhman <jbrukh@gmail.com>

// 
// BAYESIAN CLASSIFICATION REFRESHER: suppose you have a set
// of classes (e.g. categories) C := {C_1, ..., C_n}, and a
// document D consisting of words D := {W_1, ..., W_k}.
// We wish to ascertain the probability that the document
// belongs to some class C_j given some set of training data
// associating documents and classes.
//
// By Bayes Theorem, we have that
//
//    P(C_j|D) = P(D|C_j)*P(C_j)/P(D).
//
// The LHS is the probability that the document belongs to class
// C_j given the document itself (by which is meant, in practice,
// the word frequencies occurring in this document), and our program
// will calculate this probability for each j and spit out the
// most likely class for this document.
//
// P(C_j) is referred to as the "prior" probability, or the
// probability that a document belongs to C_j in general, without
// seeing the document first. P(D|C_j) is the probability of seeing
// such a document, given that it belongs to C_j. Here, by assuming
// that words appear independently in documents (this being the 
// "naive" assumption), we can estimate
//
//    P(D|C_j) ~= P(W_1|C_j)*...*P(W_k|C_j)
//
// where P(W_i|C_j) is the probability of seeing the given word
// in a document of the given class. Finally, P(D) can be seen as 
// merely a scaling factor and is not strictly relevant to
// classificiation, unless you want to normalize the resulting
// scores and actually see probabilities. In this case, note that
//
//    P(D) = SUM_j(P(D|C_j)*P(C_j))
//
// One practical issue with performing these calculations is the
// possibility of float64 underflow when calculating P(D|C_j), as
// individual word probabilities can be arbitrarily small, and
// a document can have an arbitrarily large number of them. A
// typical method for dealing with this case is to transform the
// probability to the log domain and perform additions instead
// of multiplications:
//
//   log P(C_j|D) ~ log(P(C_j)) + SUM_i(log P(W_i|C_j))
//
// where i = 1, ..., k. Note that by doing this, we are discarding
// the scaling factor P(D) and our scores are no longer
// probabilities.
//
package bayesian

import (
    "math"
    "gob"
    "os"
)

// defaultProb is the tiny non-zero probability that a word
// we have not seen before appears in the class. 
const defaultProb = 0.00000000001

// This type defines a set of classes that the classifier will
// filter: C = {C_1, ..., C_n}. You should define your classes
// as a set of constants, for example as follows:
//
//    const (
//        Good Class = "Good"
//        Bad Class = "Bad
//    )
//
type Class string

// Classifier implements the Naive Bayesian Classifier.
type Classifier struct {
    Classes []Class
    datas map[Class]*classData
}

// serializableClassifier represents an container for
// classifier objects whose fields are modifiable by
// reflection and are therefore writeable.
type serializableClassifier struct {
    Classes []Class
    Datas map[Class]*classData
}

// classData holds the frequency data for words in a
// particular class. In the future, we may replace this
// structure with a trie-like structure for more
// efficient storage.
type classData struct {
    Freqs map[string]int
    Total int
}

// newClassData creates a new empty class data node.
func newClassData() *classData {
    return &classData{
        Freqs: make(map[string]int),
    }
}

// P(W|Cj) -- the probability of seeing a particular word
// in a document of this class.
func (d *classData) getWordProb(word string) float64 {
    value, ok := d.Freqs[word]
    if !ok {
        return defaultProb
    }
    return float64(value)/float64(d.Total)
}

// P(D|C_j) -- the probability of seeing this set of words
// in a document of this class.
//
// Note that words should not be empty, and this method of
// calulation is prone to underflow if there are many words
// and their individual probabilties are small.
func (d *classData) getWordsProb(words []string) (prob float64) {
    prob = 1
    for _, word := range words {
        prob *= d.getWordProb(word)
    }
    return
}

// New creates a new Classifier. The classes the provided
// should be at least 2 in number and unique from each other.
func NewClassifier(classes ...Class) (inst *Classifier) {
    if len(classes) < 2 {
        panic("provide at least two classes")
    }
    inst = &Classifier{
            classes,
            make(map[Class]*classData),
    }
    for _, class := range classes {
        inst.datas[class] = newClassData()
    }
    return
}

// NewClassifierFromFile loads an existing classifier from
// disk. The classifier was previously saved with a call
// to WriteToFile().
func NewClassifierFromFile(name string) (c *Classifier, err os.Error) {
    file, err := os.Open(name)
    if err != nil {
        return nil, err
    }
    dec := gob.NewDecoder(file)
    w := new(serializableClassifier)
    err = dec.Decode(w)

    return &Classifier{w.Classes, w.Datas}, err
}

// getPriors returns the prior probabilities for the
// classes provided -- P(C_i). There is a way to
// smooth priors, currently not implemented here.
func (c *Classifier) getPriors() (priors []float64) {
    n := len(c.Classes)
    priors = make([]float64, n, n)
    sum := 0
    for index, class := range c.Classes {
        total := c.datas[class].Total;
        priors[index] = float64(total)
        sum += total
    }
    if sum != 0 {
        for i := 0; i < n; i++ {
            priors[i] /= float64(sum)
        }
    }
    return
}

// Learn will train the classifier on the provided data.
func (c *Classifier) Learn(words []string, which Class) {
    data := c.datas[which]
    for _, word := range words {
        data.Freqs[word]++
        data.Total++
    }
}

// Scores will produce an array of scores that correspond
// to its opinion on the document in question, and whether it
// belongs to the given class. The order of the scores
// in the return values follows the order of the inital array
// of Class objects parameterized to the NewClassifier() function.
// If no training data has been provided, c will return
// a 0 array.
//
// The value of the score is proportional to the likelihood,
// even if the score is negative, so that the score with the
// greatest value corresponds to the most likely class.
//
// Additionally, c function will return the index of the 
// maximum probability. The value of c number is given by
// scores[inx]. The class of that corresponds to c number
// is classifier.Classes[inx]. If more than one of the
// returned probabilities has the maximum values, then
// strict is false.
func (c *Classifier) Scores(words []string) (scores []float64, inx int, strict bool) {
    n := len(c.Classes)
    scores = make([]float64, n, n)
    priors := c.getPriors()

    // calculate the score for each class
    for index, class := range c.Classes {
        data := c.datas[class]
        // c is the sum of the logarithms 
        // as outlined in the refresher
        score := math.Log(priors[index])
        for _, word := range words {
            score += math.Log(data.getWordProb(word))
        }
        scores[index] = score
    }
    inx, strict = findMax(scores)
    return scores, inx, strict
}

// Probabilities works the same as Score, but delivers
// actual probabilities as discussed above. Note that float64
// underflow is possible if the word list contains too
// many doc that have probabilities very close to 0.
func (c *Classifier) Probabilities(doc []string) (scores []float64, inx int, strict bool) {
    n := len(c.Classes)
    scores = make([]float64, n, n)
    priors := c.getPriors()
    sum := float64(0)
    // calculate the score for each class
    for index, class := range c.Classes {
        data := c.datas[class]
        // c is the sum of the logarithms 
        // as outlined in the refresher
        score := priors[index]
        for _, word := range doc {
            score *= data.getWordProb(word)
        }
        scores[index] = score
        sum += score
    }
    for i := 0; i < n; i++ {
        scores[i] /= sum
    }
    inx, strict = findMax(scores)
    return scores, inx, strict
}

// WordFrequencies returns a matrix of word frequencies that currently
// exist in the classifier for each class state for the given input
// words. In other words, if you obtain the frequencies
//
//    freqs := c.WordFrequencies(/* ... array of j words ... */)
//
// then the expression freq[i][j] represents the frequency of the j-th
// word within the i-th class.
func (c *Classifier) WordFrequencies(words []string) (freqMatrix [][]float64) {
    n, l := len(c.Classes), len(words)
    freqMatrix = make([][]float64, n)
    for i, _ := range freqMatrix {
        arr := make([]float64, l)
        data := c.datas[c.Classes[i]]
        for j, _ := range arr {
            arr[j] = data.getWordProb(words[j])
        }
        freqMatrix[i] = arr
    }
    return
}

// Serialize this classifier to a file.
func (c *Classifier) WriteToFile(name string) (err os.Error) {
    file, err := os.OpenFile(name, os.O_WRONLY | os.O_CREATE, 0655)
    if err != nil {
        return err
    }
    enc := gob.NewEncoder(file)
    err = enc.Encode(&serializableClassifier{c.Classes, c.datas})
    return
}

// findMax finds the maximum of a set of scores; if the
// maximum is strict -- that is, it is the single unique
// maximum from the set -- then strict has return value
// true. Otherwise it is false.
func findMax(scores []float64) (inx int, strict bool) {
    inx = 0
    strict = true
    for i := 1; i < len(scores); i++ {
        if scores[inx] < scores[i] {
            inx = i
            strict = true
        } else if scores[inx] == scores[i] {
            strict = false
        }
    }
    return
}
