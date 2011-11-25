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
// possibility of float underflow when calculating P(D|C_j), as
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

import "math"

// defaultProb is the tiny non-zero probability that a word
// we have not seen before appears in the class. 
const defaultProb = 0.001

// the type of float we use here
type float float64

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

// classData holds the frequency data for words in a
// particular class. In the future, we may replace this
// structure with a trie-like structure for more
// efficient storage.
type classData struct {
    freqs map[string]int
    total int
}

// newClassData creates a new empty class data node.
func newClassData() *classData {
    return &classData{
        freqs: make(map[string]int),
    }
}

// P(W|Cj) -- the probability of seeing a particular word
// in a document of this class.
func (this *classData) getWordProb(word string) float {
    value, ok := this.freqs[word]
    if !ok {
        return defaultProb
    }
    return float(value)/float(this.total)
}

// P(D|C_j) -- the probability of seeing this set of words
// in a document of this class.
//
// Note that words should not be empty, and this method of
// calulation is prone to underflow if there are many words
// and their individual probabilties are small.
func (this *classData) getWordsProb(words []string) (prob float) {
    prob = 1
    for _, word := range words {
        prob *= this.getWordProb(word)
    }
    return
}

// New creates a new Classifier.
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

// getPriors returns the prior probabilities for the
// classes provided -- P(C_i). There is a way to
// smooth priors, currently not implemented here.
func (this *Classifier) getPriors() (priors []float) {
    n := len(this.Classes)
    priors = make([]float, n, n)
    sum := 0
    for index, class := range this.Classes {
        total := this.datas[class].total;
        priors[index] = float(total)
        sum += total
    }
    if sum != 0 {
        for i := 0; i < n; i++ {
            priors[i] /= float(sum)
        }
    }
    return
}

// Learn will train the classifier on the provided data.
func (this *Classifier) Learn(words []string, which Class) {
    data := this.datas[which]
    for _, word := range words {
        data.freqs[word]++
        data.total++
    }
}

// Score will produce an array of scores that correspond
// to its opinion on the document in question, and whether it
// belongs to the given class. The order of the scores
// in the return values follows the order of the inital array
// of Class objects parameterized to the NewClassifier() function.
// If no training data has been provided, this will return
// a 0 array.
//
// The value of the score is proportional to the likelihood,
// even if the score is negative, so that the score with the
// greatest value corresponds to the most likely class.
//
// Additionally, this function will return the index of the 
// maximum probability. The value of this number is given by
// scores[inx]. The class of that corresponds to this number
// is classifier.Classes[inx]. If more than one of the
// returned probabilities has the maximum values, then
// strict is false.
func (this *Classifier) Score(words []string) (scores []float, inx int, strict bool) {
    n := len(this.Classes)
    scores = make([]float, n, n)
    priors := this.getPriors()

    // calculate the score for each class
    for index, class := range this.Classes {
        data := this.datas[class]
        // this is the sum of the logarithms 
        // as outlined in the refresher
        score := math.Log(float64(priors[index]))
        for _, word := range words {
            score += math.Log(float64(data.getWordProb(word)))
        }
        scores[index] = float(score)
    }

    // calculate the index of the maximum score
    inx = 0
    strict = true
    for i := 1; i < n; i++ {
        if scores[inx] < scores[i] {
            inx = i
            strict = true
        } else if scores[inx] == scores[i] {
            strict = false
        }
    }
    return
}
