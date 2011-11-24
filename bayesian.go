// A Naive Bayesian Classifier
// Jake Brukhman <jbrukh@gmail.com>

// 
// BAYESIAN CLASSIFICATION REFRESHER: suppose you have a set
// of classes (e.g. categories) C := {C_1, ..., C_n}, and a
// document D consisting of words D := {W_1, ..., W_k}.
// We wish to ascertain the probability that the document
// belongs to some class C_j, given some set of training data
// associating documents and classes.
//
// By Bayes Theorem, we have that
//
//    P(C_j|D) = P(D|C_j)*P(C_j)/P(D).
//
// The LHS is the probability that the document belongs to class
// C_j, given the document itself, and our program will calculate
// this probability for each j and spit out the most likely class.
// P(C_j) is referred to as the "prior" probability, or the
// probability that a document belongs to C_j in general, without
// seeing the document first. P(D|C_j) is the probability of seeing
// such a document, given that it belongs to C_j. Here, by assuming
// that words appear independently in documents (e.g. naive
// assumption), we can estimate
//
//    P(D|C_j) ~= P(W_1|C_j)*...*P(W_k|C_j)
//
// where P(W_l|C_j) is the probability of seeing the given word
// in a document of the given class. Finally, P(D) is merely a
// scaling factor and is not particularly relevant, unless you want
// to normalize the resulting scores and actually see
// probabilities. In this case, note that
//
//    P(D) = SUM_j(P(D|C_j)*P(C_j)
//
// End of refresher.
//
package bayesian

// defaultProb is the tiny non-zero probability that a word
// we have not seen before appears in the class. 
const defaultProb = 0.001

// This type defines a set of classes that the classifier will
// filter: C = {C_1, ..., C_n}. You should define your classes
// as a set of constants, for example as follows:
//
//    const (
//        Good Class = "good"
//        Bad Class = "bad
//    )
//
type Class string

// Classifier implements the Naive Bayesian Classifier.
type Classifier struct {
    classes []Class
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

// P(W|Cj)
func (this *classData) getWordProb(word string) float32 {
    value, ok := this.freqs[word]
    if !ok {
        return defaultProb
    }
    return float32(value)/float32(this.total)
}

// P(D|C_j)
// Note that words should not be empty
func (this *classData) getWordsProb(words []string) (prob float32) {
    prob = 1
    for _, word := range words {
        prob *= this.getWordProb(word)
    }
    return prob
}

// New creates a new Classifier.
func NewClassifier(classes []Class) (inst *Classifier) {
    inst = new(Classifier)
    inst.classes = classes
    inst.datas = make(map[Class]*classData)
    for _, class := range classes {
        inst.datas[class] = newClassData()
    }
    return
}

// getPriors returns the prior probabilities for the
// classes provided -- P(C_i).
func (this *Classifier) getPriors() (priors []float32) {
    n := len(this.classes)
    priors = make([]float32, n, n)
    sum := 0
    for index, class := range this.classes {
        total := this.datas[class].total;
        priors[index] = float32(total)
        sum += total
    }
    if sum != 0 {
        for i := 0; i < n; i++ {
            priors[i] /= float32(sum)
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

// Score will produce an array of probabilities that correspond
// to its opinion on the document in question, and whether it
// belongs to the given class. The order of the probabilities
// in the return values follows the order of the inital array
// of Class objects parameterized to the New() function. If no
// training data has been provided, this will return a 0 array.
//
// Additionally, this function will return the most likely Class,
// as well whether this maximum is strict. If it is not strict,
// this means that more than one class has the same maximum
// probability.
func (this *Classifier) Score(words []string) (scores []float32, likely Class, strict bool) {
    n := len(this.classes)
    scores = make([]float32, n, n)
    priors := this.getPriors()
    sum := float32(0)
    for index, class := range this.classes {
        data := this.datas[class]
        score := priors[index]*data.getWordsProb(words)
        scores[index] = score
        sum += score
    }
    inx := 0
    strict = true
    for i := 0; i < n; i++ {
        scores[i] /= sum
        if scores[inx] < scores[i] {
            inx = i
            strict = true
        } else if scores[inx] == scores[i] && i != 0 {
            strict = false
        }
    }
    likely = this.classes[inx]
    return
}
