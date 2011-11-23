package bayesian

import "bytes"
import "fmt"

const defaultProb = 0.001

// This type defines a set of classes
// that the classifier will filter.
// C = {C1, ..., Cn}
type Class int

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

// P(D|Cj)
// Note that words should not be empty
func (this *classData) getWordsProb(words []string) (prob float32) {
    prob = 1
    for _, word := range words {
        prob *= this.getWordProb(word)
    }
    return prob
}

// New creates a new Classifier.
func New(classes []Class) (inst *Classifier) {
    inst = new(Classifier)
    inst.classes = classes
    inst.datas = make(map[Class]*classData)
    for _, class := range classes {
        inst.datas[class] = newClassData()
    }
    return
}

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

func (this *Classifier) Learn(words []string, which Class) {
    data := this.datas[which]
    for _, word := range words {
        _, present := data.freqs[word]
        if !present {
            data.freqs[word] = 1
        } else {
            data.freqs[word]++
        }
        data.total++
    }
}

func (this *Classifier) Score(words []string) (scores []float32) {
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
    for i := 0; i < n; i++ {
        scores[i] /= sum
    }
    return
}

func PrintArray(arr []float32) (str string) {
    buffer := bytes.NewBufferString("")
    fmt.Fprint(buffer, "[")
    for _, item := range arr {
        fmt.Fprint(buffer, item, ", ")
    }
    fmt.Fprint(buffer, "]")
    return string(buffer.Bytes())
} 
