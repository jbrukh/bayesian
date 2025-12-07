package bayesian

import (
	"encoding/gob"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
)

// defaultProb is the tiny non-zero probability that a word
// we have not seen before appears in the class. This is used
// as a fallback when Laplace smoothing cannot be applied
// (e.g., when the classifier has no training data).
const defaultProb = 1e-11

// ErrUnderflow is returned when an underflow is detected.
var ErrUnderflow = errors.New("possible underflow detected")

// ErrClassExists is returned when trying to add a class that already exists.
var ErrClassExists = errors.New("class already exists")

// ErrAlreadyConverted is returned when trying to add a class after TF-IDF conversion.
var ErrAlreadyConverted = errors.New("cannot add class after TF-IDF conversion")

// Class defines a class that the classifier will filter:
// C = {C_1, ..., C_n}. You should define your classes as a
// set of constants, for example as follows:
//
//    const (
//        Good Class = "Good"
//        Bad Class = "Bad
//    )
//
// Class values should be unique.
type Class string

// Classifier implements the Naive Bayesian Classifier.
type Classifier struct {
	Classes         []Class
	learned         int   // docs learned
	seen            int32 // docs seen
	datas           map[Class]*classData
	tfIdf           bool
	DidConvertTfIdf bool // we can't classify a TF-IDF classifier if we haven't yet
	// called ConvertTermsFreqToTfIdf
	mu sync.RWMutex // protects Classes and datas for concurrent access
}

// serializableClassifier represents a container for
// Classifier objects whose fields are modifiable by
// reflection and are therefore writeable by gob.
type serializableClassifier struct {
	Classes         []Class
	Learned         int
	Seen            int
	Datas           map[Class]*classData
	TfIdf           bool
	DidConvertTfIdf bool
}

// classData holds the frequency data for words in a
// particular class. In the future, we may replace this
// structure with a trie-like structure for more
// efficient storage.
type classData struct {
	Freqs   map[string]float64
	FreqTfs map[string][]float64
	Total   int
}

// newClassData creates a new empty classData node.
func newClassData() *classData {
	return &classData{
		Freqs:   make(map[string]float64),
		FreqTfs: make(map[string][]float64),
	}
}

// getWordProb returns P(W|C_j) -- the probability of seeing
// a particular word W in a document of this class.
// Uses Laplace smoothing (add-one smoothing) to handle unseen words:
// P(W|C) = (count(W,C) + 1) / (total_words_in_C + vocabulary_size)
func (d *classData) getWordProb(word string) float64 {
	vocab := len(d.Freqs)
	if d.Total == 0 || vocab == 0 {
		return defaultProb
	}
	value := d.Freqs[word] // 0 if not found
	return (value + 1) / (float64(d.Total) + float64(vocab))
}

// newClassifier is the internal constructor that creates a classifier.
// The classes provided should be at least 2 in number and unique,
// or this function will panic.
func newClassifier(tfIdf bool, classes []Class) *Classifier {
	n := len(classes)
	if n < 2 {
		panic("provide at least two classes")
	}

	// check uniqueness
	check := make(map[Class]struct{}, n)
	for _, class := range classes {
		check[class] = struct{}{}
	}
	if len(check) != n {
		panic("classes must be unique")
	}

	c := &Classifier{
		Classes: classes,
		datas:   make(map[Class]*classData, n),
		tfIdf:   tfIdf,
	}
	for _, class := range classes {
		c.datas[class] = newClassData()
	}
	return c
}

// NewClassifierTfIdf returns a new TF-IDF classifier. The classes provided
// should be at least 2 in number and unique, or this method will panic.
func NewClassifierTfIdf(classes ...Class) *Classifier {
	return newClassifier(true, classes)
}

// NewClassifier returns a new classifier. The classes provided
// should be at least 2 in number and unique, or this method will panic.
func NewClassifier(classes ...Class) *Classifier {
	return newClassifier(false, classes)
}

// NewClassifierFromFile loads an existing classifier from
// file. The classifier was previously saved with a call
// to c.WriteToFile(string).
func NewClassifierFromFile(name string) (c *Classifier, err error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	return NewClassifierFromReader(file)
}

// NewClassifierFromReader: This actually does the deserializing of a Gob encoded classifier
func NewClassifierFromReader(r io.Reader) (c *Classifier, err error) {
	dec := gob.NewDecoder(r)
	w := new(serializableClassifier)
	err = dec.Decode(w)

	return &Classifier{
		Classes:         w.Classes,
		learned:         w.Learned,
		seen:            int32(w.Seen),
		datas:           w.Datas,
		tfIdf:           w.TfIdf,
		DidConvertTfIdf: w.DidConvertTfIdf,
		// mu is zero-valued and ready to use
	}, err
}

// AddClass adds a new class to the classifier dynamically.
// Returns ErrClassExists if the class already exists, or
// ErrAlreadyConverted if the classifier has been converted to TF-IDF.
// This method is safe for concurrent use.
func (c *Classifier) AddClass(class Class) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if TF-IDF conversion has happened
	if c.DidConvertTfIdf {
		return ErrAlreadyConverted
	}

	// Check if class already exists
	if _, exists := c.datas[class]; exists {
		return ErrClassExists
	}

	c.Classes = append(c.Classes, class)
	c.datas[class] = newClassData()
	return nil
}

// getPriors returns the prior probabilities for the
// classes provided -- P(C_j).
// Uses Laplace smoothing to ensure no prior is zero:
// P(C_j) = (count_j + 1) / (total + num_classes)
func (c *Classifier) getPriors() (priors []float64) {
	n := len(c.Classes)
	priors = make([]float64, n)
	sum := 0
	for index, class := range c.Classes {
		total := c.datas[class].Total
		priors[index] = float64(total)
		sum += total
	}
	// Apply Laplace smoothing to priors to avoid log(0)
	floatN := float64(n)
	floatSum := float64(sum)
	for i := 0; i < n; i++ {
		priors[i] = (priors[i] + 1) / (floatSum + floatN)
	}
	return
}

// Learned returns the number of documents ever learned
// in the lifetime of this classifier.
func (c *Classifier) Learned() int {
	return c.learned
}

// Seen returns the number of documents ever classified
// in the lifetime of this classifier.
func (c *Classifier) Seen() int {
	return int(atomic.LoadInt32(&c.seen))
}


// IsTfIdf returns true if we are a classifier of type TfIdf
func (c *Classifier) IsTfIdf() bool {
	return c.tfIdf
}

// WordCount returns the number of words counted for
// each class in the lifetime of the classifier.
// This method is safe for concurrent use.
func (c *Classifier) WordCount() (result []int) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	result = make([]int, len(c.Classes))
	for inx, class := range c.Classes {
		data := c.datas[class]
		result[inx] = data.Total
	}
	return
}

// Observe should be used when word-frequencies have been already been learned
// externally (e.g., hadoop).
// This method is safe for concurrent use.
func (c *Classifier) Observe(word string, count int, which Class) {
	c.mu.Lock()
	defer c.mu.Unlock()
	data := c.datas[which]
	data.Freqs[word] += float64(count)
	data.Total += count
}

// Learn will accept new training documents for
// supervised learning.
// This method is safe for concurrent use.
func (c *Classifier) Learn(document []string, which Class) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// If we are a tfidf classifier we first need to get terms as
	// terms frequency and store that to work out the idf part later
	// in ConvertToIDF().
	if c.tfIdf {
		if c.DidConvertTfIdf {
			panic("Cannot call ConvertTermsFreqToTfIdf more than once. Reset and relearn to reconvert.")
		}

		// Term Frequency: word count in document / document length
		docTf := make(map[string]float64)
		for _, word := range document {
			docTf[word]++
		}

		docLen := float64(len(document))

		for wIndex, wCount := range docTf {
			docTf[wIndex] = wCount / docLen
			// add the TF sample, after training we can get IDF values.
			c.datas[which].FreqTfs[wIndex] = append(c.datas[which].FreqTfs[wIndex], docTf[wIndex])
		}

	}

	data := c.datas[which]
	for _, word := range document {
		data.Freqs[word]++
		data.Total++
	}
	c.learned++
}

// ConvertTermsFreqToTfIdf uses all the TF samples for the class and converts
// them to TF-IDF https://en.wikipedia.org/wiki/Tf%E2%80%93idf
// once we have finished learning all the classes and have the totals.
// This method is safe for concurrent use.
func (c *Classifier) ConvertTermsFreqToTfIdf() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.DidConvertTfIdf {
		panic("Cannot call ConvertTermsFreqToTfIdf more than once. Reset and relearn to reconvert.")
	}

	for className := range c.datas {
		for wIndex := range c.datas[className].FreqTfs {
			tfIdfAdder := float64(0)

			for tfSampleIndex := range c.datas[className].FreqTfs[wIndex] {
				// we always want a positive TF-IDF score.
				tf := c.datas[className].FreqTfs[wIndex][tfSampleIndex]
				c.datas[className].FreqTfs[wIndex][tfSampleIndex] = math.Log1p(tf) * math.Log1p(float64(c.learned)/float64(c.datas[className].Total))
				tfIdfAdder += c.datas[className].FreqTfs[wIndex][tfSampleIndex]
			}
			// convert the 'counts' to TF-IDF's
			c.datas[className].Freqs[wIndex] = tfIdfAdder
		}
	}

	c.DidConvertTfIdf = true
}

// LogScores produces "log-likelihood"-like scores that can
// be used to classify documents into classes.
//
// The value of the score is proportional to the likelihood,
// as determined by the classifier, that the given document
// belongs to the given class. This is true even when scores
// returned are negative, which they will be (since we are
// taking logs of probabilities).
//
// The index j of the score corresponds to the class given
// by c.Classes[j].
//
// Additionally returned are "inx" and "strict" values. The
// inx corresponds to the maximum score in the array. If more
// than one of the scores holds the maximum values, then
// strict is false.
//
// Unlike c.Probabilities(), this function is not prone to
// floating point underflow and is relatively safe to use.
// This method is safe for concurrent use.
func (c *Classifier) LogScores(document []string) (scores []float64, inx int, strict bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.tfIdf && !c.DidConvertTfIdf {
		panic("Using a TF-IDF classifier. Please call ConvertTermsFreqToTfIdf before calling LogScores.")
	}

	n := len(c.Classes)
	scores = make([]float64, n)
	priors := c.getPriors()

	// calculate the score for each class
	for index, class := range c.Classes {
		data := c.datas[class]
		score := math.Log(priors[index])
		for _, word := range document {
			score += math.Log(data.getWordProb(word))
		}
		scores[index] = score
	}
	inx, strict = findMax(scores)
	atomic.AddInt32(&c.seen, 1)
	return scores, inx, strict
}

// Classify returns the most likely class for the given document
// along with the log scores and whether the classification is strict.
// This is a convenience wrapper around LogScores that returns the
// Class directly instead of an index.
func (c *Classifier) Classify(document []string) (class Class, scores []float64, strict bool) {
	scores, inx, strict := c.LogScores(document)
	class = c.Classes[inx]
	return
}

// ProbScores works the same as LogScores, but delivers
// actual probabilities as discussed above. Note that float64
// underflow is possible if the word list contains too
// many words that have probabilities very close to 0.
//
// Notes on underflow: underflow is going to occur when you're
// trying to assess large numbers of words that you have
// never seen before. Depending on the application, this
// may or may not be a concern. Consider using SafeProbScores()
// instead.
//
// If all scores underflow to zero, returns equal probabilities
// for all classes (1/n each).
// This method is safe for concurrent use.
func (c *Classifier) ProbScores(doc []string) (scores []float64, inx int, strict bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.tfIdf && !c.DidConvertTfIdf {
		panic("Using a TF-IDF classifier. Please call ConvertTermsFreqToTfIdf before calling ProbScores.")
	}
	n := len(c.Classes)
	scores = make([]float64, n)
	priors := c.getPriors()
	sum := float64(0)
	// calculate the score for each class
	for index, class := range c.Classes {
		data := c.datas[class]
		score := priors[index]
		for _, word := range doc {
			score *= data.getWordProb(word)
		}
		scores[index] = score
		sum += score
	}
	// Handle underflow: if sum is 0, all scores underflowed
	// Return equal probabilities to avoid NaN
	if sum == 0 {
		equal := 1.0 / float64(n)
		for i := 0; i < n; i++ {
			scores[i] = equal
		}
		strict = false
	} else {
		for i := 0; i < n; i++ {
			scores[i] /= sum
		}
		inx, strict = findMax(scores)
	}
	atomic.AddInt32(&c.seen, 1)
	return scores, inx, strict
}

// ClassifyProb returns the most likely class for the given document
// along with the probability scores and whether the classification is strict.
// This is a convenience wrapper around ProbScores that returns the
// Class directly instead of an index.
func (c *Classifier) ClassifyProb(document []string) (class Class, scores []float64, strict bool) {
	scores, inx, strict := c.ProbScores(document)
	class = c.Classes[inx]
	return
}

// SafeProbScores works the same as ProbScores, but is
// able to detect underflow in those cases where underflow
// results in the reverse classification. If an underflow is detected,
// this method returns an ErrUnderflow, allowing the user to deal with it as
// necessary. Note that underflow, under certain rare circumstances,
// may still result in incorrect probabilities being returned,
// but this method guarantees that all error-less invocations
// are properly classified.
//
// Underflow detection is more costly because it also
// has to make additional log score calculations.
//
// When underflow is detected, the returned scores are computed from
// log-domain scores using the log-sum-exp trick for numerical stability.
// This method is safe for concurrent use.
func (c *Classifier) SafeProbScores(doc []string) (scores []float64, inx int, strict bool, err error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if c.tfIdf && !c.DidConvertTfIdf {
		panic("Using a TF-IDF classifier. Please call ConvertTermsFreqToTfIdf before calling SafeProbScores.")
	}

	n := len(c.Classes)
	scores = make([]float64, n)
	logScores := make([]float64, n)
	priors := c.getPriors()
	sum := float64(0)

	// calculate the score for each class
	for index, class := range c.Classes {
		data := c.datas[class]
		score := priors[index]
		logScore := math.Log(priors[index])
		for _, word := range doc {
			p := data.getWordProb(word)
			score *= p
			logScore += math.Log(p)
		}
		scores[index] = score
		logScores[index] = logScore
		sum += score
	}

	// Get the winner from log-domain (always reliable)
	logInx, logStrict := findMax(logScores)

	// Check for underflow: if sum is 0 or prob-domain disagrees with log-domain
	if sum == 0 {
		// Complete underflow - use log-sum-exp to recover probabilities
		err = ErrUnderflow
		scores = logScoresToProbs(logScores)
		inx, strict = logInx, logStrict
	} else {
		for i := 0; i < n; i++ {
			scores[i] /= sum
		}
		inx, strict = findMax(scores)

		// Detect partial underflow - when prob and log domains disagree
		if inx != logInx || strict != logStrict {
			err = ErrUnderflow
			// Use log-domain results as they're more reliable
			scores = logScoresToProbs(logScores)
			inx, strict = logInx, logStrict
		}
	}

	atomic.AddInt32(&c.seen, 1)
	return scores, inx, strict, err
}

// logScoresToProbs converts log-domain scores to probabilities
// using the log-sum-exp trick for numerical stability.
func logScoresToProbs(logScores []float64) []float64 {
	n := len(logScores)
	probs := make([]float64, n)

	// Find max for numerical stability
	maxLog := logScores[0]
	for i := 1; i < n; i++ {
		if logScores[i] > maxLog {
			maxLog = logScores[i]
		}
	}

	// Compute exp(log - max) and sum
	sum := 0.0
	for i := 0; i < n; i++ {
		probs[i] = math.Exp(logScores[i] - maxLog)
		sum += probs[i]
	}

	// Normalize
	for i := 0; i < n; i++ {
		probs[i] /= sum
	}

	return probs
}

// ClassifySafe returns the most likely class for the given document
// along with the probability scores, whether the classification is strict,
// and an error if underflow is detected.
// This is a convenience wrapper around SafeProbScores that returns the
// Class directly instead of an index.
func (c *Classifier) ClassifySafe(document []string) (class Class, scores []float64, strict bool, err error) {
	scores, inx, strict, err := c.SafeProbScores(document)
	class = c.Classes[inx]
	return
}

// WordFrequencies returns a matrix of word frequencies that currently
// exist in the classifier for each class state for the given input
// words. In other words, if you obtain the frequencies
//
//	freqs := c.WordFrequencies(/* [j]string */)
//
// then the expression freq[i][j] represents the frequency of the j-th
// word within the i-th class.
// This method is safe for concurrent use.
func (c *Classifier) WordFrequencies(words []string) (freqMatrix [][]float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	n, l := len(c.Classes), len(words)
	freqMatrix = make([][]float64, n)
	for i := range freqMatrix {
		arr := make([]float64, l)
		data := c.datas[c.Classes[i]]
		for j := range arr {
			arr[j] = data.getWordProb(words[j])
		}
		freqMatrix[i] = arr
	}
	return
}

// WordsByClass returns a map of words and their probability of
// appearing in the given class.
// This method is safe for concurrent use.
func (c *Classifier) WordsByClass(class Class) (freqMap map[string]float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	freqMap = make(map[string]float64)
	for word, cnt := range c.datas[class].Freqs {
		freqMap[word] = float64(cnt) / float64(c.datas[class].Total)
	}
	return freqMap
}


// WriteToFile serializes this classifier to a file.
// This method is safe for concurrent use.
func (c *Classifier) WriteToFile(name string) error {
	file, err := os.OpenFile(name, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	return c.writeGobLocked(file)
}

// WriteClassesToFile writes all classes to files.
// This method is safe for concurrent use.
func (c *Classifier) WriteClassesToFile(rootPath string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	for name := range c.datas {
		if err := c.writeClassToFileLocked(name, rootPath); err != nil {
			return err
		}
	}
	return nil
}

// WriteClassToFile writes a single class to file.
// This method is safe for concurrent use.
func (c *Classifier) WriteClassToFile(name Class, rootPath string) error {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.writeClassToFileLocked(name, rootPath)
}

// writeClassToFileLocked writes a single class to file (caller must hold lock).
func (c *Classifier) writeClassToFileLocked(name Class, rootPath string) error {
	data := c.datas[name]
	fileName := filepath.Join(rootPath, string(name))
	file, err := os.OpenFile(fileName, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer file.Close()
	return gob.NewEncoder(file).Encode(data)
}


// WriteGob serializes this classifier to GOB and writes to Writer.
// This method is safe for concurrent use.
func (c *Classifier) WriteGob(w io.Writer) (err error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.writeGobLocked(w)
}

// writeGobLocked serializes this classifier to GOB (caller must hold lock).
func (c *Classifier) writeGobLocked(w io.Writer) (err error) {
	enc := gob.NewEncoder(w)
	err = enc.Encode(&serializableClassifier{c.Classes, c.learned, int(c.seen), c.datas, c.tfIdf, c.DidConvertTfIdf})
	return
}


// ReadClassFromFile loads existing class data from a
// file.
// This method is safe for concurrent use.
func (c *Classifier) ReadClassFromFile(class Class, location string) (err error) {
	fileName := filepath.Join(location, string(class))
	file, err := os.Open(fileName)
	if err != nil {
		return err
	}
	defer file.Close()

	dec := gob.NewDecoder(file)
	w := new(classData)
	err = dec.Decode(w)
	if err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.learned++
	c.datas[class] = w
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
