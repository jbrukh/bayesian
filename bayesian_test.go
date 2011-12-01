package bayesian

import "testing"
import "fmt"

const (
    Good Class = "good"
    Bad Class = "bad"
)

func Assert(t *testing.T, condition bool, args ...interface{}) {
    if (!condition) {
        t.Fatal(args)
    }
}

func TestEmpty(t *testing.T) {
    c := NewClassifier("Good", "Bad", "Neutral")
    priors := c.getPriors()
    for _, item := range priors {
        Assert(t, item == 0)
    }
}

func TestNoClasses(t *testing.T) {
    defer func() {
        if err := recover(); err != nil {
            // we are good
        }
    }()
    c := NewClassifier()
    Assert(t, false, "should have panicked:", c)
}

func TestOneClass(t *testing.T) {
    defer func() {
        if err := recover(); err != nil {
            // we are good
        }
    }()
    c := NewClassifier(Good)
    Assert(t, false, "should have panicked:", c)
}

func TestLearn(t *testing.T) {
    c := NewClassifier(Good, Bad)
    c.Learn([]string{"tall", "handsome", "rich"}, Good)
    c.Learn([]string{"bald", "poor", "ugly"}, Bad)

    score, likely, strict := c.Scores([]string{"the", "tall", "man"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]>score[1], "not good, round 1") // this is good
    Assert(t, likely == 0, "not good, round 1")
    Assert(t, strict == true, "not strict, round 1")

    score, likely, strict = c.Scores([]string{"poor", "ugly", "girl"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]<score[1]) // this is bad
    Assert(t, likely == 1)
    Assert(t, strict == true)

    score, likely, strict  = c.Scores([]string{"the", "bad", "man"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]==score[1], "not the same") // same
    Assert(t, likely == 0, "not good") // first one is picked
    Assert(t, strict == false, "not strict")
}

func TestProbabilities(t *testing.T) {
    c := NewClassifier(Good, Bad)
    c.Learn([]string{"tall", "handsome", "rich"}, Good)
    c.Learn([]string{"bald", "poor", "ugly"}, Bad)

    score, likely, strict := c.Probabilities([]string{"the", "tall", "man"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]>score[1], "not good, round 1") // this is good
    Assert(t, likely == 0, "not good, round 1")
    Assert(t, strict == true, "not strict, round 1")

    score, likely, strict = c.Probabilities([]string{"poor", "ugly", "girl"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]<score[1]) // this is bad
    Assert(t, likely == 1)
    Assert(t, strict == true)

    score, likely, strict  = c.Probabilities([]string{"the", "bad", "man"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]==score[1], "not the same") // same
    Assert(t, likely == 0, "not good") // first one is picked
    Assert(t, strict == false, "not strict")
}

func TestWordProbs(t *testing.T) {
    c := NewClassifier(Good, Bad)
    c.Learn([]string{"tall", "handsome", "rich"}, Good)
    data := c.datas[Good]
    Assert(t, data.total == 3)
    Assert(t, data.getWordProb("tall") == float64(1)/float64(3), "tall")
    Assert(t, data.getWordProb("rich") == float64(1)/float64(3), "rich")
}

func TestFreqMatrixConstruction(t *testing.T) {
    c := NewClassifier(Good, Bad)
    freqs := c.WordFrequencies([]string{"a", "b"})
    Assert(t, len(freqs) == 2, "size")
    for i := 0; i < 2; i++ {
        for j := 0; j < 2; j++ {
            Assert(t, freqs[i][j] == defaultProb, i, j)
        }
    }
}
