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
    
    score, likely, strict := c.Score([]string{"the", "tall", "man"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]>score[1], "not good, round 1") // this is good
    Assert(t, likely == Good, "not good, round 1")
    Assert(t, strict == true, "not strict, round 1")
     
    score, likely, strict = c.Score([]string{"poor", "ugly", "girl"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]<score[1]) // this is bad
    Assert(t, likely == Bad)
    Assert(t, strict == true)
    
    score, likely, strict  = c.Score([]string{"the", "bad", "man"})
    fmt.Printf("%v\n", score)
    Assert(t, score[0]==score[1], "not the same") // same
    Assert(t, likely == Good, "not good") // first one is picked
    Assert(t, strict == false, "not strict")
}

func TestWordProbs(t *testing.T) {
    c := NewClassifier(Good, Bad)
    c.Learn([]string{"tall", "handsome", "rich"}, Good)
    data := c.datas[Good]
    Assert(t, data.total == 3)
    Assert(t, data.getWordProb("tall") == float64(1)/float64(3))
    Assert(t, data.getWordsProb([]string{"tall","rich"}) == float64(1)/float64(9))
}
