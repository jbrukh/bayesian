package bayesian

import "testing"
import "fmt"

func TestEmpty(t *testing.T) {
    c := New([]Class{1,2,3})
    priors := c.getPriors()
    for _, item := range priors {
        if item != 0 {
            t.Fail()
        }
    }
}

func TestLearn(t *testing.T) {
    const (
        Good Class = 0
        Bad Class = 1
    )
    c := New([]Class{Good, Bad})
    c.Learn([]string{"tall", "handsome", "rich"}, Good)
    c.Learn([]string{"bald", "poor", "ugly"}, Bad)
    
    score := c.Score([]string{"the", "tall", "man"})
    fmt.Printf("%v\n", score)
    if score[0] <= score[1] {
        t.Fail()
    }
    score = c.Score([]string{"the", "bad", "man"})
    fmt.Printf("%v\n", score)
    if score[0] != score[1] {
        t.Fail()
    }
}
