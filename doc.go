/*Package bayesian is a Naive Bayesian Classifier
  Jake Brukhman <jbrukh@gmail.com>

  BAYESIAN CLASSIFICATION REFRESHER: suppose you have a set
  of classes (e.g. categories) C := {C_1, ..., C_n}, and a
  document D consisting of words D := {W_1, ..., W_k}.
  We wish to ascertain the probability that the document
  belongs to some class C_j given some set of training data
  associating documents and classes.

  By Bayes' Theorem, we have that

    P(C_j|D) = P(D|C_j)*P(C_j)/P(D).

  The LHS is the probability that the document belongs to class
  C_j given the document itself (by which is meant, in practice,
  the word frequencies occurring in this document), and our program
  will calculate this probability for each j and spit out the
  most likely class for this document.

  P(C_j) is referred to as the "prior" probability, or the
  probability that a document belongs to C_j in general, without
  seeing the document first. P(D|C_j) is the probability of seeing
  such a document, given that it belongs to C_j. Here, by assuming
  that words appear independently in documents (this being the
  "naive" assumption), we can estimate

    P(D|C_j) ~= P(W_1|C_j)*...*P(W_k|C_j)

  where P(W_i|C_j) is the probability of seeing the given word
  in a document of the given class. Finally, P(D) can be seen as
  merely a scaling factor and is not strictly relevant to
  classificiation, unless you want to normalize the resulting
  scores and actually see probabilities. In this case, note that

    P(D) = SUM_j(P(D|C_j)*P(C_j))

  One practical issue with performing these calculations is the
  possibility of float64 underflow when calculating P(D|C_j), as
  individual word probabilities can be arbitrarily small, and
  a document can have an arbitrarily large number of them. A
  typical method for dealing with this case is to transform the
  probability to the log domain and perform additions instead
  of multiplications:

    log P(C_j|D) ~ log(P(C_j)) + SUM_i(log P(W_i|C_j))

  where i = 1, ..., k. Note that by doing this, we are discarding
  the scaling factor P(D) and our scores are no longer
  probabilities; however, the monotonic relationship of the
  scores is preserved by the log function.
*/
package bayesian
