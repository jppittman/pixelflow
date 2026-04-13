# Functional Machine Learning and Linear Algebra

## Denotational Design for Neural Networks
Instead of graphs and stateful execution, we represent mathematical models as pure functions:

- A Matrix is a linear map `A -> B`.
- A Neural Network layer is a generic function `(A -> B) + Bias -> NonLinearity`.