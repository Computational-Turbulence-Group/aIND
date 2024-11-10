# IND

Python package for informative/non-informative decomposition of variables in a
dynamical system.

## Contents
<!-- toc -->

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

<!-- tocstop -->

## Introduction

The variables in a complex dynamical system are entangled: the evolution of a
given variable depends on the others. As consequence, it is possible to gain
some understanding about one variable by observing the evolution of other
variables in the system. This has led to the development of methods that tries 
to assess how variables are _informative_ to each other in a dynamical system.

The proposed informative/non-informative decomposition (IND for short) aims as
splitting a given _source_ variable into a component that contains all the
information to predict the state of a target variable in the future; and a
non-informative (residual) contribution that shares no information with the
future state of the _target_ variable.

To discern what informative means, IND uses the Shannon mutual-information
([Shannon, 1948](#shannon)). The details of the algorithm can be found in 
[Arranz & Lozano-Duran (2024)](#arranz).


## Installation


## Usage

The best way of learning how to use IND is to use the notebooks.

## References
<a id="shannon"></a> 
[Shannon, C.E., 1948](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x). 
A mathematical theory of communication. 
The Bell system technical journal, 27(3), pp.379-423.

<a id="arranz"></a> 
[Arranz, G. and Lozano-Durán, A., 2024](https://arxiv.org/pdf/2402.11448.pdf). 
Informative and non-informative decomposition of turbulent flow fields. 

