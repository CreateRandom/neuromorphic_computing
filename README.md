# neuromorphic_computing
A repo for the course Neuromorphic Computing

## Paper
The link for the overleaf document is [here](https://www.overleaf.com/3335321595kpjcffpqnpvt)

## Paper shortlist
* [Constraint satisfaction problem paper](https://www.frontiersin.org/articles/10.3389/fnins.2017.00714/full): Sudoku, Map Coloring, Spin? Problem: @Larsie7205
* [ANN to SNN conversion](https://dl.acm.org/citation.cfm?id=2851613.2851724): Train ANN, convert, then 'optimize': @CreateRandom
* [SNN spiking](https://arxiv.org/pdf/1602.08323.pdf): @Taufred
* [STDP (unsupervised learning)](https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full): @Ionaism

### ANN to SNN conversion

#### Cool 

* Training an ANN, converting an SNN and optimizing can produce actual gains, it's not just an academic exercise
* Faster speed (number of input spikes for the SNN), up to only 42 % of computational steps of ANN
* Greater energy efficiency (number of spikes), low energy settings
* Possible to optimize for either
* Could nicely be applied to other problems
* The researchers released a toolkit that might already offer some of the desired functionality [link](http://sensors.ini.uzh.ch/news_page/snn-conversion-2017.html)

#### Not so cool
* Relatively high number of different methods to implement, some from other papers --> might not be transparent
* Lots of conditions to run and compare for a full replication --> time constraints? Overhead
* The paper is a bit unclear on some aspects, though still generally accessible

#### Unclear
* What framework did they run this on? Probably not tested on neuromorphic hardware?
* It reads like all the optimizations are performed on the ANN, but the paper is murky on that

## Collaborators:
* Taufred
