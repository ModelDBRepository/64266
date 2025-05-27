# The role of efficient neurotransmitter release in barrel map development

*Modeling in support of*:

Lu HC, Butts DA, Kaeser PS, She WC, Janz R, Crair MC (2006) *Journal of Neuroscience* **26**: 2692-2703.

Mice with a loss-of-function mutation of calcium/calmodulin-activated adenylyl cyclase I (AC1) - *barrelless mice* - have strikingly aberrant cortical development: the thalamic afferents into the barrel cortex do not segregate into whisker-specific barrels. This paper investigates the link between this mutation and the "barrelless" phenotype, and demonstrates that the loss-of-function mutation leads to deficits in presynaptic mechanisms at the thalamocortical synapse.

How might presynaptic deficits disrupt whisker-specific segregation in the barrel cortex? We used a model to demonstrate one possibility: decrease in the release probability at the thalamocortical synapse (which is observed in the *barrelless* mutant) can influence the balance between LTP and LTD (in favor of LTD), which can disrupt whisker segregation. Though how this occurs is easily explained with a conceptual model (described succinctly in the above paper), we also produced a computational simulation of this phenomenon.

## Simple computational model of barrel-specific segregation

This simulation was designed based on the limited knowledge of the physiology of the developing barrel cortex, and made minimal assumptions about unmeasured parameters. See the Methods section of the paper for details. This model is a proof-of-principle for how changes in presynaptic release probability can disrupt barrel segregation, and is by no means a proof that it occurs through this mechanism.

The source code [BRLSmodel.c](http://people.deas.harvard.edu/%7Edbutts/SourceCode/BarrelSim/BRLSmodel.c) is written in ansi c++ and was compiled on a Mac running OS 10. Many features of the simulation can be adjusted through the 'define' section at the beginning. On most unix systems, to compile at a UNIX prompt:

c++ BRLSmodel.c -O3 -lm -o sim

To run:

./sim 32000 0.3 0.78 1 P03seg.dat

where `32000` is the simulation time in seconds, `0.3` is the release probability, `0.78` is the threshold of the cortical neuron, `1` is the random number seed (any integer), and `P03seg.dat` is the data file produced by the simulation. The data file is a table with successive columns representing the firing rate of the cortical neuron, the synaptic weights of the primary whisker, and those of the adjacent whisker, at 100 s intervals. All of these parameters can be changed by modifying the source code and recompiling. A [file](http://people.deas.harvard.edu/%7Edbutts/SourceCode/BarrelSim/Figs.txt) demonstrating how this data can be analyzed in Matlab is also included.

The threshold is initially chosen to result in a 0.1 Hz firing of the cortical neuron, and can initially be found by running the simulation in 'test' mode by entering a negative threshold. This will prevent synaptic weights from changing and average the postsynaptic firing rate over a longer time:

sim 4000 0.3 -0.78

The source code of this simulation, as well as text files with command that produce sample data, and the data itself can be found in the following [directory](http://people.deas.harvard.edu/%7Edbutts/SourceCode/BarrelSim/). Please feel free to [email me](mailto:dbutts@deas.harvard.edu) with questions about the simulation.

---

2025-05-27 – Standardized to Markdown.