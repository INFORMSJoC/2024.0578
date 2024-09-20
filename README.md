[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# ANATRA: A Noise-Aware Trust-Region Algorithm

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[A Novel Noise-Aware Classical Optimizer for Variational Quantum Algorithms](https://doi.org/10.1287/ijoc.2024.0578) by 
Jeffrey M. Larson, Matt Menickelly, and Jiahao Shi.

## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2024.0578

https://doi.org/10.1287/ijoc.2024.0578.cd

Below is the BibTex for citing this snapshot of the repository.

```
@misc{ANATRA,
  author =        {Larson, Jeffrey M. and Menickelly, Matt and Shi, Jiahao},
  publisher =     {INFORMS Journal on Computing},
  title =         {{ANATRA: A Noise-Aware Trust-Region Algorithm}},
  year =          {2024},
  doi =           {10.1287/ijoc.2024.0578.cd},
  url =           {https://github.com/INFORMSJoC/2024.0578},
  note =          {Available for download at https://github.com/INFORMSJoC/2024.0578},
}  
```

## Description

This software is sufficient to reproduce the simulated results demonstrated in the paper
referenced above. 

## Python Dependencies

ANATRA depends on several subroutines found in the IBCDFO implementation of POUNDers.
Consult the top-level README of the [IBCDFO repository](https://github.com/POptUS/IBCDFO)
for Python installation instructions.

The QAOA experiments use [Qiskit](https://github.com/Qiskit); 
please see the Qiskit documentation for proper installation.

As discussed in the paper, the optimizers that we did not implement are SPSA, which is included in Qiskit, and NOMAD,
ImFil, and PyBOBYQA, which are wrapped and distributed in [scikit-quant](https://scikit-quant.readthedocs.io/en/latest/). 
Please consult the scikit-quant documentation for questions pertaining to the wrapped software. 

For exact version information and to ensure you have all necessary packages available to your python environment,
please see the requirements.txt file in this repository. 

## Results

Once you've set up your python environment, the following commands will generate all the data used in the experiments
of the paper.

For the 2D quadratic tests with gaussian noise: 
````
python quadratic_all_other_methods.py 2 'gaussian'
python quadratic_pybobyqa_vs_anatra.py 2 'gaussian'
````
Change the first arguments in the commands from "2" to "10" to switch from 2D to 10D.
Likewise, change `gaussian` to `uniform` for uniform noise. 

Similarly, for the 2D Rosenbrock tests, the experiments with gaussian noise can be run with 
````
python rosenbrock_all_other_methods.py 2 'gaussian'
python rosenbrock_pybobyqa_vs_anatra.py 2 'gaussian'
````

The dimension and noise type arguments can be changed as in the experiments with the quadratic function. 

Finally, for the simulated QAOA tests on the toy graph, run
````
python QAOA_all_other_methods.py 0
python QAOA_pybobyqa_vs_anatra.py 0
````
Change the `0` argument to a `1` for the Chvatal graph. Inspect the source to see other graphs that were not experimented
with in the paper. Additionally, the source exposes how to change the depth of the QAOA circuit. 

Once all data is generated, to generate plots similar to those found in the paper, use the commands
````
python paper_figures_quadratic.py
python paper_figures_rosenbrock.py
python paper_figures_QAOA.py
````


## Ongoing Development and Support 
While this code is a snapshot, we do expect to eventually include ANATRA in the IBCDFO repository linked previously. 
A future release of this code will address the practical issue mentioned in the paper that the cost of linear algebra
routines in `lagrange_utils.py` is currently dominating the cost of evaluating the blackbox, even in the QAOA
simulation. Of course, however, addressing this practical issue will almost certainly result in performance differences
from the snapshot stored here.

For support involving this code, contact Matt Menickelly directly by email, provided ANATRA has not yet been released 
in IBCDFO. 