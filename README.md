# Minimum Chips
Angus Gray-Weale
2024-10-04

Minimum chips implements mathematical operations on a large array for
use with high-performance computing systems. The array is split into
blocks, each one of which is shared between one or more MPI processes.
The code is optionally accelerated with OpenMP. Both Monte-Carlo
simulations of material properties, and solvers for integral equations
for the structures of fluids have been built on the foundation
implemented in `minimum_chips`. The most expensive step in the solution
of integral equations for the structures of fluids of particles with
internal structure is ideal for GPGPU acceleration. CUDA is used for
this step. The public version of this code illustrates the use of C++
templates to program algorithms that may be applied to correlation
functions of different mathematical structure.

## Projects that use this code

### Neutron scattering of high temperature molten slags

High temperatures in important industrial processes, like the
manufacture of steel, makes optimisation of the products and processes
difficult and dangerous. My team developed models for the statistics of
the microstructure of certain molten silicates important in steel
making, and with European colleagues used a beam of neutrons from the
nuclear reactor at the [Institut Laue-Langevin](https://www.ill.eu/) in
France to validate the models. This project laid the foundation for
later work by metallurgists studying the effects of impurities on
metals, and by geologists studying subduction.

![silica](silica.png)

### An explanation for the hydrophobic effect

The force that drives the folding of proteins is explained by exact
analytical results for the statistics of the organisation of water
molecules in the neighbourhood of the protein of the protein. These
results are obtained by asymptotic analysis of infinite series of
interactions of increasing complexity written as graphs. In 2023 the
Royal Society of Chemistry invited me to review this work for their main
chemical physics journal.

### The dynamic surface tension of water

The surface tension of water takes about one thousandth of a second
after a fresh surface is made to reach its normal value. We explained
this change in terms of the changing composition of a surface layer.
This dynamic surface tension is critical in inkjet technologies, and our
discoveries on the composition of the surface of a water droplet is
important in atmospheric chemistry.

![hydroxide](oh_coordination.png)

### Hypervirials and diagrammatic expansions

Most theories of fluid structure assume that atoms and ions remain
spherical. The reason for this simplifying assumption is that if the
particles are rigid, it is easy to work out the force on each particle:
just add up the contributions from all the particles in range of the
interaction. Real atoms, molecules, and ions are polarisable, so that
their electron clouds are distorted by interactions or applied electric
fields. To calculate the force on a polarisable particle, you need first
to know its polarisation, but that polarisation is affected by forces
through the fields. I discovered an algorithm that made the more
difficult calculation for polarisable particles as computationally
tractable as the calculation for rigid particles.

## [How many chips is the minimum?](https://www.abc.net.au/news/2019-12-21/minimum-chips-size-debate-brandon-gatgens/11772776)

![Some chips](chips.png)
