# CUDA-Lesson-2
## Why does this exist?
Code for teaching blocks &amp; threads on matrix transposition with CUDA.
This code is aimed at teaching CUDA, and provides a key which would be achievable to a new learner of CUDA (which means we DON'T use the most efficient coalesced transposition provided by the CUDA sample files). The key for matrix transposition currently does not function with small thread batches, and is instead is meant to display the power provided by CUDA. This is achieved by throwing a rediculous amount of blocks and threads at the problem, and is a relatively maintainable approach given that we will run out of memory on the heap well before we run out of scheduling space for threads (in most cases).
#
