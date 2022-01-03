**Welcome to Median Filter using CUDA programming!**
This repository was created by Duc Pham Minh, within the scope of Computer Architecture course, Hanoi University of Science and Technology.

**Requirements:**
Some preparations are needed for this particular code to work:
- Hardware: Computer with NVidia Graphics Card, along with CUDA toolkit installed
  + CUDA toolkit should be installed using terminal: conda install cudatoolkit
- Software:
  + An IDE which is compatible with Python programming
  + Python intepreter & Python Environment, preferred Python 3.8 (which is the version this project created with), Conda environment
  + These following Python libraries: numpy, numba, opencv-python, matplotlib

**Cautions:**
- The initial parameters for the threads_per_block and the blocks_per_grid might not be suitable for your device rightaway. Maybe consider lowering the values if CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES occurs.
- If you do not own a NVidia Graphics Card or CUDA compatible device, it's hard to run around. Consider seeing the result shown in the "Result" directory as a reference instead.
