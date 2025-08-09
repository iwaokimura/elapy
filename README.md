# About elapy

This is a Python implementation of Energy Landscape Analysis Toolbox/Toolkit (ELAT). An Ising model is fit to the input data that should be a {0,1}-valued or {-1,1}-valued matrix. From the estimated Ising model, local minimum patterns, basins of attractions, and a disconnectivity graph showing energy barriers between local minimum patterns are calculated. For more details, please see the original repository or the original paper shown below.

The original Matlab codes written by Dr. T. Ezaki are available at: <https://github.com/tkEzaki/energy-landscape-analysis>.

The original paper (T. Ezaki, et al., Philos. Trans. A Math. Phys. Eng. Sci., 2017) is available at: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5434078/>.

Japanese documents are also available at the original codes' author's website: <https://sites.google.com/site/ezakitakahiro/software>.

# Open notebook

- [Open notebook in Colab](https://colab.research.google.com/github/okumakito/elapy/blob/main/elapy.ipynb) (executable)
- [Open notebook in GitHub](https://github.com/okumakito/elapy/blob/main/elapy.ipynb) (not executable)

# Required packages

numpy, scipy, pandas, matplotlib, seaborn, networkx, and plotly

# Install

```bash
git clone https://github.com/okumakito/elapy.git
```

or, if you want to install using pip, 

```bash
pip install git+https://github.com/iwaokimura/elapy.git
```

# Usage

- Import

  ```python
  import elapy as ela
  ```

- Convert your data to a {0,1}-valued pandas DataFrame object with rows represent variables and columns represent observations or time points. You can load a test data provided by the original repository (<https://github.com/tkEzaki/energy-landscape-analysis>) by the following command:
  
  ```python
  data = ela.load_testdata(1)  # the argument is 1, 2, 3, or 4
  ```

- Fit an Ising model to the data. Please choose one from the two functions below depending on the data size.
  
  ```python
  h, W = ela.fit_exact(data)  # exact fitting based on likelihood function
  h, W = ela.fit_approx(data) # approximated fitting based on pseudo-likelihood function
  ```

- Check fitting accuracy scores. The first measure is based on Shannon entropy. The second measure is based on Kullback–Leibler divergence. Both measures take 1 for the best fitting and 0 for the worst fitting. The values of the two measures are always the same for the case of exact fitting.
  
  ```python
  acc1, acc2 = ela.calc_accuracy(h, W, data)
  print(acc1, acc2)
  ```

- Calculate a basin graph.
  
  ```python
  graph = ela.calc_basin_graph(h, W, data)
  ```

- Calculate a disconnectivity graph.
  
  ```python
  D = ela.calc_discon_graph(h, W, data, graph)
  ```

- Calculate each state's frequency and transitions between states.
  
  ```python
  freq, trans, trans2 = ela.calc_trans(data, graph)
  ```

- [Optional] Calculate transition matrix based on Boltzmann machine.
  
  ```python
  P = ela.calc_trans_bm(h, W, data)
  ```

- [Optional] Calculate the depth threshold (Caution: this function is time consuming).

  ```python
  n, k = data.shape
  th = ela.calc_depth_threshold(n, k)
  print(th)
  ```

- Plot figures
  
  ```python
  ela.plot_local_min(data, graph)
  ela.plot_basin_graph(graph)
  ela.plot_discon_graph(D)
  ela.plot_landscape(D)
  ela.plot_landscape3d(D)
  ela.plot_trans(freq, trans, trans2)
  ```

# Major differences from the original codes

- Each {0,1}-valued vector is encoded to a decimal value differently from the original codes. The last element is taken as the most significant bit (MSB) in the original codes (for example, [0,0,0,1] -> 8) whereas the first element is taken as the MSB in this implementation (for example, [0,0,0,1] -> 1). Moreover, 1 is added to each decimal value in the original codes to avoid 0. The original notations can be used in `plot_basin_graph` function as follows:
  
  ```python
  ela.plot_basin_graph(graph, original_notation=True)
  ```

- 3D plot of a basin graph is not provided in this repository.
- `calc_trans_bm` is added to calculate the transition matrix based on Boltzmann machine. The transition probability is for the case of synchronous updates, that is, when the values of all variables are updated simultaneously. The row number and column number correspond to the destination node number and source node number, respectively.
- `calc_depth_threshold` is added to calculate the depth threshold using Monte Carlo method. An energy valley deeper than the threshold is considered to have a significant depth. The depth of a state is defined as the energy gap between the local minimum pattern of the state and the saddle pattern with the minimum energy that connects the state and another state.
- `plot_landscape` is added to show a two-dimensional landscape reconstructed from the disconnectivity graph. Movement is only allowed on the white lines.
- `plot_landscape3d` is added to show a three-dimensional landscape reconstructed from the disconnectivity graph. Movement is only allowed on the white lines.
