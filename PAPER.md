A Physics-Informed Neural Network for the 2D Transient Heat Equation: Accuracy, Amortized Cost, and Sensitivity Analysis Against an Independent Finite-Difference Baseline

ABSTRACT

Physics-informed neural networks (PINNs) are often described as a faster alternative to traditional numerical methods for solving partial differential equations. However, the training time of a PINN can be significant, so its computational benefit needs to be considered over repeated use rather than only during inference.

This project trains a PINN for the 2D transient heat equation using a Gaussian hot-spot initial condition. Its results are compared with an independently implemented finite-difference (FD) solver. Before using the FD solver as a reference, its accuracy is checked against the analytical Fourier-series solution at different grid resolutions. The FD solver shows second-order convergence with respect to grid spacing.

The trained PINN achieves relative L2 errors between 0.39% and 0.67% at five evaluation times when compared with the FD reference. The initial and Dirichlet boundary conditions are also satisfied with small errors.

The computational analysis gives a different result. After including the PINN training time and the cost of repeated inference, approximately 3.36 million repeated evaluations would be required for the PINN to reach the same total computational cost as repeatedly running the FD solver for this problem. This is far beyond the number of evaluations normally expected for a single fixed geometry.

A sensitivity analysis was also carried out using a reduced training budget. The results show that network depth has a larger effect on accuracy than the number of collocation points. Overall, the results show that the PINN can provide an accurate approximation for this problem, but its computational benefit depends on the size of the numerical problem and the number of times the trained model is used.

1. INTRODUCTION

The 2D heat equation,

∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)

is a standard parabolic partial differential equation with a known analytical solution for simple initial and boundary conditions. This makes it suitable for testing a physics-informed neural network because the network output can be compared with an independent numerical solution.

This project uses the 2D heat equation to study three main points.

1. Accuracy — how close is the PINN solution to an independently validated numerical reference?

2. Amortized cost — after including the training time, does repeated PINN inference become cheaper than repeatedly solving the problem numerically?

3. Sensitivity — under a fixed training budget, does increasing the number of collocation points or increasing network depth have a larger effect on accuracy?

The project uses a shared computational setup in which the PINN is trained to satisfy the heat equation together with the initial and boundary conditions. An independently implemented finite-difference solver is used as the numerical reference.

2. PROBLEM SETUP

The problem is defined on the following domain:

Domain: [0,1] × [0,1], with t ∈ [0,0.5].

PDE:

u_t = α(u_xx + u_yy)

where α = 0.01.

Initial condition:

u(x,y,0) = exp(-((x-0.5)² + (y-0.5)²) / (2·0.1²))

The initial condition represents a Gaussian hot spot located at the center of the domain.

Boundary condition:

u = 0 on all four boundaries.

At t = 0.5, the diffusion length scale is approximately √(4αT) = 0.141. This keeps the main hot spot away from the boundaries during the simulated time period. Therefore, the Dirichlet boundary condition has a relatively small effect on the main heat distribution in this particular setup.

3. METHOD

3.1 Independent Finite-Difference Reference

An explicit FTCS (Forward-Time, Centered-Space) finite-difference solver is implemented in src/fd_solver.py. The reference solution uses a 64×64 grid.

The timestep is selected using the standard 2D stability condition:

r = α·dt/dx² ≤ 0.25

A safety factor of 0.9 is applied to the timestep.

Before using the FD solver as the reference for evaluating the PINN, its accuracy is checked against the analytical Fourier-series solution of the same problem. The test is performed at three grid resolutions: 32×32, 64×64, and 96×96.

The results show second-order convergence. This provides a check that the FD implementation is behaving as expected before it is used as the reference solution for the PINN.

3.2 PINN Architecture and Training

The PINN is implemented as a fully connected neural network in src/pinn_model.py.

The network takes x, y, and t as inputs and produces the predicted temperature value û(x,y,t).

The architecture contains:

• 4 hidden layers
• 64 units per hidden layer
• tanh activation
• One output for the predicted temperature

The training loss is defined as:

L = λ_PDE·L_PDE + λ_IC·L_IC + λ_BC·L_BC

The weights used are:

(λ_PDE, λ_IC, λ_BC) = (1, 10, 10)

The initial and boundary condition terms therefore have higher weights than the PDE residual.

The PDE loss is calculated using automatic differentiation of the network output with respect to its inputs. The terms u_t, u_xx, and u_yy are calculated using torch.autograd.grad in src/losses.py. retain_graph=True is used so that the same computation graph can be used when calculating the second derivatives.

The training uses:

• 5,000 interior collocation points
• 1,000 initial-condition points
• 1,000 boundary points
• 250 boundary points per wall
• RANDOM_SEED = 42

The points are resampled for each training run.

Adam is used as the optimizer. The learning rate is cosine-decayed from 1e-3 to 1e-5 over 20,000 epochs. Training uses the fixed collocation set with full-batch gradient descent.

An early-stopping threshold of 1e-6 is defined for the total loss. This threshold was not reached during the configured training period.

3.3 Correctness Verification

Before training the PINN, the automatic-differentiation calculations are checked separately using experiments/phase0_autodiff_check.py.

The checks compare the automatic derivatives with hand-derived analytical derivatives. They cover a first derivative, a Laplacian, and a complete PDE residual for a known test function.

All four checks passed. This provides a check against errors such as incorrect derivative order, incorrect grad_outputs shape, or an incorrect sign in the PDE residual calculation.

3.4 Bug Found and Fixed in the PDE Residual Diagnostic

A problem was found in src/evaluation.py in the evaluate_pde_residual_on_grid function.

The function originally calculated the PDE derivatives separately instead of using the existing derivative calculation in losses.py. There were two related problems with graph retention.

First, the second-derivative calls used create_graph=False without explicitly retaining the graph. Under these conditions, torch.autograd.grad could free the shared computation graph after the first second-derivative calculation, preventing the calculation of u_yy.

Second, an explicit graph-retention flag added to the shared helper also affected the first derivatives u_t, u_x, and u_y. Setting the flag to False for evaluation therefore stopped the tracking needed for the later second derivatives.

The fix was to use the existing compute_pde_residual() function from losses.py with create_graph=True and detach only the final numerical result.

This avoids maintaining two separate implementations of the same derivative calculation and keeps the evaluation code consistent with the training loss.

4. RESULTS

The complete console output is stored in full_run.log. The figures are stored in the figures/ directory, while the numerical results are stored in the results/*.json files.

4.1 FD Solver Validation Against the Analytical Solution

Table 1. FD Solver Validation

Grid        Relative L2 Error       RMSE

32×32       1.41×10⁻⁴              6.21×10⁻⁵
64×64       3.47×10⁻⁵              1.55×10⁻⁵
96×96       1.53×10⁻⁵              6.84×10⁻⁶

Figure 1. FD solver convergence

The error decreases close to the expected factor of four when the grid spacing is reduced. This indicates second-order convergence and provides evidence that the FD solver is suitable for use as the numerical reference.

4.2 PINN Training

The PINN completed all 20,000 configured training epochs. The early-stopping threshold of 1e-6 was not reached.

The final total loss was 3.38×10⁻⁵. Training took 1,681.9 seconds, or approximately 28.0 minutes, on CPU.

Figure 2. PINN training loss

The loss decreases overall during training, although there are periodic increases in the curve. These changes occur during the cosine learning-rate schedule while the model is trained on the fixed collocation points.

4.3 PINN Accuracy Against the FD Reference

Table 2. PINN Accuracy

Time       Relative L2 Error       RMSE          Maximum Absolute Error

t=0.0      0.668%                  1.17×10⁻³     4.15×10⁻³
t=0.1      0.538%                  8.55×10⁻⁴     4.80×10⁻³
t=0.2      0.588%                  8.65×10⁻⁴     4.27×10⁻³
t=0.3      0.391%                  5.40×10⁻⁴     1.92×10⁻³
t=0.5      0.385%                  4.75×10⁻⁴     1.41×10⁻³

The relative L2 error ranges from 0.385% to 0.668% across the five evaluation times.

The t=0.0 error is 0.668%, which is also the initial-condition error because t=0 is included as one of the evaluation times.

The boundary conditions are also satisfied with small errors. The maximum boundary error is 2.02×10⁻³, while the mean boundary error is 3.91×10⁻⁴ across the sampled boundary points and times.

Figure 3. FD and PINN comparison at all evaluation times

Figure 4. Relative L2 error by evaluation time

The largest error occurs at t=0, where the Gaussian hot spot has its steepest gradients. As time increases, the heat distribution becomes smoother and the error decreases.

4.4 Amortized Computational Cost

Table 3. Computational Cost

Quantity                                      Value

FD solve time, single query, 64×64 grid       4.09 ms
PINN training time                             1,681.9 s (28.0 min)
PINN inference time, 4,096 points              3.59 ms
Break-even query count                         ≈ 3,357,647

Figure 5. Amortized cost crossover

Figure 6. PINN inference time versus query size

The PINN inference time for 4,096 points is 3.59 ms, while the complete FD solve for the 64×64 grid takes 4.09 ms. The difference is small compared with the 28.0 minutes required to train the PINN.

Based on these values, approximately 3.36 million repeated evaluations would be needed for the PINN to reach the same total computational cost as repeatedly running the FD solver for this problem.

For this particular 64×64 problem, such a number of evaluations is not a realistic usage pattern for a single fixed geometry.

This result does not mean that PINNs cannot be useful. The computational balance can change with problem size and query volume. FD computation becomes more expensive as grid resolution and dimensionality increase, while a trained PINN can continue to provide predictions through inference. Repeated evaluation of a problem or use across a family of initial or boundary conditions could therefore produce a different cost comparison.

For the single fixed 64×64 heat-equation problem tested here, the FD solver is already sufficiently fast that the PINN training cost is not recovered through repeated inference within a realistic number of queries.

4.5 Sensitivity Analysis

The sensitivity experiments use 3,000 training epochs instead of the 20,000 epochs used for the main PINN experiment. This was done to keep the total experiment time manageable.

Therefore, the absolute error values in this section should not be compared directly with the fully trained model in Section 4.3. The results are intended for comparison between the configurations in the same sensitivity experiment.

4.5.1 Collocation Point Count

Architecture: 4×64

Evaluation time: t=0.5

Table 4. Collocation Point Sensitivity

Interior Points       Relative L2 Error       Training Time

2,500                  3.86%                   2.0 min
5,000                  4.12%                   4.2 min
10,000                 3.97%                   8.4 min

Figure 7. Collocation count sensitivity

The three configurations have similar error values at the reduced training budget. Increasing the number of interior points from 2,500 to 10,000 therefore does not produce a clear improvement in accuracy.

Training time increases as the number of collocation points increases. In this experiment, using more than 2,500 points did not provide a measurable accuracy benefit.

4.5.2 Architecture Depth

Collocation points: 5,000

Evaluation time: t=0.5

Table 5. Architecture Sensitivity

Architecture          Parameters       Relative L2 Error       Training Time

3×32                   2,273            7.86%                   1.1 min
4×64                   12,801           4.12%                   4.2 min
5×64                   16,961           1.66%                   5.5 min

Figure 8. Architecture sensitivity

Network depth has a much larger effect on accuracy than the collocation point count in this experiment.

Increasing the network from 3×32 to 5×64 reduces the relative L2 error from 7.86% to 1.66%. Training time increases from 1.1 minutes to 5.5 minutes.

The results suggest that, at this training budget, network capacity has a stronger effect on accuracy than increasing the number of collocation points.

5. DISCUSSION

The results give a more specific picture of the PINN's performance than simply describing it as either successful or fast.

The trained PINN reaches below 1% relative L2 error when compared with the independently validated FD reference. This shows that the network can provide an accurate approximation for the selected 2D heat-equation problem.

The computational results are different. The PINN requires about 28 minutes of training, while a complete 64×64 FD solve takes only a few milliseconds. Its inference time is also close to the FD solve time. As a result, the training cost is not recovered unless the trained network is evaluated approximately 3.36 million times.

The cost comparison is specific to the problem tested here. A larger grid, a higher-dimensional problem, or a situation requiring many repeated solutions could change the comparison.

The sensitivity experiments also show that network architecture has a stronger effect on accuracy than collocation point count under the reduced training budget used in the sweep. Increasing the network depth produced a clear reduction in error, while increasing the number of collocation points did not produce a similar improvement.

6. LIMITATIONS

Single PDE Instance

The experiment uses one PDE, one diffusivity value, one Gaussian initial condition, and one domain size. The computational comparison is therefore specific to this problem. Larger grids, 3D problems, or stiffer PDEs could produce different results.

CPU-Only Timing

All timing measurements for training, FD solving, and PINN inference were obtained using a CPU. GPU execution would change the training and inference times and could also change the break-even point.

Reduced Budget for Sensitivity Analysis

The sensitivity experiments use 3,000 epochs instead of the 20,000 epochs used for the main experiment. Their error values therefore represent comparisons within the sensitivity sweep rather than fully trained accuracy values.

Mild Boundary Condition Test

The Gaussian hot spot remains well inside the domain during the simulated period because the diffusion length is approximately 0.141. The homogeneous Dirichlet boundary condition is therefore not strongly tested by this particular setup.

No Loss-Weight Search

The loss weights (λ_PDE, λ_IC, λ_BC) = (1, 10, 10) are fixed values. They were selected based on common PINN practice and were not systematically tuned.

7. FUTURE WORK

8. Increase the FD Grid Size

Larger FD grids or a 3D problem could be tested to determine whether there is a problem size where the amortized cost becomes more favorable for the PINN.

2. Repeat the Cost Analysis on GPU

The computational comparison could be repeated using GPU hardware to measure the effect on both PINN training/inference and FD computation.

3. Use a Parametric Problem

The PINN could be extended to handle a range of parameters, such as different values of α or different initial-condition centers and widths. This would allow one trained network to be used for multiple problem configurations.

4. Tune the Loss Weights

The PDE, initial-condition, and boundary-condition weights could be varied systematically to study their effect on training and final accuracy.

REFERENCES

Raissi, M., P. Perdikaris, and G. E. Karniadakis. “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.” Journal of Computational Physics, 2019.

Standard second-order FTCS finite-difference scheme and stability bound for the 2D heat equation. See a standard numerical PDE methods textbook. Used in this project as the independent numerical validation method.
