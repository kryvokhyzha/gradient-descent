# ***Introduction***
Gradient descent is a way to minimize an objective function $$J(\theta)$$ parameterized by a model's
parameters $$\theta\in R^d$$ by updating the parameters in the opposite direction of the gradient of the
objective function $$\Delta_\theta J(\theta)$$ w.r.t. to the parameters. The learning rate $$\eta$$ determines the size of the
steps we take to reach a (local) minimum. In other words, we follow the direction of the slope of the
surface created by the objective function downhill until we reach a valley

## ***Classic Gradient Descent***
Classic gradient descent computes the gradient of the cost function w.r.t. to the parameters $$\theta$$  for the entire training dataset:

$$\theta = {\theta- \eta\Delta_\theta J(\theta)}$$

As we need to calculate the gradients for the whole dataset to perform just one update, batch gradient
descent can be very slow and is intractable for datasets that do not fit in memory. Batch gradient
descent also does not allow us to update our model online, i.e. with new examples on-the-fly.
In code, batch gradient descent looks something like this:
```bash
for i in range ( nb_epochs ):
params_grad = evaluate_gradient ( loss_function , data , params )
params = params - learning_rate * params_grad
```
## ***Stochastic Gradient Descent***
Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example
$$x^i$$ and label $$y^i$$:

$$\theta = {\theta- \eta\Delta_\theta J(\theta; x^i; y^i) }$$

While batch gradient descent converges to the minimum of the basin the parameters are placed in,
SGD’s fluctuation, on the one hand, enables it to jump to new and potentially better local minima.
On the other hand, this ultimately complicates convergence to the exact minimum, as SGD will keep
overshooting. However, it has been shown that when we slowly decrease the learning rate, SGD
shows the same convergence behaviour as batch gradient descent, almost certainly converging to a
local or the global minimum for non-convex and convex optimization respectively. Its code fragment
simply adds a loop over the training examples and evaluates the gradient w.r.t. each example.
```bash
for i in range ( nb_epochs ):
np. random . shuffle ( data )
for example in data :
params_grad = evaluate_gradient ( loss_function , example , params )
params = params - learning_rate * params_grad
```

## ***Momentum***
Momentum is a method that helps accelerate SGD in the relevant direction and dampens
oscillations. It does this by adding a fraction $$\gamma$$ of the update vector of the past time step to the current update vector.

$$v_t = {\gamma v_{t-1} + \eta\Delta_\theta J(\theta) }$$

$$\theta = {\theta - v_t}$$

The momentum term $$\gamma$$ is usually set to 0.9 or a similar value.
Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum
as it rolls downhill, becoming faster and faster on the way (until it reaches its terminal velocity, if there is air resistance, i.e. $$\gamma$$ < 1). The same thing happens to our parameter updates: The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.
## ***RMSprop***
 Instead of accumulating all past squared gradients, RMPprop restricts the window of accumulated past gradients to some fixed size $$w$$.
Instead of inefficiently storing $$w$$ previous squared gradients, the sum of gradients is recursively
defined as a decaying average of all past squared gradients. The running average $$E[g^2]_t$$ at time step $$t$$ then depends (as a fraction $$\gamma$$ similarly to the Momentum term) only on the previous average and the current gradient.
 
$$E[g^2]_t = {0.9E[g^2]_{t-1} + 0.1g^2_t}$$

$$\theta_{t+1} = {\theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_t}$$

 RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients.
Hinton suggests to be set $$\gamma$$ to 0.9, while a good default value for the learning rate $$\eta$$ is 0.001.

## ***Adam***
Adaptive Moment Estimation (Adam) is another method that computes adaptive learning rates
for each parameter. In addition to storing an exponentially decaying average of past squared gradients
$$v_t$$ like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients
$$m_t$$, similar to momentum:

$$m_t = {\beta_1m_{t-1} + (1 - \beta_1)g_t}$$

$$v_t = {\beta_2v_{t-1} + (1 - \beta_2)g_t}$$

$$m_t$$ and $$v_t$$ are estimates of the first moment (the mean) and the second moment (the uncentered
variance) of the gradients respectively, hence the name of the method. As $$m_t$$ and $$v_t$$ are initialized as vectors of 0’s, the authors of Adam observe that they are biased towards zero, especially during the initial time steps, and especially when the decay rates are small.
They counteract these biases by computing bias-corrected first and second moment estimates:

$$\hat{m_t} = {\frac{m_t}{1-\beta_{1}^t}}$$

$$\hat{v_t} = {\frac{v_t}{1-\beta_{2}^t}}$$

They then use these to update the parameters just as we have seen in Adadelta and RMSprop, which
yields the Adam update rule:

$$\theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{\hat{v_t}}+\epsilon}\hat{m_t}$$

The authors propose default values of 0.9 for $$\beta_{1}$$ , 0.999 for $$\beta_{2}$$, and $$10^{-8}$$ for $$\epsilon$$. They show empirically that Adam works well in practice and compares favorably to other adaptive learning-method algorithms.
