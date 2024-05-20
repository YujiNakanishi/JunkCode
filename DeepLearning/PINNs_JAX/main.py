import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import flax
from flax import linen as nn
from flax.training import train_state
import optax
import sys
import pandas as pd

#####モデルの定義
@jit
def snake(x, a2 = 4.):
    return x + (1.- jnp.cos(a2*x))/a2

class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = snake(nn.Dense(100)(x))
        x = snake(nn.Dense(100)(x))
        x = snake(nn.Dense(100)(x))
        x = snake(nn.Dense(100)(x))
        x = nn.Dense(1)(x)

        return x

key = jax.random.PRNGKey(0)
net = Net()
params = net.init(key, jnp.ones(1))["params"]
tx = optax.adam(learning_rate=1e-5)
state = train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

#####ロス関数の定義
@jit
def bc_loss(params, state, inputs, outputs):
    preds = state.apply_fn({"params" : params}, inputs)
    loss = optax.l2_loss(preds - outputs).mean()

    return loss

@jit
def pde_loss(params, state, inputs):
    apply_sum = lambda inputs, state, params: jnp.sum(state.apply_fn({"params" : params}, inputs))
    dx_apply = lambda inputs, state, params: jnp.sum(grad(apply_sum)(inputs, state, params))
    dxx_apply = grad(dx_apply)

    dxxT = dxx_apply(inputs, state, params)
    return optax.l2_loss(dxxT).mean()

@jit
def loss_fn(params, state, key):
    x_bc1 = jnp.zeros((1, 1))
    loss_bc1 = bc_loss(params, state, x_bc1, 0.)
    x_bc2 = jnp.ones((1, 1))
    loss_bc2 = bc_loss(params, state, x_bc2, 1.)

    x = jax.random.uniform(key, (100, 1))
    loss_pde = pde_loss(params, state, x)

    loss = loss_bc1 + loss_bc2 + loss_pde
    return loss

grad_loss = jit(grad(loss_fn))

@jit
def step(state, key):
    grads = grad_loss(state.params, state, key)
    state = state.apply_gradients(grads=grads)
    return state

for i in range(10000):
    key, subkey = jax.random.split(key)
    state = step(state, key)

x = (jnp.arange(101)*0.01).reshape((-1, 1))
T = state.apply_fn({"params" : state.params}, x)
data = jnp.concatenate((x, T), axis = 1)
data = pd.DataFrame(jnp.asarray(data))
data.to_csv("test.csv")