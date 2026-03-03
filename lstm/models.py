import equinox as eqx
import jax
import jax.random as jrandom
import jax.numpy as jnp
import jax.nn as jnn
import math
from jaxtyping import Array, PRNGKeyArray
from typing import Any


def default_floating_dtype():
    return jnp.float32


def default_init(
    key: PRNGKeyArray, shape: tuple[int, ...], dtype: Any, lim: float
) -> jax.Array:
    return jrandom.uniform(key, shape, dtype, minval=-lim, maxval=lim)


class EALSTMCell(eqx.Module):
    weight_ih: Array
    weight_hh: Array
    bias: Array
    input_gate: eqx.nn.Linear
    dynamic_input_size: int = eqx.field(static=True)
    static_input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        dynamic_input_size: int,
        static_input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        dtype=None,
        initial_forget_bias: float = 3.0,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey, lkey = jrandom.split(key, 4)
        lim = math.sqrt(1 / hidden_size)
        self.weight_ih = default_init(
            ihkey, (3 * hidden_size, dynamic_input_size), dtype, lim
        )
        self.weight_hh = default_init(hhkey, (3 * hidden_size, hidden_size), dtype, lim)
        if use_bias:
            f_bias = jnp.full((hidden_size,), initial_forget_bias, dtype=dtype)
            g_bias = jnp.zeros((hidden_size,), dtype=dtype)
            o_bias = jnp.zeros((hidden_size,), dtype=dtype)
            self.bias = jnp.concatenate([f_bias, g_bias, o_bias], axis=0)
        else:
            self.bias = jnp.zeros((3 * hidden_size,), dtype=dtype)
        self.dynamic_input_size = dynamic_input_size
        self.static_input_size = static_input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.input_gate = eqx.nn.Linear(
            static_input_size, hidden_size, use_bias=use_bias, key=lkey
        )

    def __call__(
        self,
        inputs: tuple[Array, Array],
        hidden: tuple[Array, Array],
        *,
        key: PRNGKeyArray | None = None,
    ):
        x, i = inputs
        h, c = hidden
        gates = self.weight_ih @ x + self.weight_hh @ h + self.bias
        f, g, o = jnp.split(gates, 3)
        f = jnn.sigmoid(f)
        g = jnn.tanh(g)
        o = jnn.sigmoid(o)
        c = f * c + i * g
        h = o * jnn.tanh(c)
        return (h, c)


class EALSTM(eqx.Module):
    cell: EALSTMCell
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        dynamic_input_size: int,
        static_input_size: int,
        hidden_size: int,
        out_size: int,
        dropout_rate: float,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        ckey, lkey = jrandom.split(key)
        self.cell = EALSTMCell(
            dynamic_input_size,
            static_input_size,
            hidden_size,
            use_bias=use_bias,
            dtype=dtype,
            key=ckey,
        )
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=use_bias, key=lkey)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self, dynamic_inputs: Array, static_inputs: Array, *, key: PRNGKeyArray
    ):
        i = jnn.sigmoid(self.cell.input_gate(static_inputs))
        scan_fn = lambda state, x: (self.cell((x, i), state), None)
        init_state = (
            jnp.zeros(self.cell.hidden_size),
            jnp.zeros(self.cell.hidden_size),
        )
        (out, _), _ = jax.lax.scan(scan_fn, init_state, dynamic_inputs)
        out = self.dropout(out, key=key)
        return self.linear(out)
