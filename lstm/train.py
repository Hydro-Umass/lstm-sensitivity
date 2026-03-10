import optax
import equinox as eqx
import jax
import jax.random as jrn
import jax.numpy as jnp
from tqdm import tqdm
import h5py

from lstm import camels
from lstm.models import EALSTM


def train_ealstm(
    h5path,
    hidden_size=256,
    dropout_rate=0.1,
    batch_size=512,
    seed=5678,
    epochs=30,
    clip_gradient_norm=1.0,
    preload=False,
    perturbation=None
):
    eps = 1e-6
    data_key, model_key, dropout_key, pert_key = jrn.split(jrn.PRNGKey(seed), 4)
    with h5py.File(h5path, "r") as f:
        dataset_size = f["x"].shape[0]
        nvars = f["x"].shape[2]
        nsvars = f["xs"].shape[1]
    model = EALSTM(nvars, nsvars, hidden_size, 1, dropout_rate, key=model_key)
    steps_per_epoch = dataset_size // batch_size
    total_steps = epochs * steps_per_epoch
    # define the optimizer, learning rate schedule and optimizer state
    boundaries = [15 * steps_per_epoch, 25 * steps_per_epoch]
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.constant_schedule(1e-3),
            optax.constant_schedule(5e-4),
            optax.constant_schedule(1e-4),
        ],
        boundaries=boundaries,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(clip_gradient_norm), optax.adam(lr_schedule)
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    # define training data loader
    train_loader = camels.dataloader(h5path, batch_size, data_key, shuffle=True, preload=preload, perturbation=perturbation, pert_key=pert_key if perturbation is not None else None)

    @eqx.filter_value_and_grad
    def compute_loss(model, x, xs, y, s, key):
        keys = jrn.split(key, x.shape[0])
        pred_y = jax.vmap(model)(x, xs, key=keys)
        return jnp.mean(jnp.square(y - pred_y) / jnp.square(s + eps))

    @eqx.filter_jit
    def make_step(model, x, xs, y, s, opt_state, key):
        loss, grads = compute_loss(model, x, xs, y, s, key)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    pbar = tqdm(total=total_steps, desc="Training")
    for step, (xs, xss, ys, sbs) in zip(range(total_steps), train_loader):
        dropout_key, step_key = jrn.split(dropout_key)
        loss, model, opt_state = make_step(model, xs, xss, ys, sbs, opt_state, step_key)
        loss = loss.item()
        current_epoch = (step + 1) // steps_per_epoch
        pbar.set_postfix(epoch=current_epoch, loss=f"{loss:.4f}")
        pbar.update(1)
        # Log at end of each epoch
        if (step + 1) % steps_per_epoch == 0:
            tqdm.write(f"Epoch {current_epoch} completed - Loss: {loss:.6f}")
    return model
