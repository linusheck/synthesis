import click
import os

import paynt.parser.sketch

import stormpy
stormpy.set_loglevel_error()
import random
import numpy as np
import time
import json
from tqdm import tqdm

def sample_to_list(sample, dt_colored_mdp_factory, model_info):
    bitvector, unreachable_states = sample
    state_to_choice = bitvector_to_state_to_choice(bitvector, model_info)
    result_list = []
    for state, choice in enumerate(state_to_choice):
        if unreachable_states[state]:
            result_list.append(-1)
        else:
            result_list.append(dt_colored_mdp_factory.choice_to_action[choice])
    
    return result_list


def get_bitvector_from_scheduler(scheduler, model_info):
    res_bitvector = stormpy.storage.BitVector(model_info["nr_choices"])
    for state in range(model_info["nr_states"]):
        choice_index = scheduler.get_choice(state).get_deterministic_choice()
        res_bitvector.set(model_info["nondeterministic_choice_indices"][state] + choice_index)

    return res_bitvector


def bitvector_to_state_to_choice(bitvector, model_info):
    state_to_choice = [None] * model_info["nr_states"]
    for state in range(model_info["nr_states"]):
        for choice in range(model_info["nr_choices_per_state"][state]):
            if bitvector.get(model_info["nondeterministic_choice_indices"][state] + choice):
                state_to_choice[state] = model_info["nondeterministic_choice_indices"][state] + choice
                break
    return state_to_choice

def state_to_choice_to_bitvector(state_to_choice, model_info):
    bitvector = stormpy.storage.BitVector(model_info["nr_choices"])
    unreachable_states = []
    for state, choice in enumerate(state_to_choice):
        if choice is not None:
            unreachable_states.append(False)
            bitvector.set(choice)
        else:
            unreachable_states.append(True)
    return bitvector, unreachable_states

def remove_unreachable_choices_from_bitvector(bitvector, dt_colored_mdp_factory, model_info):
    state_to_choice = bitvector_to_state_to_choice(bitvector, model_info)
    state_to_choice = dt_colored_mdp_factory.discard_unreachable_choices(state_to_choice)
    new_bitvector, unreachable_states = state_to_choice_to_bitvector(state_to_choice, model_info)
    return new_bitvector, unreachable_states

def propose_policy(current, model_info):
    """Propose a neighbor by changing one random state's action. Symmetric proposal."""
    state = random.randint(0, model_info["nr_states"] - 1)
    new_action = random.randint(0, model_info["nr_choices_per_state"][state] - 1)

    proposed = stormpy.storage.BitVector(current)
    base = model_info["nondeterministic_choice_indices"][state]
    for c in range(model_info["nr_choices_per_state"][state]):
        proposed.set(base + c, False)
    proposed.set(base + new_action)

    return proposed, state

def check_satisfying(bitvector, dt_colored_mdp_factory, specification):
    submdp = dt_colored_mdp_factory.build_from_choice_mask(bitvector)
    result = submdp.model_check_property(specification.all_properties()[0])
    return result.sat


# ---------------------------------------------------------------------------
# MCMC sampler
# ---------------------------------------------------------------------------

def mcmc_uniform(initial_policy, model_info, dt_colored_mdp_factory, specification,
                 n_samples=1000, burn_in=1000, thin=100, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    current = initial_policy
    _, current_unreachable = remove_unreachable_choices_from_bitvector(current, dt_colored_mdp_factory, model_info)

    samples = []
    total_steps = burn_in + n_samples * thin
    n_accepted = 0
    n_skipped_unreachable = 0

    for step in tqdm(range(total_steps), desc="MCMC", unit="step"):
        # --- propose ---
        proposed, changed_state = propose_policy(current, model_info)

        # --- accept / reject ---
        if current_unreachable[changed_state]:
            # Changed an unreachable state: induced DTMC unchanged, accept for free.
            current = proposed
            n_accepted += 1
            n_skipped_unreachable += 1
        elif check_satisfying(proposed, dt_colored_mdp_factory, specification):
            current = proposed
            _, current_unreachable = remove_unreachable_choices_from_bitvector(current, dt_colored_mdp_factory, model_info)
            n_accepted += 1
        # else: reject, stay at current

        # --- collect ---
        if step >= burn_in and (step - burn_in) % thin == 0:
            reduced, unreachable = remove_unreachable_choices_from_bitvector(current, dt_colored_mdp_factory, model_info)
            samples.append((reduced, unreachable))

    print(f"acceptance rate:      {n_accepted / total_steps * 100:.1f}%")
    print(f"unreachable shortcut: {n_skipped_unreachable / total_steps * 100:.1f}%")
    n_unique = len(set(bv for bv, _ in samples))
    print(f"unique policies:      {n_unique}/{len(samples)}")
    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument('project', type=click.Path(exists=True))
@click.option("--sketch", default="sketch.templ", show_default=True,
    help="name of the sketch file in the project")
@click.option("--props", default="sketch.props", show_default=True,
    help="name of the properties file in the project")
@click.option("--seed", type=int, default=None, show_default=True, help="random seed for policy sampling")
@click.option("--samples", type=int, default=1000, show_default=True, help="number of policies to collect")
@click.option("--burn-in", type=int, default=1000, show_default=True, help="MCMC burn-in steps before collecting")
@click.option("--thin", type=int, default=10, show_default=True, help="collect every thin-th step after burn-in")
@click.option("--output", type=click.Path(), default=None, show_default=True, help="file to write the sampled policies to json")
def main(project, sketch, props, seed, samples, burn_in, thin, output):
    sketch_path = os.path.join(project, sketch)
    properties_path = os.path.join(project, props)
    dt_colored_mdp_factory = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)

    underlying_mdp = dt_colored_mdp_factory.quotient_mdp
    specification = dt_colored_mdp_factory.specification

    assert len(specification.constraints) > 0, "specification must have at least one constraint"

    # Get an initial satisfying policy via model checking
    all_choices = stormpy.storage.BitVector(underlying_mdp.nr_choices, True)
    full_mdp = dt_colored_mdp_factory.build_from_choice_mask(all_choices)
    mc_result = full_mdp.model_check_property(specification.all_properties()[0])
    assert mc_result.sat, "no satisfying policy exists for this specification"
    scheduler = mc_result.result.scheduler

    model_info = {
        "nr_states": underlying_mdp.nr_states,
        "nr_choices": underlying_mdp.nr_choices,
        "nondeterministic_choice_indices": underlying_mdp.nondeterministic_choice_indices,
        "nr_choices_per_state": []
    }

    model_info["nr_choices_per_state"] = [model_info["nondeterministic_choice_indices"][i] - model_info["nondeterministic_choice_indices"][i-1] for i in range(1, len(model_info["nondeterministic_choice_indices"]))]

    initial_policy = get_bitvector_from_scheduler(scheduler, model_info)

    start_time = time.time()
    all_samples = mcmc_uniform(
        initial_policy, model_info, dt_colored_mdp_factory, specification,
        n_samples=samples, burn_in=burn_in, thin=thin, seed=seed,
    )
    end_time = time.time()
    print(f"sampling took {end_time - start_time:.2f} seconds")

    print(f"number of policies collected: {len(all_samples)}")

    output_dict = {"X" : dt_colored_mdp_factory.relevant_state_valuations, "Y" : [sample_to_list(sample, dt_colored_mdp_factory, model_info) for sample in all_samples]}
    if output is not None:
        with open(output, "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    main()
