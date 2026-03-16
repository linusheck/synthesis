import click
import os

import paynt.parser.sketch

import stormpy
import random
import numpy as np
import time
import json

from sklearn import tree, svm
import matplotlib.pyplot as plt

from get_predicates import get_atomic_predicate_evals


def get_mdp_features_list(dt_colored_mdp_factory, additional_atomic_predicates={}):
    features = dt_colored_mdp_factory.relevant_state_valuations
    for predicate_name, predicate_eval in additional_atomic_predicates.items():
        for state in range(len(features)):
            features[state].append(1 if predicate_eval.get(state) else 0)
    return features


def sample_to_list(sample, dt_colored_mdp_factory, model_info):
    bitvector, unreachable_states = sample
    state_to_choice = bitvector_to_state_to_choice(bitvector, model_info)
    result_list = []
    for state, choice in enumerate(state_to_choice):
        if unreachable_states[state] or not dt_colored_mdp_factory.state_is_relevant_bv.get(state):
            result_list.append(-1)
        else:
            result_list.append(dt_colored_mdp_factory.choice_to_action[choice])
    
    return result_list


def get_optimality_specification(specification):
    specification.constraints[0].threshold = 0
    specification.constraints[0].property.raw_formula.set_bound(specification.constraints[0].formula.comparison_type, stormpy.ExpressionManager().create_rational(stormpy.Rational(0)))
    opt_property = stormpy.Property("", specification.constraints[0].formula.clone())

    paynt_opt_property = paynt.verification.property.construct_property(opt_property, 0, False)
    properties = [paynt_opt_property]

    return paynt.verification.property.Specification(properties)


def get_scheduler(model, prop):
    formula = prop.formula
    res = stormpy.model_checking(model, formula, extract_scheduler=True)
    return res.scheduler


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

def state_to_choice_to_bitvector(state_to_choice, dt_colored_mdp_factory, model_info):
    bitvector = stormpy.storage.BitVector(model_info["nr_choices"])
    unreachable_states = []
    for state, choice in enumerate(state_to_choice):
        if choice is not None:
            unreachable_states.append(False)
            if dt_colored_mdp_factory.state_is_relevant_bv.get(state):
                bitvector.set(choice)
        else:
            unreachable_states.append(True)
    return bitvector, unreachable_states


def remove_unreachable_choices_from_bitvector(bitvector, dt_colored_mdp_factory, model_info):
    state_to_choice = bitvector_to_state_to_choice(bitvector, model_info)
    state_to_choice = dt_colored_mdp_factory.discard_unreachable_choices(state_to_choice)
    new_bitvector, unreachable_states = state_to_choice_to_bitvector(state_to_choice, dt_colored_mdp_factory, model_info)
    return new_bitvector, unreachable_states

# maybe completing the bitvector should also be randomized so that we are closer to the uniform sampling?
def complete_bitvector_for_eval(bitvector, unreachable_states, dt_colored_mdp_factory, model_info):
    completed_bitvector = stormpy.storage.BitVector(bitvector)
    for state, unreachable in enumerate(unreachable_states):
        if unreachable:
            selected_state_choice = random.randint(0, model_info["nr_choices_per_state"][state]-1)
            completed_bitvector.set(model_info["nondeterministic_choice_indices"][state] + selected_state_choice)
        elif not dt_colored_mdp_factory.state_is_relevant_bv.get(state):
            completed_bitvector.set(model_info["nondeterministic_choice_indices"][state] + 0) # take the first choice for irrelevant states, should not matter which one we take as they do not influence the behavior of the system

    return completed_bitvector



def mcmc_base(shed_bitvector, model_info, dt_colored_mdp_factory, specification, step_count=10000, burn_in=None, sample_steps=None, seed=None, solution_cache=None):

    shed_bitvector, unreachable_states = remove_unreachable_choices_from_bitvector(shed_bitvector, dt_colored_mdp_factory, model_info)

    if burn_in is None:
        all_sat_policies = [shed_bitvector]
        unreachable_states_list = [unreachable_states]
    else:
        all_sat_policies = []
        unreachable_states_list = []
    current_policy = shed_bitvector
    current_unreachable_states = unreachable_states

    if solution_cache is None:
        solution_cache_sat = set()
        solution_cache_unsat = set()
    else:
        solution_cache_sat = solution_cache["sat"]
        solution_cache_unsat = solution_cache["unsat"]

    cache_hits = 0

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for current_step in range(step_count):

        # sample new policy, to keep the transformation of the policy uniform we remove all unreachable states from the sampling
        selected_state = random.randint(0, model_info["nr_states"]-1)
        while current_unreachable_states[selected_state] or not dt_colored_mdp_factory.state_is_relevant_bv.get(selected_state):
            selected_state = random.randint(0, model_info["nr_states"]-1)
        selected_state_choice = random.randint(0, model_info["nr_choices_per_state"][selected_state]-1)

        completed_bitvector = complete_bitvector_for_eval(current_policy, current_unreachable_states, dt_colored_mdp_factory, model_info)

        new_bitvector = stormpy.storage.BitVector(completed_bitvector)
        for choice in range(model_info["nr_choices_per_state"][selected_state]):
            new_bitvector.set(model_info["nondeterministic_choice_indices"][selected_state] + choice, False)
        new_bitvector.set(model_info["nondeterministic_choice_indices"][selected_state] + selected_state_choice)

        new_bitvector_reachable, new_unreachable_states = remove_unreachable_choices_from_bitvector(new_bitvector, dt_colored_mdp_factory, model_info)
        cached_sat = False

        if new_bitvector_reachable in solution_cache_sat:
            cached_sat = True
            cache_hits += 1
        elif new_bitvector_reachable in solution_cache_unsat:
            cache_hits += 1
            continue

        # check if new policy satisfies specification
        if not cached_sat:
            submdp_new = dt_colored_mdp_factory.build_from_choice_mask(new_bitvector)
            mc_result_new = submdp_new.model_check_property(specification.all_properties()[0])

        # new_value = mc_result_new.value
        # print(new_value)
        # print(mc_result_new.sat)

        if cached_sat or mc_result_new.sat:
            if (burn_in is None or current_step >= burn_in) and (sample_steps is None or current_step % sample_steps == 0):
                if new_bitvector_reachable not in all_sat_policies:
                    all_sat_policies.append(new_bitvector_reachable)
                    unreachable_states_list.append(new_unreachable_states)
            current_policy = new_bitvector_reachable
            current_unreachable_states = new_unreachable_states
            if not cached_sat:
                solution_cache_sat.add(new_bitvector_reachable)
        else:
            solution_cache_unsat.add(new_bitvector_reachable)
        
    print(f"cache hit percentage: {(cache_hits/step_count*100 if step_count!=0 else 0.0):.2f}%")
    return list(zip(all_sat_policies, unreachable_states_list)), (current_policy, current_unreachable_states)


# I had to test this, obviously this does not work at all
def rejection_sampling(model_info, dt_colored_mdp_factory, specification, step_count=10000, seed=None):

    all_sat_policies = []
    unreachable_states_list = []

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    for _ in range(step_count):
        bitvector = stormpy.storage.BitVector(model_info["nr_choices"])
        for state in range(model_info["nr_states"]):
            selected_state_choice = random.randint(0, model_info["nr_choices_per_state"][state]-1)
            bitvector.set(model_info["nondeterministic_choice_indices"][state] + selected_state_choice)

        submdp = dt_colored_mdp_factory.build_from_choice_mask(bitvector)
        mc_result = submdp.model_check_property(specification.all_properties()[0])

        if mc_result.sat:
            bitvector_reachable, unreachable_states = remove_unreachable_choices_from_bitvector(bitvector, dt_colored_mdp_factory, model_info)
            if bitvector_reachable not in all_sat_policies:
                all_sat_policies.append(bitvector_reachable)
                unreachable_states_list.append(unreachable_states)

    return list(zip(all_sat_policies, unreachable_states_list)), None



@click.command()
@click.argument('project', type=click.Path(exists=True))
@click.option("--sketch", default="sketch.templ", show_default=True,
    help="name of the sketch file in the project")
@click.option("--props", default="sketch.props", show_default=True,
    help="name of the properties file in the project")
@click.option("--relative-eps", type=float, default=None, show_default=True, help="relative epsilon threhshold computed from random policy")
@click.option("--seed", type=int, default=None, show_default=True, help="random seed for policy sampling")
@click.option("--steps", type=int, default=10000, show_default=True, help="number of MCMC steps")
@click.option("--burn-in", type=int, default=None, show_default=True, help="number of burn in steps for MCMC sampling")
@click.option("--sample-steps", type=int, default=None, show_default=True, help="interval for collecting samples during MCMC sampling, e.g. if set to 10 then every 10th step after burn in will be collected as a sample")
@click.option("--output", type=click.Path(), default=None, show_default=True, help="file to write the sampled policies to json")
@click.option("--append-stats", is_flag=True, default=False, show_default=True, help="whether to append sampling and learning stats to results/sampling_stats.csv")
def main(project, sketch, props, relative_eps, seed, steps, burn_in, sample_steps, output, append_stats):
    sketch_path = os.path.join(project, sketch)
    props_path = os.path.join(project, props)

    project_name = os.path.basename(project)

    if append_stats:
        stats_file = 'results/sampling_stats.csv'
        if not os.path.exists(stats_file):
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, 'w') as f:
                f.write("model,sampling_steps,eps,sampling_time,dt_learning_time,sampled_policies,initial_tree_size,smallest_tree_size\n")
    
    sketch_path = os.path.join(project, sketch)
    properties_path = os.path.join(project, props)
    dt_colored_mdp_factory = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)

    underlying_mdp = dt_colored_mdp_factory.quotient_mdp
    specification = dt_colored_mdp_factory.specification

    if len(specification.constraints) == 0:
        assert False, "currently only specifications with constraints are supported for optimality checking"
    else:
        optimality_specification = get_optimality_specification(specification)

    all_choices = stormpy.storage.BitVector(underlying_mdp.nr_choices, True)
    full_mdp = dt_colored_mdp_factory.build_from_choice_mask(all_choices)
    full_mc_result = full_mdp.model_check_property(optimality_specification.all_properties()[0])
    opt_result_value = full_mc_result.value

    scheduler = full_mc_result.result.scheduler

    random_choices = dt_colored_mdp_factory.get_random_choices()
    submdp_random = dt_colored_mdp_factory.build_from_choice_mask(random_choices)
    mc_result_random = submdp_random.model_check_property(optimality_specification.all_properties()[0])
    random_result_value = mc_result_random.value

    print("optimal value:", opt_result_value)
    print("random value:", random_result_value)
    
    if relative_eps is not None:
        opt_random_diff = opt_result_value - random_result_value
        eps_optimum_threshold = opt_result_value - relative_eps * opt_random_diff

        specification.constraints[0].threshold = eps_optimum_threshold
        specification.constraints[0].property.raw_formula.set_bound(specification.constraints[0].formula.comparison_type, stormpy.ExpressionManager().create_rational(stormpy.Rational(eps_optimum_threshold)))
    else:
        specification.constraints[0].threshold = opt_result_value
        specification.constraints[0].property.raw_formula.set_bound(specification.constraints[0].formula.comparison_type, stormpy.ExpressionManager().create_rational(stormpy.Rational(opt_result_value)))

    # model info important for working with bitvectors
    model_info = {
        "nr_states": underlying_mdp.nr_states,
        "nr_choices": underlying_mdp.nr_choices,
        "nondeterministic_choice_indices": underlying_mdp.nondeterministic_choice_indices,
        "nr_choices_per_state": []
    }

    model_info["nr_choices_per_state"] = [model_info["nondeterministic_choice_indices"][i] - model_info["nondeterministic_choice_indices"][i-1] for i in range(1, len(model_info["nondeterministic_choice_indices"]))]


    shed_bitvector = get_bitvector_from_scheduler(scheduler, model_info)

    sampling_start_time = time.time()
    all_samples, last_sample = mcmc_base(shed_bitvector, model_info, dt_colored_mdp_factory, specification, step_count=steps, burn_in=burn_in, sample_steps=sample_steps, seed=seed)
    # all_samples, last_sample = rejection_sampling(model_info, dt_colored_mdp_factory, specification, step_count=steps, seed=seed)
    sampling_end_time = time.time()
    print(f"sampling took {sampling_end_time - sampling_start_time:.2f} seconds")

    print(f"number of policies satisfying specification found: {len(all_samples)}")

    # additional_atomic_predicates = get_atomic_predicate_evals(dt_colored_mdp_factory)
    additional_atomic_predicates = {}

    output_dict = {"X" : get_mdp_features_list(dt_colored_mdp_factory, additional_atomic_predicates), "Y" : [sample_to_list(sample, dt_colored_mdp_factory, model_info) for sample in all_samples]}
    if output is not None:
        with open(output, "w") as f:
            json.dump(output_dict, f, indent=4)

    exit()


    # print(output_dict['X'])
    # exit()
    # for x in output_dict["Y"]:
    #     print(x)

    # state_feature_to_considered_actions = {tuple(x): set() for x in output_dict["X"]}
    # for i in range(len(output_dict["Y"])):
    #     for j in range(len(output_dict["Y"][i])):
    #         if output_dict["Y"][i][j] != -1:
    #             state_feature_to_considered_actions[tuple(output_dict["X"][j])].add(output_dict["Y"][i][j])
    # print(state_feature_to_considered_actions)
    # product = 1
    # for action_set in state_feature_to_considered_actions.values():
    #     product *= len(action_set)
    # print(product)
    # exit()


    # clf = svm.SVC(kernel="linear")
    # clf = clf.fit(output_dict["X"], output_dict["Y"][0])

    filter_unreachable_class = True
    smallest_tree = None
    smallest_tree_nodes = None

    initial_tree = None
    initial_tree_nodes = None

    learning_start_time = time.time()

    used_predicate_indices = set()
    
    for i in range(len(output_dict["Y"])):
        clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=None, random_state=0) # if random_state is None (default) then scikit does not have to be deterministic
        # Filter out points with class -1
        if filter_unreachable_class:
            X = [x for j, x in enumerate(output_dict["X"]) if output_dict["Y"][i][j] != -1]
            Y = [y for y in output_dict["Y"][i] if y != -1]
        else:
            X = output_dict["X"]
            Y = output_dict["Y"][i]
        
        clf = clf.fit(X, Y)
        num_nodes = clf.tree_.node_count - clf.tree_.n_leaves
        # print(f"Tree depth: {clf.get_depth()}, Number of nodes: {num_nodes}")

        for j in range(clf.tree_.node_count):
            if clf.tree_.children_left[j] != -1: # -2 means it's a leaf node
                used_predicate_indices.add(clf.tree_.feature[j])
        
        if smallest_tree_nodes is None or num_nodes < smallest_tree_nodes:
            smallest_tree_nodes = num_nodes
            smallest_tree = clf
            smallest_tree_policy = output_dict["Y"][i]

        if i == 0:
            initial_tree = clf
            initial_tree_nodes = num_nodes

    learning_end_time = time.time()
    print(f"learning took {learning_end_time - learning_start_time:.2f} seconds")

    print(f"Number of used predicates: {len(used_predicate_indices)} out of {len(output_dict['X'][0])}")

    # print("Smallest tree policy:", smallest_tree_policy)
    print(f"Initial tree has depth {initial_tree.get_depth()} and {initial_tree_nodes} nodes")
    print(f"Smallest tree has depth {smallest_tree.get_depth()} and {smallest_tree_nodes} nodes")
    # tree.plot_tree(smallest_tree)
    # plt.savefig("tree_output.png", dpi=300, bbox_inches='tight')

    if append_stats:
        with open(stats_file, 'a') as f:
            f.write(f"{project_name},{steps},{relative_eps},{sampling_end_time - sampling_start_time:.2f},{learning_end_time - learning_start_time:.2f},{len(all_samples)},{initial_tree_nodes},{smallest_tree_nodes}\n")

if __name__ == "__main__":
    main()