

import paynt
import stormpy
import payntbind

import paynt.parser.sketch
import paynt.verification.property

import pysat

import os
import operator
import itertools
import time

import click



# CONSTANTS
GATE_TYPES = ['AND', 'OR'] # XOR can be added




def get_state_to_choices(dt_colored_mdp_factory, scheduler):
    state_to_choice = payntbind.synthesis.schedulerToStateToGlobalChoice(scheduler, dt_colored_mdp_factory.quotient_mdp, [x for x in range(dt_colored_mdp_factory.quotient_mdp.nr_choices)])
    state_to_choice = dt_colored_mdp_factory.discard_unreachable_choices(state_to_choice)
    return state_to_choice

def get_state_classes_from_scheduler(dt_colored_mdp_factory, scheduler):
    state_classes = {action: stormpy.BitVector(dt_colored_mdp_factory.quotient_mdp.nr_states) for action in dt_colored_mdp_factory.action_labels}
    
    state_to_choices = get_state_to_choices(dt_colored_mdp_factory, scheduler)

    for state, choice in enumerate(state_to_choices):

        if not dt_colored_mdp_factory.state_is_relevant_bv.get(state):
            continue
        
        action_index = dt_colored_mdp_factory.choice_to_action[choice]
        action_label = dt_colored_mdp_factory.action_labels[action_index]

        state_classes[action_label].set(state)

    relevant_actions = []
    for action, state_class in state_classes.items():
        if state_class.number_of_set_bits() > 0:
            relevant_actions.append(action)

    state_classes = {action: state_class for action, state_class in state_classes.items() if action in relevant_actions}
            
    return state_classes


def model_to_predicate_string(model, varmap, atomic_predicates, predicate_depth):
    """
    Converts the SAT model and varmap into a string representing the learned predicate.
    Handles both depth 0 (single predicate or negation) and tree (depth > 0) cases.
    """
    model_set = set(model)
    if predicate_depth == 0:
        # Only one selector is active
        selector_vars = varmap['selector_vars']
        for i, (sel_pos, sel_neg) in enumerate(selector_vars):
            if sel_pos in model_set:
                return atomic_predicates[i]
            if sel_neg in model_set:
                return f"!{atomic_predicates[i]}"
        return "<unsat>"

    # For tree case
    num_leaves = 2 ** predicate_depth
    num_gates = num_leaves - 1
    gate_types = GATE_TYPES

    # Get leaf predicates
    leaf_selector_vars = varmap['leaf_selector_vars']
    leaf_predicates = []
    for l in range(num_leaves):
        selectors = leaf_selector_vars[l]
        pred_str = None
        for i, (sel_pos, sel_neg) in enumerate(selectors):
            if sel_pos in model_set:
                pred_str = atomic_predicates[i]
                break
            if sel_neg in model_set:
                pred_str = f"!{atomic_predicates[i]}"
                break
        if pred_str is None:
            pred_str = "<unsat>"
        leaf_predicates.append(pred_str)

    # Recursively build tree
    def build_tree(node):
        if node >= num_gates:
            return leaf_predicates[node - num_gates]
        # Find gate type
        gate_type_vars = varmap['gate_type_vars'][node]
        for idx, tvar in enumerate(gate_type_vars):
            if tvar in model_set:
                gate_type = gate_types[idx]
                break
        else:
            gate_type = "<unsat>"
        left = build_tree(2 * node + 1)
        right = build_tree(2 * node + 2)
        return f"({left} {gate_type} {right})"

    return build_tree(0)


def solve_sat_encoding(cnf):
    """
    Solves the given CNF using a PySAT solver.
    Returns the model (list of variable assignments) if satisfiable, else None.
    """
    from pysat.solvers import Minisat22
    with Minisat22(bootstrap_with=cnf.clauses) as solver:
        if solver.solve():
            return solver.get_model()
        else:
            return None


def create_sat_encoding(state_class, atomic_predicates_evals, predicate_tree_depth):
    """
    Create a SAT encoding for finding a predicate (with a logic formula as a full binary tree of given depth) that splits the state space according to state_class.
    - state_class: BitVector, True for states in the class, False otherwise
    - atomic_predicates_evals: dict[str, BitVector], mapping predicate string to BitVector of states where it holds
    - predicate_tree_depth: int, depth of the binary tree (0 = just atomic predicates)
    Returns: (cnf, varmap) where cnf is a pysat.formula.CNF, varmap is a dict for variable interpretation
    """

    from pysat.formula import CNF, IDPool
    cnf = CNF()
    vpool = IDPool()
    varmap = {}

    num_states = state_class.size()
    atomic_predicates = list(atomic_predicates_evals.keys())

    # For each state, create a SAT variable representing the output of the learned predicate on that state
    # This is only used for predicate_tree_depth == 0 (single predicate case)
    state_out_vars = [vpool.id(f'out_{s}') for s in range(num_states)]
    varmap['state_out_vars'] = state_out_vars

    # For predicate_tree_depth == 0, fallback to the simple case
    if predicate_tree_depth == 0:
        # --- Predicate size 0: Only allow a single atomic predicate (or its negation) as the split ---

        # For each atomic predicate (and its negation), create a selector variable
        selector_vars = []
        for i, pred in enumerate(atomic_predicates):
            sel_pos = vpool.id(f'sel_{pred}_pos')  # Select positive (non-negated) version
            sel_neg = vpool.id(f'sel_{pred}_neg')  # Select negated version
            selector_vars.append((sel_pos, sel_neg))
        varmap['selector_vars'] = selector_vars

        # Only one selector can be active (either pos or neg for one predicate)
        all_selectors = [v for pair in selector_vars for v in pair]
        cnf.append(all_selectors)  # At least one selector must be true
        for i in range(len(all_selectors)):
            for j in range(i+1, len(all_selectors)):
                cnf.append([-all_selectors[i], -all_selectors[j]])  # At most one selector is true

        # For each state, encode the output according to the selected predicate/negation
        for s in range(num_states):
            out_var = state_out_vars[s]
            for i, pred in enumerate(atomic_predicates):
                sel_pos, sel_neg = selector_vars[i]
                holds = atomic_predicates_evals[pred].get(s)
                # If sel_pos is chosen, out_var == holds
                # (sel_pos => (out_var == holds))
                if holds:
                    cnf.append([-sel_pos, out_var])
                else:
                    cnf.append([-sel_pos, -out_var])
                # If sel_neg is chosen, out_var == not holds
                # (sel_neg => (out_var == not holds))
                if not holds:
                    cnf.append([-sel_neg, out_var])
                else:
                    cnf.append([-sel_neg, -out_var])

        # For each state, enforce that out_var matches state_class
        for s in range(num_states):
            out_var = state_out_vars[s]
            if state_class.get(s):
                cnf.append([out_var])  # Output must be true for states in the class
            else:
                cnf.append([-out_var])  # Output must be false for other states

        return cnf, varmap


    # --- Predicate tree depth > 0: Build a logic circuit (full binary tree) of gates and leaves ---

    # For a full binary tree of depth d:
    # Number of leaves = 2**d
    # Number of gates (internal nodes) = 2**d - 1
    # Total nodes = 2**(d+1) - 1
    num_leaves = 2 ** predicate_tree_depth
    num_gates = num_leaves - 1
    total_nodes = num_leaves + num_gates

    # Node indices: gates 0..num_gates-1, leaves num_gates..total_nodes-1
    # For each node and state, create output variable
    node_out_vars = [[vpool.id(f'node_{n}_out_{s}') for s in range(num_states)] for n in range(total_nodes)]
    varmap['node_out_vars'] = node_out_vars

    # For each gate node, create type selector variables (AND/OR/XOR)
    gate_types = GATE_TYPES
    gate_type_vars = []
    for g in range(num_gates):
        # Each gate can be one of AND, OR, XOR
        type_vars = [vpool.id(f'gate_{g}_type_{t}') for t in gate_types]
        gate_type_vars.append(type_vars)
        # Exactly one type per gate
        cnf.append(type_vars)  # At least one type
        for i in range(len(type_vars)):
            for j in range(i+1, len(type_vars)):
                cnf.append([-type_vars[i], -type_vars[j]])  # At most one type
    varmap['gate_type_vars'] = gate_type_vars

    # For each leaf, create selector variables for atomic predicates (and their negations)
    leaf_selector_vars = []
    for l in range(num_leaves):
        selectors = []
        for pred in atomic_predicates:
            sel_pos = vpool.id(f'leaf_{l}_sel_{pred}_pos')  # Select positive version
            sel_neg = vpool.id(f'leaf_{l}_sel_{pred}_neg')  # Select negated version
            selectors.append((sel_pos, sel_neg))
        leaf_selector_vars.append(selectors)
        # Exactly one selector per leaf
        all_selectors = [v for pair in selectors for v in pair]
        cnf.append(all_selectors)  # At least one selector
        for i in range(len(all_selectors)):
            for j in range(i+1, len(all_selectors)):
                cnf.append([-all_selectors[i], -all_selectors[j]])  # At most one selector
    varmap['leaf_selector_vars'] = leaf_selector_vars

    # For each leaf and state, encode output according to selected predicate/negation
    for l in range(num_leaves):
        leaf_node = num_gates + l
        for s in range(num_states):
            out_var = node_out_vars[leaf_node][s]
            for i, pred in enumerate(atomic_predicates):
                sel_pos, sel_neg = leaf_selector_vars[l][i]
                holds = atomic_predicates_evals[pred].get(s)
                # If sel_pos is chosen, out_var == holds
                if holds:
                    cnf.append([-sel_pos, out_var])
                else:
                    cnf.append([-sel_pos, -out_var])
                # If sel_neg is chosen, out_var == not holds
                if not holds:
                    cnf.append([-sel_neg, out_var])
                else:
                    cnf.append([-sel_neg, -out_var])

    # For each gate, define its children (full binary tree):
    # For gate at index g (in 0..num_gates-1), its left child is at 2g+1, right child at 2g+2
    for g in range(num_gates):
        gate_node = g
        left_child = 2 * g + 1
        right_child = 2 * g + 2
        for s in range(num_states):
            out_var = node_out_vars[gate_node][s]
            # Check if left_child and right_child are gates or leaves
            if left_child < num_gates:
                left_var = node_out_vars[left_child][s]
            else:
                left_var = node_out_vars[num_gates + (left_child - num_gates)][s]
            if right_child < num_gates:
                right_var = node_out_vars[right_child][s]
            else:
                right_var = node_out_vars[num_gates + (right_child - num_gates)][s]
            # For each type, encode output
            # t_and, t_or, t_xor = gate_type_vars[g]
            t_and, t_or = gate_type_vars[g]
            # AND: out_var <-> (left_var & right_var)
            cnf.append([-t_and, -left_var, -right_var, out_var])  # If t_and and both children false, out_var false
            cnf.append([-t_and, left_var, -out_var])              # If t_and and left true, out_var must match right
            cnf.append([-t_and, right_var, -out_var])             # If t_and and right true, out_var must match left
            # OR: out_var <-> (left_var | right_var)
            cnf.append([-t_or, left_var, out_var])                # If t_or and left true, out_var true
            cnf.append([-t_or, right_var, out_var])               # If t_or and right true, out_var true
            cnf.append([-t_or, -left_var, -right_var, -out_var])  # If t_or and both false, out_var false
            # XOR: out_var <-> (left_var ^ right_var)
            # cnf.append([-t_xor, -left_var, -right_var, -out_var]) # If t_xor and both false, out_var false
            # cnf.append([-t_xor, left_var, right_var, -out_var])   # If t_xor and both true, out_var false
            # cnf.append([-t_xor, left_var, -right_var, out_var])   # If t_xor and left true, right false, out_var true
            # cnf.append([-t_xor, -left_var, right_var, out_var])   # If t_xor and left false, right true, out_var true

    # The output of the root node (gate 0) must match state_class
    root_node = 0
    for s in range(num_states):
        out_var = node_out_vars[root_node][s]
        if state_class.get(s):
            cnf.append([out_var])  # Output must be true for states in the class
        else:
            cnf.append([-out_var]) # Output must be false for other states

    return cnf, varmap



def clean_useless_predicates(predicate_to_valuation):

    # remove predicates that are true for all states or false for all states
    cleaned_predicate_to_valuation = {}
    for predicate, valuation in predicate_to_valuation.items():
        if valuation.number_of_set_bits() > 0 and valuation.number_of_set_bits() < valuation.size():
            cleaned_predicate_to_valuation[predicate] = valuation

    return cleaned_predicate_to_valuation


def get_atomic_predicate_evals(dt_colored_mdp_factory):

    predicate_to_valuation = {}

    state_varriables = dt_colored_mdp_factory.variables
    state_valuations = dt_colored_mdp_factory.relevant_state_valuations

    constant_comparison_operators = {
        '==': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        # '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge
    }

    for var_id, variable in enumerate(state_varriables):
        domain_list = list(variable.domain)[:-1]
        for constant in domain_list:
            for op_str, op_func in constant_comparison_operators.items():
                if f"{variable.name}{op_str}{constant}" not in predicate_to_valuation:
                    predicate_to_valuation[f"{variable.name}{op_str}{constant}"] = stormpy.BitVector(dt_colored_mdp_factory.quotient_mdp.nr_states)
                valuation = predicate_to_valuation[f"{variable.name}{op_str}{constant}"]
                for state, state_valuation in enumerate(state_valuations):
                    if op_func(state_valuation[var_id], constant):
                        valuation.set(state)

    comparison_opertators = {
        '==': operator.eq,
        '!=': operator.ne,
        '<': operator.lt,
        '<=': operator.le,
        '>': operator.gt,
        '>=': operator.ge
    }

    vars_indices = list(range(len(state_varriables)))

    for var1_id, var2_id in itertools.combinations(vars_indices, 2):
        var1 = state_varriables[var1_id]
        var2 = state_varriables[var2_id]
        for op_str, op_func in comparison_opertators.items():
            if f"{var1.name}{op_str}{var2.name}" not in predicate_to_valuation:
                predicate_to_valuation[f"{var1.name}{op_str}{var2.name}"] = stormpy.BitVector(dt_colored_mdp_factory.quotient_mdp.nr_states)
            valuation = predicate_to_valuation[f"{var1.name}{op_str}{var2.name}"]
            for state, state_valuation in enumerate(state_valuations):
                if op_func(state_valuation[var1_id], state_valuation[var2_id]):
                    valuation.set(state)

    predicate_to_valuation = clean_useless_predicates(predicate_to_valuation)

    return predicate_to_valuation


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


@click.command()
@click.argument('project', type=click.Path(exists=True))
@click.option("--sketch", default="sketch.templ", show_default=True,
    help="name of the sketch file in the project")
@click.option("--props", default="sketch.props", show_default=True,
    help="name of the properties file in the project")
def main(project, sketch, props):
    sketch_path = os.path.join(project, sketch)
    props_path = os.path.join(project, props)
    
    sketch_path = os.path.join(project, sketch)
    properties_path = os.path.join(project, props)
    dt_colored_mdp_factory = paynt.parser.sketch.Sketch.load_sketch(sketch_path, properties_path)

    underlying_mdp = dt_colored_mdp_factory.quotient_mdp
    specification = dt_colored_mdp_factory.specification

    if len(specification.constraints) == 0:
        optimality_specification = specification
    else:
        optimality_specification = get_optimality_specification(specification)

    scheduler = get_scheduler(underlying_mdp, optimality_specification.all_properties()[0])

    # print(dir(scheduler))
    # print(dt_colored_mdp_factory.state_is_relevant)
    # print(len(dt_colored_mdp_factory.relevant_state_valuations))
    # print(underlying_mdp)

    state_classes = get_state_classes_from_scheduler(dt_colored_mdp_factory, scheduler)

    # for action, state_class in state_classes.items():
    #     print(f"Action: {action}")
    #     print(f"State class (number of states: {state_class.number_of_set_bits()})")


    atomic_predicates_evals = get_atomic_predicate_evals(dt_colored_mdp_factory)

    # print(len(atomic_predicates_evals), "atomic predicates extracted")


    # exit()

    
    for action in state_classes.keys():
        for predicate_depth in range(4):
            t0 = time.perf_counter()
            cnf, varmap = create_sat_encoding(state_classes[action], atomic_predicates_evals, predicate_depth)
            t1 = time.perf_counter()
            print(f"Encoding construction time (size {predicate_depth} for action {action}): {t1 - t0:.4f} seconds")

            t2 = time.perf_counter()
            model = solve_sat_encoding(cnf)
            t3 = time.perf_counter()
            print(f"Solve time (size {predicate_depth} for action {action}): {t3 - t2:.4f} seconds")
            if model is not None:
                print(f"SAT: Found a predicate of size {predicate_depth} for action {action}")
                predicate_str = model_to_predicate_string(model, varmap, list(atomic_predicates_evals.keys()), predicate_depth)
                print(f"Learned predicate: {predicate_str}")
                break
            else:
                print(f"UNSAT: No predicate of size {predicate_depth} found for action {action}")
        print()
    exit()
        
        
    




if __name__ == "__main__":
    main()
