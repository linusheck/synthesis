import stormpy

import math
import itertools
from collections import OrderedDict

from .statistic import Statistic
from .models import MarkovChain, DTMC, MDP
from .quotient import JaniQuotientContainer, POMDPQuotientContainer

from ..profiler import Timer,Profiler

from ..sketch.holes import Holes,DesignSpace

import logging
logger = logging.getLogger(__name__)

from stormpy.synthesis import dtmc_from_mdp


class Synthesizer:

    def __init__(self, sketch, quotient_container = None):
        MarkovChain.initialize(sketch.properties, sketch.optimality_property)

        self.sketch = sketch
        if quotient_container is not None:
            self.quotient_container = quotient_container
        else:
            if sketch.is_pomdp:
                self.quotient_container = POMDPQuotientContainer(sketch)
                # self.quotient_container.unfoldFullMemory(memory_size = 1)
                self.quotient_container.unfoldPartialMemory()
            else:
                self.quotient_container = JaniQuotientContainer(sketch)

        self.stat = Statistic(sketch, self.method_name)
        Profiler.initialize()

    @property
    def method_name(self):
        return "1-by-1"
    
    @property
    def has_optimality(self):
        return self.sketch.optimality_property is not None

    def print_stats(self, short_summary = False):
        print(self.stat.get_summary(short_summary))

    def run(self):
        assert not sketch.prism.model_type == stormpy.storage.PrismModelType.POMDP, "1-by-1 method does not support POMDP"
        self.stat.start()
        satisfying_assignment = None
        for hole_combination in self.design_space.all_hole_combinations():
            assignment = self.design_space.construct_assignment(hole_combination)
            dtmc = DTMC(self.sketch, assignment)
            self.stat.iteration_dtmc(dtmc.states)
            constraints_sat = dtmc.check_properties(self.sketch.properties)
            self.stat.pruned(1)
            if not constraints_sat:
                continue
            if not self.has_optimality:
                satisfying_assignment = assignment
                break
            _,improved = dtmc.check_optimality(self.sketch.optimality_property)
            if improved:
                satisfying_assignment = assignment

        self.stat.finished(satisfying_assignment)


class SynthesizerAR(Synthesizer):
    
    @property
    def method_name(self):
        return "AR"

    def synthesize(self, family):

        self.stat.family(family)
        self.stat.start()

        # initiate AR loop
        satisfying_assignment = None
        families = [family]
        while families:
            family = families.pop(-1)
            # logger.debug("analyzing family {}".format(family))
            mdp = self.quotient_container.build(family)
            self.stat.iteration_mdp(mdp.states)
            feasible,undecided_properties,undecided_results = mdp.check_properties(family.properties)
            properties = undecided_properties

            if feasible == True and not self.has_optimality:
                # logger.debug("AR: found feasible family")
                satisfying_assignment = family.pick_any()
                break

            can_improve = feasible is None
            if feasible == True and self.has_optimality:
                # check optimality
                results,optimum,improving_assignment,can_improve = mdp.check_optimality(self.sketch.optimality_property)

                if optimum is not None:
                    self.sketch.optimality_property.update_optimum(optimum)
                    satisfying_assignment = improving_assignment
                if can_improve:
                    undecided_results.append(results)

            if not can_improve:
                self.stat.pruned(family.size)
                continue

            # split family wrt first undecided result
            subfamilies = self.quotient_container.prepare_split(mdp, undecided_results[-1], properties)
            for subfamily in subfamilies:
                families.append(subfamily)

        self.stat.finished(satisfying_assignment)

        return satisfying_assignment

    def run(self):
        self.synthesize(self.sketch.design_space)


class SynthesizerCEGIS(Synthesizer):

    @property
    def method_name(self):
        return "CEGIS"

    def run(self):
        self.stat.start()

        satisfying_assignment = None
        self.sketch.design_space.z3_initialize()
        self.sketch.design_space.z3_encode()

        assignment = self.sketch.design_space.pick_assignment()
        while assignment is not None:
            # logger.debug("analyzing assignment {}".format(assignment))
            # build DTMC
            dtmc = self.quotient_container.build(assignment)
            self.stat.iteration_dtmc(dtmc.states)

            # model check all properties
            sat,unsat_properties = dtmc.check_properties_all(self.sketch.properties)
            if self.has_optimality:
                optimum,improves = dtmc.check_optimality(self.sketch.optimality_property)
                unsat_properties.append(self.sketch.optimality_property)

            # analyze model checking results
            if sat:
                if not self.has_optimality:
                    satisfying_assignment = assignment
                    break
                if improves:
                    self.sketch.optimality_property.update_optimum(optimum)
                    satisfying_assignment = assignment

            # construct a conflict to each unsatisfiable property
            conflicts = []
            for prop in unsat_properties:
                conflict = [hole_index for hole_index,_ in enumerate(assignment)]
                conflicts.append(conflict)

            # use conflicts to exclude the generalizations of this assignment
            for conflict in conflicts:
                self.sketch.design_space.exclude_assignment(assignment, conflict)
                self.stat.pruned(1)

            # construct next assignment
            assignment = self.sketch.design_space.pick_assignment()

        self.stat.finished(satisfying_assignment)




class SynthesizerPOMDP():

    def __init__(self, sketch):
        assert sketch.is_pomdp

        MarkovChain.initialize(sketch.properties, sketch.optimality_property)
        Profiler.initialize()

        self.sketch = sketch
        self.quotient_container = POMDPQuotientContainer(sketch)

        self.total_iters = 0

    def synthesize(self, family = None):
        Profiler.start("synthesis")
        if family is None:
            family = self.sketch.design_space
        synthesizer = SynthesizerAR(self.sketch, self.quotient_container)
        synthesizer.synthesize(family)
        synthesizer.print_stats(short_summary = True)
        Profiler.stop()
        print("current optimal solution: ", self.sketch.optimality_property.optimum)
        print("", flush=True)
        self.total_iters += synthesizer.stat.iterations_mdp

    def choose_consistent(self, full_space, restriction):
        design_space = full_space.copy()
        for obs,choices in restriction.items():
            hole_index = self.quotient_container.pomdp_manager.action_holes[obs][0]
            design_space = design_space.assume_suboptions(hole_index, choices)
        # print("full design space", self.sketch.design_space)                
        # print("restricted design space: ", design_space)
        print("reduced design space from {} to {}".format(full_space.size, design_space.size))
        return design_space

    def choose_consistent_and_break_symmetry(self, full_space, observation_choices):
        design_space = full_space.copy()
        for obs,choices in observation_choices.items():
            hole_indices = self.quotient_container.pomdp_manager.action_holes[obs]
            if len(hole_indices) == 1:
                if len(choices) == 1:
                    # consistent observation
                    hole_index = hole_indices[0]
                    design_space = design_space.assume_suboptions(hole_index, choices)
            else:
                # have multiple holes for this observation
                for index,hole_index in enumerate(hole_indices):
                    options = full_space[hole_index].options.copy()
                    options.remove(choices[index])
                    design_space = design_space.assume_suboptions(hole_index, options)
        # print("full design space", self.sketch.design_space)                
        # print("restricted design space: ", design_space)
        print("reduced design space from {} to {}".format(full_space.size, design_space.size))
        return design_space

    def strategy_1(self):
        # self.sketch.optimality_property.optimum = 0.75
        self.quotient_container.pomdp_manager.set_memory_size(1)
        self.quotient_container.unfoldPartialMemory()
        self.synthesize()
        Profiler.print()

    def strategy_2(self):
        # analyze POMDP
        assert len(self.sketch.properties) == 0
        self.quotient_container.unfoldPartialMemory()

        mdp = self.quotient_container.build(self.sketch.design_space)
        bounds = mdp.model_check_property(self.sketch.optimality_property)
        selection = self.quotient_container.scheduler_selection(mdp, bounds.scheduler)
        print("scheduler selected: ", selection)

        # associate observations with respective choices
        observation_choices = {}
        pm = self.quotient_container.pomdp_manager
        for obs in range(self.quotient_container.pomdp.nr_observations):
            hole_indices = pm.action_holes[obs]
            if len(hole_indices) == 0:
                continue
            hole_index = hole_indices[0]
            assert len(selection[hole_index]) >= 1
            observation_choices[obs] = selection[hole_index]
        print("observation choices: ", observation_choices)

        # map consistent observations to corresponding choices
        consistent_restriction = {obs:choices for obs,choices in observation_choices.items() if len(choices) == 1}
        print("consistent restriction" , consistent_restriction)

        # synthesize optimal solution for k=1 (full, restricted)
        # restrict options of consistent holes to a scheduler selection
        design_space = self.choose_consistent(self.sketch.design_space, consistent_restriction)
        self.synthesize(design_space)

        # synthesize optimal solution for k=2 (partial, restricted)
        # gradually inject memory to inconsistent observations
        for obs in range(self.quotient_container.pomdp.nr_observations):
            if pm.action_holes[obs] and obs not in consistent_restriction:
                print("injecting memory to observation ", obs)
                print("scheduler chose actions ", observation_choices[obs])
                self.quotient_container.pomdp_manager.inject_memory(obs)
                self.quotient_container.unfoldPartialMemory()
                # design_space = self.choose_consistent(self.sketch.design_space, consistent_restriction)
                design_space = self.choose_consistent_and_break_symmetry(self.sketch.design_space, observation_choices)
                self.synthesize(design_space)

        # synthesize optimal solution for k=2 (partial, unrestricted)
        # print("synthesizing solution for k=2 (partial, unrestricted)")
        # self.synthesize()

        # total stats
        print("total iters: ", self.total_iters)
        Profiler.print()

    def strategy_3(self):
        
        assert len(self.sketch.properties) == 0
        pomdp = self.quotient_container.pomdp
        pm = self.quotient_container.pomdp_manager

        obs_memory_size = [1] * pomdp.nr_observations
        max_memory_size = 1

        for i in range(10):
            # analyze quotient MDP
            self.quotient_container.unfoldPartialMemory()
            mdp = self.quotient_container.build(self.sketch.design_space)
            result = mdp.analyze_property(self.sketch.optimality_property)
            selection = self.quotient_container.scheduler_selection(mdp, result.scheduler)
            print("scheduler selected: ", selection)

            # fix choices associated with consistent actions
            design_space = self.sketch.design_space.copy()
            for hole_index,hole in enumerate(self.sketch.design_space):
                options = selection[hole_index]
                if len(options) > 1:
                    continue
                # print("restricting hole {} to option {}".format(hole_index,options[0]))
                # design_space[hole_index] = hole.subhole(options)

            # synthesize
            print("reduced design space from {} to {}".format(self.sketch.design_space.size, design_space.size))
            self.synthesize(design_space)

            # identify observations having inconsistent action or memory choices
            obs_list = range(self.quotient_container.pomdp.nr_observations)
            inconsistent_holes = [hole_index for hole_index,options in enumerate(selection) if len(options)>1]
            inconsistent_obs = []
            for obs in obs_list:
                for hole_index in itertools.chain(pm.action_holes[obs],pm.memory_holes[obs]):
                    if hole_index in inconsistent_holes:
                        inconsistent_obs.append(obs)
                        break
            
            # inject memory into observation having inconsistent action holes
            can_add_memory = [obs for obs in obs_list if obs_memory_size[obs] < max_memory_size]
            want_add_memory = [obs for obs in can_add_memory if obs in inconsistent_obs]

            if not want_add_memory:
                # all observation are at max memory
                max_memory_size += 1
                can_add_memory = [obs for obs in obs_list if obs_memory_size[obs] < max_memory_size]
                want_add_memory = [obs for obs in can_add_memory if obs in inconsistent_obs]

            assert want_add_memory
            obs = want_add_memory[0]

            print("injecting memory into observation ", obs)

            pm.inject_memory(obs)
            obs_memory_size[obs] += 1

        exit()



        # # associate observations with respective choices
        # observation_choices = {}
        
        #     
        #     if len(hole_indices) == 0:
        #         continue
        #     hole_index = hole_indices[0]
        #     assert len(selection[hole_index]) >= 1
        #     observation_choices[obs] = selection[hole_index]
        # print("observation choices: ", observation_choices)

    def run(self):

        
        # self.strategy_1()
        self.strategy_2()
        # self.strategy_3()
        exit()

        


        # for i in range(3):
        #     print("splitter frequency: ", self.quotient_container.splitter_frequency)
        #     obs = self.quotient_container.suggest_injection()
        #     print("suggesting split at observation ", obs)

        #     self.quotient_container.pomdp_manager.inject_memory(obs)
        #     self.quotient_container.unfoldPartialMemory()
        #     self.synthesize()
        #     print("current stats: {} sec, {} iters".format(round(self.total_time,2),self.total_iters))
        





    

    












class FamilyHybrid():
    ''' Family adopted for CEGAR-CEGIS analysis. '''

    # TODO: more efficient state-hole mapping?

    _choice_to_hole_indices = {}

    def __init__(self, *args):
        super().__init__(*args)

        self._state_to_hole_indices = None  # evaluated on demand

        # dtmc corresponding to the constructed assignment
        self.dtmc = None
        self.dtmc_state_map = None

    def initialize(*args):
        Family.initialize(*args)

        # map edges of a quotient container to hole indices
        jani = Family._quotient_container.jani_program
        _edge_to_hole_indices = dict()
        for aut_index, aut in enumerate(jani.automata):
            for edge_index, edge in enumerate(aut.edges):
                if edge.color == 0:
                    continue
                index = jani.encode_automaton_and_edge_index(aut_index, edge_index)
                assignment = Family._quotient_container.edge_coloring.get_hole_assignment(edge.color)
                hole_indices = [index for index, value in enumerate(assignment) if value is not None]
                _edge_to_hole_indices[index] = hole_indices

        # map actions of a quotient MDP to hole indices
        FamilyHybrid._choice_to_hole_indices = []
        choice_origins = Family._quotient_mdp.choice_origins
        matrix = Family._quotient_mdp.transition_matrix
        for state in range(Family._quotient_mdp.nr_states):
            for choice_index in range(matrix.get_row_group_start(state), matrix.get_row_group_end(state)):
                choice_hole_indices = set()
                for index in choice_origins.get_edge_index_set(choice_index):
                    hole_indices = _edge_to_hole_indices.get(index, set())
                    choice_hole_indices.update(hole_indices)
                FamilyHybrid._choice_to_hole_indices.append(choice_hole_indices)

    def split(self):
        assert self.split_ready
        return FamilyHybrid(self, self.suboptions[0]), FamilyHybrid(self, self.suboptions[1])

    @property
    def state_to_hole_indices(self):
        '''
        Identify holes relevant to the states of the MDP and store only significant ones.
        '''
        # if someone (i.e., CEGIS) asks for state indices, the model should already be analyzed
        assert self.constructed and self.analyzed

        # lazy evaluation
        if self._state_to_hole_indices is not None:
            return self._state_to_hole_indices

        
        # logger.debug("Constructing state-holes mapping via edge-holes mapping.")

        self._state_to_hole_indices = []
        matrix = self.mdp.transition_matrix
        for state in range(self.mdp.nr_states):
            state_hole_indices = set()
            for choice_index in range(matrix.get_row_group_start(state), matrix.get_row_group_end(state)):
                state_hole_indices.update(FamilyHybrid._choice_to_hole_indices[self.choice_map[choice_index]])
            state_hole_indices = set(
                [index for index in state_hole_indices if len(self.design_space[Family.sketch.design_space.holes[index]]) > 1]
            )
            self._state_to_hole_indices.append(state_hole_indices)

        return self._state_to_hole_indices

    @property
    def state_to_hole_indices_choices(self):
        '''
        Identify holes relevant to the states of the MDP and store only significant ones.
        '''
        # if someone (i.e., CEGIS) asks for state indices, the model should already be analyzed
        assert self.constructed and self.analyzed

        # lazy evaluation
        if self._state_to_hole_indices is not None:
            return self._state_to_hole_indices

        Profiler.start("is - MDP holes (choices)")
        logger.debug("Constructing state-holes mapping via choice-holes mapping.")

        self._state_to_hole_indices = []
        matrix = self.mdp.transition_matrix
        for state in range(self.mdp.nr_states):
            state_hole_indices = set()
            for choice_index in range(matrix.get_row_group_start(state), matrix.get_row_group_end(state)):
                quotient_choice_index = self.choice_map[choice_index]
                choice_hole_indices = FamilyHybrid._choice_to_hole_indices[quotient_choice_index]
                state_hole_indices.update(choice_hole_indices)
            state_hole_indices = set(
                [index for index in state_hole_indices if len(self.options[Family.hole_list[index]]) > 1])
            self._state_to_hole_indices.append(state_hole_indices)
        Profiler.stop()
        return self._state_to_hole_indices

    def pick_member(self):
        # pick hole assignment

        self.pick_assignment()
        if self.member_assignment is not None:

            # collect edges relevant for this assignment
            indexed_assignment = Family.sketch.design_space.index_map(self.member_assignment)
            subcolors = Family._quotient_container.edge_coloring.subcolors(indexed_assignment)
            collected_edge_indices = stormpy.FlatSet(
                Family._quotient_container.color_to_edge_indices.get(0, stormpy.FlatSet())
            )
            for c in subcolors:
                collected_edge_indices.insert_set(Family._quotient_container.color_to_edge_indices.get(c))

            # construct the DTMC by exploring the quotient MDP for this subfamily
            self.dtmc, self.dtmc_state_map = stormpy.synthesis.dtmc_from_mdp(self.mdp, collected_edge_indices)
            logger.debug(f"Constructed DTMC of size {self.dtmc.nr_states}.")

            # assert absence of deadlocks or overlapping guards
            # assert self.dtmc.labeling.get_states("deadlock").number_of_set_bits() == 0
            assert self.dtmc.labeling.get_states("overlap_guards").number_of_set_bits() == 0
            assert len(self.dtmc.initial_states) == 1  # to avoid ambiguity

        # success
        return self.member_assignment

    def exclude_member(self, conflicts):
        '''
        Exclude the subfamily induced by the selected assignment and a set of conflicts.
        '''
        assert self.member_assignment is not None

        for conflict in conflicts:
            counterexample_clauses = dict()
            for var, hole in Family._solver_meta_vars.items():
                if Family._hole_indices[hole] in conflict:
                    option_index = Family._hole_option_indices[hole][self.member_assignment[hole][0]]
                    counterexample_clauses[hole] = (var == option_index)
                else:
                    all_options = [var == Family._hole_option_indices[hole][option] for option in self.options[hole]]
                    counterexample_clauses[hole] = z3.Or(all_options)
            counterexample_encoding = z3.Not(z3.And(list(counterexample_clauses.values())))
            Family._solver.add(counterexample_encoding)
        self.member_assignment = None

    def analyze_member(self, formula_index):
        assert self.dtmc is not None
        result = stormpy.model_checking(self.dtmc, Family.formulae[formula_index].formula)
        value = result.at(self.dtmc.initial_states[0])
        satisfied = Family.formulae[formula_index].satisfied(value)
        return satisfied, value

    def print_member(self):
        print("> DTMC info:")
        dtmc = self.dtmc
        tm = dtmc.transition_matrix
        for state in range(dtmc.nr_states):
            row = tm.get_row(state)
            print("> ", str(row))

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------

# ----- Adaptivity ----- #
# idea: switch between cegar/cegis, allocate more time to the more efficient method

class StageControl:
    # switching
    def __init__(self):
        self.stage_timer = Timer()
        # cegar/cegis stats
        self.stage_time_cegar, self.stage_pruned_cegar, self.stage_time_cegis, self.stage_pruned_cegis = 0, 0, 0, 0
        # multiplier to derive time allocated for cegis; =1 is fair, <1 favours cegar, >1 favours cegis
        self.cegis_allocated_time_factor = 1.0
        # start with AR
        self.stage_cegar = True
        self.cegis_allocated_time = 0

    def start(self, request_stage_cegar):
        self.stage_cegar = request_stage_cegar
        self.stage_timer.reset()
        self.stage_timer.start()

    def step(self, models_pruned):
        '''Performs a stage step, returns True if the method switch took place'''

        # record pruned models
        self.stage_pruned_cegar += models_pruned / self.models_total if self.stage_cegar else 0
        self.stage_pruned_cegis += models_pruned / self.models_total if not self.stage_cegar else 0

        # in cegis mode, allow cegis another stage step if some time remains
        if not self.stage_cegar and self.stage_timer.read() < self.cegis_allocated_time:
            return False

        # stage is finished: record time
        self.stage_timer.stop()
        current_time = self.stage_timer.read()
        if self.stage_cegar:
            # cegar stage over: allocate time for cegis and switch
            self.stage_time_cegar += current_time
            self.cegis_allocated_time = current_time * self.cegis_allocated_time_factor
            self.stage_start(request_stage_cegar=False)
            return True

        # cegis stage over
        self.stage_time_cegis += current_time

        # calculate average success rate, adjust cegis time allocation factor
        success_rate_cegar = self.stage_pruned_cegar / self.stage_time_cegar
        success_rate_cegis = self.stage_pruned_cegis / self.stage_time_cegis
        if self.stage_pruned_cegar == 0 or self.stage_pruned_cegis == 0:
            cegar_dominance = 1
        else:
            cegar_dominance = success_rate_cegar / success_rate_cegis
        cegis_dominance = 1 / cegar_dominance
        self.cegis_allocated_time_factor = cegis_dominance

        # switch back to cegar
        self.start(request_stage_cegar=True)
        return True



class SynthesizerHybrid(Synthesizer):
    
    def __init__(self, sketch):
        super().__init__(sketch)

        sketch.construct_jani()
        sketch.design_space.z3_initialize()
        
        self.stage_control = StageControl()

        # ar family stack
        self.families = []


    @property
    def method_name(self):
        return "hybrid"


    def analyze_family_cegis(self, family):
        return None
        '''
        Analyse a family against selected formulae using precomputed MDP data
        to construct generalized counterexamples.
        '''

        # TODO preprocess only formulae of interest

        logger.debug(f"CEGIS: analyzing family {family.design_space} of size {family.design_space.size}.")

        assert family.constructed
        assert family.analyzed

        # list of relevant holes (open constants) in this subfamily
        relevant_holes = [hole for hole in family.design_space.holes if len(family.hole_options[hole]) > 1]

        # prepare counterexample generator
        logger.debug("CEGIS: preprocessing quotient MDP")
        raw_formulae = [f.property.raw_formula for f in Family.formulae]
        counterexample_generator = stormpy.synthesis.SynthesisCounterexample(
            family.mdp, len(Family.sketch.design_space.holes), family.state_to_hole_indices, raw_formulae, family.bounds
        )
        

        # process family members
        assignment = family.pick_member()

        while assignment is not None:
            logger.debug(f"CEGIS: picked family member: {assignment}.")

            # collect indices of violated formulae
            violated_formulae_indices = []
            for formula_index in family.formulae_indices:
                # logger.debug(f"CEGIS: model checking DTMC against formula with index {formula_index}.")
                sat, result = family.analyze_member(formula_index)
                logger.debug(f"Formula {formula_index} is {'SAT' if sat else 'UNSAT'}")
                if not sat:
                    violated_formulae_indices.append(formula_index)
                formula = Family.formulae[formula_index]
                if sat and formula.optimality:
                    formula.improve_threshold(result)
                    counterexample_generator.replace_formula_threshold(
                        formula_index, formula.threshold, family.bounds[formula_index]
                    )
            # exit()
            if not violated_formulae_indices:
                return True


            # some formulae were UNSAT: construct counterexamples
            counterexample_generator.prepare_dtmc(family.dtmc, family.dtmc_state_map)
            
            conflicts = []
            for formula_index in violated_formulae_indices:
                # logger.debug(f"CEGIS: constructing CE for formula with index {formula_index}.")
                conflict_indices = counterexample_generator.construct_conflict(formula_index)
                # conflict = counterexample_generator.construct(formula_index, self.use_nontrivial_bounds)
                conflict_holes = [Family.hole_list[index] for index in conflict_indices]
                generalized_count = len(Family.hole_list) - len(conflict_holes)
                logger.debug(
                    f"CEGIS: found conflict involving {conflict_holes} (generalized {generalized_count} holes)."
                )
                conflicts.append(conflict_indices)
                # exit()

            exit()
            family.exclude_member(conflicts)

            # pick next member
            Profiler.start("is - pick DTMC")
            assignment = family.pick_member()
            Profiler.stop()

            # record stage
            if self.stage_control.step(0):
                # switch requested
                Profiler.add_ce_stats(counterexample_generator.stats)
                return None

        # full family pruned
        logger.debug("CEGIS: no more family members.")
        Profiler.add_ce_stats(counterexample_generator.stats)
        return False

    def run(self):
        
        # initialize family description
        logger.debug("Constructing quotient MDP of the superfamily.")
        self.models_total = self.sketch.design_space.size

        # FamilyHybrid.initialize(self.sketch)

        qmdp = MDP(self.sketch)
        # exit()

        # get the first family to analyze
        
        family = FamilyHybrid()
        family.construct()
        satisfying_assignment = None

        # CEGAR the superfamily
        self.stage_control.stage_start(request_stage_cegar=True)
        feasible, optimal_value = family.analyze()
        exit()

        self.stage_step(0)


        # initiate CEGAR-CEGIS loop (first phase: CEGIS) 
        self.families = [family]
        logger.debug("Initiating CEGAR--CEGIS loop")
        while self.families:
            logger.debug(f"Current number of families: {len(self.families)}")

            # pick a family
            family = self.families.pop(-1)
            if not self.stage_cegar:
                # CEGIS
                feasible = self.analyze_family_cegis(family)
                exit()
                if feasible and isinstance(feasible, bool):
                    logger.debug("CEGIS: some is SAT.")
                    satisfying_assignment = family.member_assignment
                    break
                elif not feasible and isinstance(feasible, bool):
                    logger.debug("CEGIS: all UNSAT.")
                    self.stage_step(family.size)
                    continue
                else:  # feasible is None:
                    # stage interrupted: leave the family to cegar
                    # note: phase was switched implicitly
                    logger.debug("CEGIS: stage interrupted.")
                    self.families.append(family)
                    continue
            else:  # CEGAR
                assert family.split_ready

                # family has already been analysed: discard the parent and refine
                logger.debug("Splitting the family.")
                subfamily_left, subfamily_right = family.split()
                subfamilies = [subfamily_left, subfamily_right]
                logger.debug(
                    f"Constructed two subfamilies of size {subfamily_left.size} and {subfamily_right.size}."
                )

                # analyze both subfamilies
                models_pruned = 0
                for subfamily in subfamilies:
                    self.iterations_cegar += 1
                    logger.debug(f"CEGAR: iteration {self.iterations_cegar}.")
                    subfamily.construct()
                    Profiler.start("ar - MDP model checking")
                    feasible, optimal_value = subfamily.analyze()
                    Profiler.stop()
                    if feasible and isinstance(feasible, bool):
                        logger.debug("CEGAR: all SAT.")
                        satisfying_assignment = subfamily.member_assignment
                        if optimal_value is not None:
                            self._check_optimal_property(
                                subfamily, satisfying_assignment, cex_generator=None, optimal_value=optimal_value
                            )
                        elif satisfying_assignment is not None and self._optimality_setting is None:
                            break
                    elif not feasible and isinstance(feasible, bool):
                        logger.debug("CEGAR: all UNSAT.")
                        models_pruned += subfamily.size
                        continue
                    else:  # feasible is None:
                        logger.debug("CEGAR: undecided.")
                        self.families.append(subfamily)
                        continue
                self.stage_step(models_pruned)

        if PRINT_PROFILING:
            Profiler.print()

        if self.input_has_optimality_property() and self._optimal_value is not None:
            assert not self.families
            logger.info(f"Found optimal assignment: {self._optimal_value}")
            return self._optimal_assignment, self._optimal_value
        elif satisfying_assignment is not None:
            logger.info(f"Found satisfying assignment: {readable_assignment(satisfying_assignment)}")
            return satisfying_assignment, None
        else:
            logger.info("No more options.")
            return None, None



