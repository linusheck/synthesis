import math
import itertools
import z3

class Hole:
    '''
    Hole with a name, a list of options and corresponding option labels.
    Options for each hole are simply indices of the corresponding actions.
    Each hole is identified by its position in Holes, therefore, this order must
      be preserved in the refining process.
    Option labels are not refined when assuming suboptions so that the correct
      label can be accessed by the value of an option.
    '''
    def __init__(self, name, options, option_labels):
        self.name = name
        self.options = options
        self.option_labels = option_labels

    @property
    def size(self):
        return len(self.options)

    def __str__(self):
        labels = [self.option_labels[option] for option in self.options]
        if self.size == 1:
            return"{}={}".format(self.name,labels[0]) 
        else:
            return self.name + ": {" + ",".join(labels) + "}"

    def copy(self):
        return Hole(self.name, self.options.copy(), self.option_labels)

    def assume_suboptions(self, suboptions):
        return Hole(self.name, suboptions.copy(), self.option_labels)


class Holes(list):
    ''' List of holes. '''

    def __init__(self, *args):
        super().__init__(*args)

    @property
    def num_holes(self):
        return len(self)

    @property
    def hole_indices(self):
        return list(range(len(self)))

    @property
    def size(self):
        ''' Family size. '''
        return math.prod([hole.size for hole in self])

    def __str__(self):
        return ", ".join([str(hole) for hole in self]) 

    def copy(self):
        return Holes([hole.copy() for hole in self])

    def assume_hole_suboptions(self, hole_index, suboptions):
        ''' Assume suboptions of a certain hole. '''
        result = self.copy()
        result[hole_index] = self[hole_index].assume_suboptions(suboptions)
        return result

    def assume_suboptions(self, hole_suboptions):
        ''' Assume suboptions for each hole. '''
        return Holes([hole.assume_suboptions(hole_suboptions[hole_index]) for hole_index,hole in enumerate(self)])

    def pick_any(self):
        suboptions = [[hole.options[0]] for hole in self]
        return self.assume_suboptions(suboptions)

    def includes(self, hole_options):
        for hole_index,option in hole_options.items():
            if not option in self[hole_index].options:
                return False
        return True


class DesignSpace(Holes):
    '''
    List of holes supplied with
    - a list of constraints to investigate in this design space
    - (optionally) z3 encoding of this design space
    :note z3 (re-)encoding construction must be invoked manually
    '''

    # z3 solver containing description of the complete design space
    solver = None
    # for each hole, a corresponding solver variable
    solver_vars = None

    def __init__(self, holes, properties = None):
        super().__init__(holes)
        self.properties = properties
        self.encoding = None

    def set_properties(self, properties):
        self.properties = properties

    def copy(self):
        design_space = DesignSpace(super().copy())
        design_space.set_properties(self.properties.copy())
        return design_space

    def z3_initialize(self):
        ''' Use this design space as a baseline for future refinements. '''
        DesignSpace.solver = z3.Solver()
        DesignSpace.solver_vars = []
        for hole_index, hole in enumerate(self):
            var = z3.Int(hole_index)
            DesignSpace.solver_vars.append(var)
            DesignSpace.solver.add(var >= 0)
            DesignSpace.solver.add(var < hole.size)

    def z3_encode(self):
        ''' Encode this design space. '''
        hole_clauses = []
        for hole_index,hole in enumerate(self):
            clauses = z3.Or(
                [DesignSpace.solver_vars[hole_index] == option for option in hole.options]
            )
            hole_clauses.append(clauses)
        self.encoding = z3.And(hole_clauses)

    def pick_assignment(self):
        '''
        Pick any (feasible) hole assignment.
        :return None if no instance remains
        '''
        # get satisfiable assignment within this design space
        solver_result = DesignSpace.solver.check(self.encoding)
        if solver_result != z3.sat:
            # no further instances
            return None

        # construct the corresponding singleton
        sat_model = DesignSpace.solver.model()
        assignment = []
        for hole_index,hole, in enumerate(self):
            var = DesignSpace.solver_vars[hole_index]
            option = sat_model[var].as_long()
            assignment.append([option])
        return self.assume_suboptions(assignment)

    def exclude_assignment(self, assignment, conflict):
        '''
        Exclude assignment from the design space using provided conflict.
        :param assignment hole assignment that yielded unsatisfiable DTMC
        :param conflict indices of relevant holes in the corresponding counterexample
        '''
        counterexample_clauses = []
        for hole_index,var in enumerate(DesignSpace.solver_vars):
            if hole_index in conflict:
                counterexample_clauses.append((var == assignment[hole_index].options[0]))
            else:
                all_options = [var == option for option in self[hole_index].options]
                counterexample_clauses.append(z3.Or(all_options))
        counterexample_encoding = z3.Not(z3.And(counterexample_clauses))
        DesignSpace.solver.add(counterexample_encoding)


class CombinationColoring:
    '''
    Dictionary of colors associated with different hole combinations.
    Note: color 0 is reserved for general hole-free objects.
    '''
    def __init__(self, holes):
        '''
        :param holes of the initial design space
        '''
        self.holes = holes
        self.coloring = {}
        self.reverse_coloring = {}

    @property
    def colors(self):
        return len(self.coloring)

    def get_or_make_color(self, hole_assignment):
        new_color = self.colors + 1
        color = self.coloring.get(hole_assignment, new_color)
        if color == new_color:
            self.coloring[hole_assignment] = color
            self.reverse_coloring[color] = hole_assignment
        return color

    def subcolors(self, subspace):
        ''' Collect colors that are valid within the provided design subspace. '''
        colors = set()
        for combination,color in self.coloring.items():
            contained = True
            for hole_index,hole in enumerate(subspace):
                if combination[hole_index] is None:
                    continue
                if combination[hole_index] not in hole.options:
                    contained = False
                    break
            if contained:
                colors.add(color)

        return colors

    def subcolors_proper(self, hole_index, options):
        colors = set()
        for combination,color in self.coloring.items():
            if combination[hole_index] in options:
                colors.add(color)
        return colors

    def get_hole_assignments(self, colors):
        ''' Collect all hole assignments associated with provided colors. '''
        hole_assignments = [set() for hole in self.holes]

        for color in colors:
            if color == 0:
                continue
            combination = self.reverse_coloring[color]
            for hole_index,assignment in enumerate(combination):
                if assignment is None:
                    continue
                hole_assignments[hole_index].add(assignment)
        hole_assignments = [list(assignments) for assignments in hole_assignments]

        return hole_assignments