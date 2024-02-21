import itertools

from .fol.language import DataType
from .fol.logic import Atom, Clause, Const, FuncTerm, Var
from .fol.logic_ops import subs
from .logic_utils import (get_all_vars_with_dtype, invalid_var_dtypes,
                         is_tautology, true)


# TODOL refine_from_modeb, generate_by_refinement
class RefinementGenerator(object):
    """
    refinement operations for clause generation
    Parameters
    ----------
    lang : .language.Language
    max_depth : int
        max depth of nests of function symbols
    max_body_len : int
        max number of atoms in body of clauses
    """

    def __init__(self, lang, mode_declarations, max_depth=1, max_body_len=1, max_var_num=5, refinement_types=['atom']):
        self.lang = lang
        self.mode_declarations = mode_declarations
        self.vi = 0 # counter for new variable generation
        self.max_depth = max_depth
        self.max_body_len = max_body_len
        self.max_var_num = max_var_num
        self.refinement_types = refinement_types


    def _init_recall_counter_dic(self, mode_declarations):
        dic = {}
        for md in mode_declarations:
            dic[str(md)] = 0
        return dic

    def _check_recall(self, clause, mode_declaration):
        """Return a boolean value that represents the mode declaration can be used or not
        in terms of the recall.
        """
        pred_num = clause.count_by_predicate(mode_declaration.pred)
        md_recall = mode_declaration.recall
        return pred_num < md_recall
        #return self.recall_counter_dic[str(mode_declaration)] < mode_declaration.recall

    def _increment_recall(self, mode_declaration):
        self.recall_counter_dic[str(mode_declaration)] += 1

    def _remove_invalid_clauses(self, clauses):
        """Fiter and obtain valid clauses."""
        result = []
        for clause in clauses:
            var_dtypes = get_all_vars_with_dtype([clause.head]+clause.body)
            if not invalid_var_dtypes(var_dtypes) and not is_tautology(clause) and len(clause.all_vars()) <= self.max_var_num:
                result.append(clause)
        return result



    def get_max_obj_id(self, clause):
        object_vars = clause.all_vars_by_dtype('object')
        object_ids = [int(x.name.split('O')[-1]) for x in object_vars]
        if len(object_ids) == 0:
            return 0
        else:
            return max(object_ids)


    def __generate_new_variable(self, n):
        # We assume that we have only object variables as new variables
        # O1, O2, ....
        #new_var = Var('O' + str(self.object_counter))
        #new_var = Var("__Y" + str(self.vi) + "__")
        #self.vi += 1
        #return new_var
        #new_var = Var('O' + str(n+1))
        #self.object_counter += 1
        return new_var



    def generate_new_variable(self, clause):
        obj_id = self.get_max_obj_id(clause)
        return Var('O' + str(obj_id+1))


    def refine_from_modeb(self, clause, modeb):
        """Generate clauses by adding atoms to body using mode declaration.
        Args:
              clause (Clause): A clause.
              modeb (ModeDeclaration): A mode declaration for body.
        """
        # list(list(Term))
        if not self._check_recall(clause, modeb):
            # the input modeb has been used as many as its recall (maximum number  to be called) already
            return []
        terms_list = self.generate_term_combinations(clause, modeb)

        C_refined = []
        for terms in terms_list:
            if len(terms) == len(list(set(terms))):
                # terms: (O0, X)
                if not modeb.ordered:
                    terms = sorted(terms)
                new_atom = Atom(modeb.pred, terms)
                if not new_atom in clause.body:
                    new_body = sorted(clause.body + [new_atom])
                    new_clause = Clause(clause.head, new_body)
                    # remove tautology
                    if not (len(clause.body)==1 and clause.head == clause.body[0]):
                        C_refined.append(new_clause)
        #self._increment_recall(modeb)
        return list(set(C_refined))

    def generate_term_combinations(self, clause, modeb):
        """Generate possible term list for new body atom.
        Enumerate possible assignments for each place in the mode predicate,
        generate all possible assignments by enumerating the combinations.
        Args:
            modeb (ModeDeclaration): A mode declaration for body.
        """
        assignments_list = []
        for mt in modeb.mode_terms:
            if mt.mode == '+':
                # var_candidates = clause.var_all()
                assignments = clause.all_vars_by_dtype(mt.dtype)
            elif mt.mode == '-':
                # get new variable
                # How to think as candidates? maybe [O3] etc.
                # we get only object variable e.g. O3
                # new_var = self.generate_new_variable()
                assignments = [self.generate_new_variable(clause)]
            elif mt.mode == '#':
                # consts = self.lang.get_by_dtype(mt.mode.dtype)
                assignments = self.lang.get_by_dtype(mt.dtype)
            assignments_list.append(assignments)
        # generate all combinations by cartesian product
        # e.g. [[O2], [red,blue,yellow]]
        # -> [[O2,red],[O2,blue],[O2,yellow]]
        ##print(assignments_list)
        ##print(list(itertools.product(*assignments_list)))
        ##print(clause, modeb, assignments_list)
        #print(clause, modeb)
        #print(assignments_list)
        if modeb.ordered:
            return itertools.product(*assignments_list)
        else:
            return itertools.combinations(assignments_list[0], modeb.pred.arity)

    def refine_clause(self, clause):
        C_refined = []
        if 'atom' in self.refinement_types:
            for modeb in self.mode_declarations:
                C_refined.extend(self.refine_from_modeb(clause, modeb))
        if 'func' in self.refinement_types:
            C_refined.extend(self.apply_func(clause))
        if 'const' in self.refinement_types:
            C_refined.extend(self.subs_const(clause))
        if 'var' in self.refinement_types:
            C_refined.extend(self.subs_var(clause))
        # C_refined.extend(self.swap_vars(clause))
        result = self._remove_invalid_clauses(list(set(C_refined)))
        return result



    def refine_clauses(self, clauses):
        """Perform refinement for given set of clauses.
        Args:
            clauses (list(Clauses)): A set of clauses.
        Returns:
            list(Clauses): A set of refined clauses using modeb declarations.
        """
        result = []
        for clause in clauses:
            if len(clause.body) == 1 and clause.body[0].pred.name == '.':
                clause.body = []
            C_refined = self.refine_clause(clause)
            # put it back to the original state
            clause.body = [true]
            for c in C_refined:
                if not (c in result):
                    result.append(c)
        return result

    def apply_func(self, clause):
        """
        z/f(x_1, ..., x_n) for every variable in C and every n-ary function symbol f in the language
        """
        refined_clauses = []
        #if (len(clause.body) >= self.max_body_len) or (len(clause.all_consts()) >= 1):
        for z, dtype in clause.head.all_vars_and_dtypes():
            # for z in clause.all_vars():
            for f in self.lang.funcs:
                if f.out_dtype == dtype:
                    new_vars = [self.lang.var_gen.generate()
                                for v in range(f.arity)]
                    func_term = FuncTerm(f, new_vars)
                    # TODO: check variable z's depth
                    result = subs(clause, z, func_term)
                    if result.max_depth() <= self.max_depth:
                        result.rename_vars()
                        refined_clauses.append(result)
        return refined_clauses

    def subs_var(self, clause):
        """
        z/x for every distinct variables x and z in C
        """
        refined_clauses = []
        # to HEAD
        all_vars = clause.head.all_vars()
        combs = itertools.combinations(all_vars, 2)
        for u, v in combs:
            result = subs(clause, u, v)
            result.rename_vars()
            refined_clauses.append(result)
        return refined_clauses

    def swap_vars(self, clause):
        """Swapping variables in the body"""
        if len(clause.body) == 0:
            return []
        else:
            refined_clauses = []
            var_dtype_list = get_all_vars_with_dtype(clause.body)
            for vd_1, vd_2 in itertools.combinations(var_dtype_list, 2):
                # vd1, vd2 are tuples of (var, dtype)
                var_1 = vd_1[0]
                dtype_1 = vd_1[1]
                var_2 = vd_2[0]
                dtype_2 = vd_2[1]
                if dtype_1 == dtype_2 and var_1 != var_2:
                    var_tmp = Var("__tmp__")
                    new_body = []
                    for b_atom in clause.body:
                        new_b_atom = subs(b_atom, var_1, var_tmp)
                        new_b_atom = subs(new_b_atom, var_2, var_1)
                        new_b_atom = subs(new_b_atom, var_tmp, var_2)
                        new_body.append(new_b_atom)
                    new_clause = Clause(clause.head, new_body)
                    refined_clauses.append(new_clause)
            return refined_clauses



    def subs_const(self, clause):
        """
        z/a for every variable z in C and every constant a in the language
        """
        #if (len(clause.body) >= self.max_body_len) or (clause.max_depth() >= 1):
        #    return []

        if (len(clause.body) >= self.max_body_len) or (clause.max_depth() >= 2):
            return []
        refined_clauses = []
        all_vars = clause.head.all_vars_by_dtype(DataType('colors'))
        consts = [term for term in self.lang.get_by_dtype_name('colors') if type(term) == Const]
        for v, c in itertools.product(all_vars, consts):
            result = subs(clause, v, c)
            result.rename_vars()
            refined_clauses.append(result)
        return refined_clauses