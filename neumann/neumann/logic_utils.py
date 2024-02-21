from hypothesis import target
from tqdm import tqdm

from .fol.data_utils import DataUtils
from .fol.language import DataType
from .fol.logic import *

p_ = Predicate(".", 1, [DataType("spec")])
false = Atom(p_, [Const("__F__", dtype=DataType("spec"))])
true = Atom(p_, [Const("__T__", dtype=DataType("spec"))])


def get_lang(
    lark_path,
    lang_base_path,
    dataset_type,
    dataset,
    term_depth,
    use_learned_clauses=False,
):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    print("Loading FOL language.")
    du = DataUtils(
        lark_path=lark_path,
        lang_base_path=lang_base_path,
        dataset_type=dataset_type,
        dataset=dataset,
    )
    lang = du.load_language()
    if not use_learned_clauses:
        clauses = add_true_atoms(du.load_clauses(du.base_path + "clauses.txt", lang))
    else:
        clauses = add_true_atoms(
            du.load_clauses(du.base_path + "learned_clauses.txt", lang)
        )
    bk_clauses = add_true_atoms(du.load_clauses(du.base_path + "bk_clauses.txt", lang))
    bk = du.load_atoms(du.base_path + "bk.txt", lang)
    terms = generate_terms(lang, max_depth=term_depth)
    print("{} terms are generated!".format(len(terms)))
    atoms = generate_atoms(lang, terms, dataset_type)
    print("{} ground atoms are generated!".format(len(atoms)))
    # atoms = du.get_facts(lang)
    return lang, clauses, bk, bk_clauses, terms, atoms


def get_lang_behind_the_scenes(lark_path, lang_base_path, term_depth):
    """Load the language of first-order logic from files.

    Read the language, clauses, background knowledge from files.
    Atoms are generated from the language.
    """
    print("Loading FOL language.")
    du = DataUtils(
        lark_path=lark_path,
        lang_base_path=lang_base_path,
        dataset_type="behind-the-scenes",
    )
    lang = du.load_language()
    clauses = add_true_atoms(du.load_clauses(du.base_path + "clauses.txt", lang))
    bk_clauses = add_true_atoms(du.load_clauses(du.base_path + "bk_clauses.txt", lang))
    bk = du.load_atoms(du.base_path + "bk.txt", lang)
    print("Generating Temrs ...")
    terms = generate_terms(lang, max_depth=term_depth)
    print("Generating Atoms ...")
    atoms = generate_atoms(lang, terms, "")
    # atoms = du.get_facts(lang)
    return lang, clauses, bk, bk_clauses, terms, atoms


def _add_true_atom(clauses):
    """Add true atom T to body: p(X,Y):-. => p(X,Y):-T."""
    cs = []
    for clause in clauses:
        if len(clause.body) == 0:
            clause.body.append(true)
            cs.append(clause)
        else:
            cs.append(clause)
    return cs


def to_list(func_term):
    if type(func_term) == FuncTerm:
        return [func_term.args[0]] + to_list(func_term.args[1])
    else:
        return [func_term]


def generate_terms(lang, max_depth):
    consts = lang.consts
    funcs = lang.funcs
    terms = consts
    print("Generating terms... ")
    for i in range(max_depth):
        new_terms = []
        for f in funcs:
            terms_list = []
            for in_dtype in f.in_dtypes:
                terms_dt = [term for term in terms if term.dtype == in_dtype]
                terms_list.append(terms_dt)
                args_list = list(set(itertools.product(*terms_list)))

            for args in args_list:
                # generate list by removing duplications of elements
                if len(args) == 2:
                    if not args[0] in to_list(args[1]):
                        # print(args[1], to_list(args[1]))
                        # for list pruning adhoc
                        new_terms.append(FuncTerm(f, args))
                        new_terms = list(set(new_terms))
                        terms.extend(new_terms)
                else:
                    new_terms.append(FuncTerm(f, args))
                    new_terms = list(set(new_terms))
                    terms.extend(new_terms)

    for x in sorted(list(set(terms))):
        print(x)
    return sorted(list(set(terms)))


def __generate_terms(lang, max_depth):
    consts = lang.consts
    funcs = lang.funcs
    terms = consts
    for i in tqdm(range(max_depth)):
        new_terms = []
        for f in funcs:
            terms_list = []
            for in_dtype in f.in_dtypes:
                terms_dt = [term for term in terms if term.dtype == in_dtype]
                terms_list.append(terms_dt)
                args_list = list(set(itertools.product(*terms_list)))

            for args in args_list:
                new_terms.append(FuncTerm(f, args))
                new_terms = list(set(new_terms))
                terms.extend(new_terms)
    return sorted(list(set(terms)))


def generate_atoms(lang, terms, max_term_depth=2):
    # spec_atoms = [false, true]
    atoms = []
    # terms = generate_terms(lang, max_depth=max_term_depth)
    for pred in lang.preds:
        dtypes = pred.dtypes
        terms_list = [
            [term for term in terms if term.dtype == dtype] for dtype in dtypes
        ]
        # consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
        args_list = []
        # print("Generating ground atoms for predicate: {}".format(pred.name))
        for terms_ in set(itertools.product(*terms_list)):
            args_list.append(terms_)
        for args in args_list:
            atoms.append(Atom(pred, args))
    atoms = set(atoms)
    # print("Sorting {} atoms...".format(len(list(atoms))))
    return [true] + sorted(list(atoms))


def generate_bk(lang):
    atoms = []
    for pred in lang.preds:
        if pred.name in ["diff_color", "diff_shape"]:
            dtypes = pred.dtypes
            consts_list = [lang.get_by_dtype(dtype) for dtype in dtypes]
            args_list = itertools.product(*consts_list)
            for args in args_list:
                if len(args) == 1 or (
                    args[0] != args[1] and args[0].mode == args[1].mode
                ):
                    atoms.append(Atom(pred, args))
    return atoms


def get_index_by_predname(pred_str, atoms):
    for i, atom in enumerate(atoms):
        if atom.pred.name == pred_str:
            return i
        assert 1, pred_str + " not found."


def parse_clauses(lang, clause_strs):
    du = DataUtils(lang)
    return [du.parse_clause(c) for c in clause_strs]


def get_all_vars_with_dtype(atoms):
    var_dtype_list = []
    for atom in atoms:
        vd_list = atom.all_vars_and_dtypes()
        for vd in vd_list:
            if not vd in var_dtype_list:
                var_dtype_list.append(vd)

    return var_dtype_list


def get_all_vars_with_depth(atoms):
    var_depth_list = []
    for atom in atoms:
        vd_list = atom.all_vars_with_depth()
        for v, depth in vd_list:
            if not v in [vd[0] for vd in var_depth_list]:
                var_depth_list.append((v, depth))
            else:
                for i, (v_, depth_) in enumerate(var_depth_list):
                    if v == v_ and depth > depth_:
                        del var_depth_list[i]
                        var_depth_list.append((v, depth))
    return var_depth_list


def get_terms_by_dtype(dtype, terms):
    return [term for term in terms if term.dtype == dtype]


def get_terms_by_dtype_and_depth(dtype, depth, terms, max_term_depth):
    return [
        term
        for term in terms
        if term.dtype == dtype and term.max_depth() + depth <= max_term_depth
    ]


def generate_by_cartesian_product(dtypes, depths, terms, max_term_depth):
    """Generall all possible combinations of terms to be substituted to remove existentially quantified variables."""
    # enumerate possible terms for each data type
    terms_list = []
    for i in range(len(dtypes)):
        terms_list.append(
            get_terms_by_dtype_and_depth(dtypes[i], depths[i], terms, max_term_depth)
        )

    # generate possible substitutions by taking cartesian product
    subs_terms_list = []
    for terms in list(itertools.product(*terms_list)):
        # after taking cartesian product, each element represents one possible substitution
        # to remove all variables in the atoms
        subs_terms_list.append(terms)
    # e.g. if the data type is shape, then subs_consts_list = [(red,), (yellow,), (blue,)]
    return subs_terms_list


def make_var_term_pairs(vars, terms_list):
    """Generate pairs (Variable, Term) as a list of substitutions."""
    # generate substitutions by combining variables to the head of subs_consts_list
    theta_list = []
    for subs_terms in terms_list:
        theta = []
        for i, const in enumerate(subs_terms):
            s = (vars[i], const)
            if s not in theta:
                theta.append(s)
        # if theta not in theta_list:
        theta_list.append(theta)
    # e.g. theta_list: [[(Z, red)], [(Z, yellow)], [(Z, blue)]]
    return theta_list


def generate_substitutions(atoms, terms, max_term_depth):
    """Generate substitutions from given body atoms.
    Generate the possible substitutions from given list of atoms by enumerating terms that matches the data type.

    Args:
        atoms (list(atom)): The body atoms which may contain existentially quantified variables.

    Returns:
        theta_list (list(substitution)): The list of substitutions of the given body atoms.
    """

    var_dtype_list = get_all_vars_with_dtype(atoms)
    var_depth_list = get_all_vars_with_depth(atoms)
    vars = [vd[0] for vd in var_dtype_list]
    dtypes = [vd[1] for vd in var_dtype_list]
    depths = [vd[1] for vd in var_depth_list]
    # check the data type consistency
    assert not invalid_var_dtypes(
        var_dtype_list
    ), "Invalid data type to generate substitutions for existentially quantified variables."
    # in case there is no variables in the body
    if len(list(set(dtypes))) == 0:
        return []
    terms_list = generate_by_cartesian_product(dtypes, depths, terms, max_term_depth)
    theta_list = make_var_term_pairs(vars, terms_list)
    # remove duplications of substitutions
    return list(set([tuple(x) for x in theta_list]))


def invalid_var_dtypes(var_dtypes):
    # check the contradiciton of the List[(var, dtype)]
    if len(var_dtypes) < 2:
        return False
    for i in range(len(var_dtypes) - 1):
        for j in range(i, len(var_dtypes)):
            if (
                var_dtypes[i][0] == var_dtypes[j][0]
                and var_dtypes[i][1] != var_dtypes[j][1]
            ):
                return True
    return False


def is_tautology(clause):
    return len(clause.body) == 1 and clause.head == clause.body[0]


def add_true_atom(clause):
    if len(clause.body) == 0:
        return Clause(clause.head, [true])
    else:
        return clause


def remove_true_atom(clause):
    if len(clause.body) == 1 and clause.body[0] == true:
        return Clause(clause.head, [])
    else:
        return clause


def add_true_atoms(clauses):
    return [add_true_atom(c) for c in clauses]


def remove_true_atoms(clauses):
    return [remove_true_atom(c) for c in clauses]
