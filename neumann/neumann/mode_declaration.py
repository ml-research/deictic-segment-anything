from fol.language import DataType


class ModeDeclaration(object):
    """from https://www.cs.ox.ac.uk/activities/programinduction/Aleph/aleph.html
    p(ModeType, ModeType,...)
    Here are some examples of how they appear in a file:
    :- mode(1,mem(+number,+list)).
    :- mode(1,dec(+integer,-integer)).
    :- mode(1,mult(+integer,+integer,-integer)).
    :- mode(1,plus(+integer,+integer,-integer)).
    :- mode(1,(+integer)=(#integer)).
    :- mode(*,has_car(+train,-car)).
    Each ModeType is either (a) simple; or (b) structured.
    A simple ModeType is one of:
    (a) +T specifying that when a literal with predicate symbol p appears in a
    hypothesised clause, the corresponding argument should be an "input" variable of type T;
    (b) -T specifying that the argument is an "output" variable of type T; or
    (c) #T specifying that it should be a constant of type T.
    All the examples above have simple modetypes.
    A structured ModeType is of the form f(..) where f is a function symbol,
    each argument of which is either a simple or structured ModeType.
    Here is an example containing a structured ModeType:
    To make this more clear, here is an example for the mode declarations for
    the grandfather task from
    above::- modeh(1, grandfather(+human, +human)).:-
    modeb(*, parent(-human, +human)).:-
    modeb(*, male(+human)).
    The  first  mode  states  that  the  head  of  the  rule
    (and  therefore  the  targetpredicate) will be the atomgrandfather.
    Its parameters have to be of the typehuman.
    The  +  annotation  says  that  the  rule  head  needs  two  variables.
    Thesecond mode declaration states theparentatom and declares again
    that theparameters have to be of type human.
    Here,  the + at the second parametertells, that the system is only allowed to
    introduce the atomparentin the clauseif it already contains a variable of type human.
    The first attribute introduces a new variable into the clause.
    The  modes  consist  of  a  recall n that  states  how  many  versions  of  the
    literal are allowed in a rule and an atom with place-markers that state the literal to-gether
    with annotations on input- and output-variables as well as constants (see[Mug95]).

    Args:
        recall (int): The recall number i.e. how many times the declaration can be instanciated
        pred (Predicate): The predicate.
        mode_terms (ModeTerm): Terms for mode declarations.
    """

    def __init__(self, mode_type, recall, pred, mode_terms, ordered=True):
        self.mode_type = mode_type  # head or body
        self.recall = recall
        self.pred = pred
        self.mode_terms = mode_terms
        self.ordered = ordered

    def __str__(self):
        s = 'mode_' + self.mode_type + '('
        for mt in self.mode_terms:
            s += str(mt)
            s += ','
        s = s[0:-1]
        s += ')'
        return s

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())


class ModeTerm(object):
    """Terms for mode declarations. It has mode (+, -, #) and data types.
    """

    def __init__(self, mode, dtype):
        self.mode = mode
        assert mode in ['+', '-', '#'], "Invalid mode declaration."
        self.dtype = dtype

    def __str__(self):
        return self.mode + self.dtype.name

    def __repr__(self):
        return self.__str__()


def get_mode_declarations_clevr(lang, obj_num):
    p_image = ModeTerm('+', DataType('image'))
    m_object = ModeTerm('-', DataType('object'))
    p_object = ModeTerm('+', DataType('object'))
    s_color = ModeTerm('#', DataType('color'))
    s_shape = ModeTerm('#', DataType('shape'))
    s_material = ModeTerm('#', DataType('material'))
    s_size = ModeTerm('#', DataType('size'))

    # modeh_1 = ModeDeclaration('head', 'kp', p_image)

    """
    kp1(X):-in(O1,X),in(O2,X),size(O1,large),shape(O1,cube),size(O2,large),shape(O2,cylinder).
    kp2(X):-in(O1,X),in(O2,X),size(O1,small),material(O1,metal),shape(O1,cube),size(O2,small),shape(O2,sphere).
    kp3(X):-in(O1,X),in(O2,X),size(O1,large),color(O1,blue),shape(O1,sphere),size(O2,small),color(O2,yellow),shape(O2,sphere)."""

    modeb_list = [
        #ModeDeclaration('body', obj_num, lang.get_pred_by_name(
        #    'in'), [m_object, p_image]),
        ModeDeclaration('body', 2, lang.get_pred_by_name(
            'color'), [p_object, s_color]),
        ModeDeclaration('body', 2, lang.get_pred_by_name(
            'shape'), [p_object, s_shape]),
        #ModeDeclaration('body', 1, lang.get_pred_by_name(
        #    'material'), [p_object, s_material]),
        ModeDeclaration('body', 2, lang.get_pred_by_name(
            'size'), [p_object, s_size]),
    ]
    return modeb_list


def get_mode_declarations_kandinsky(lang, obj_num):
    p_image = ModeTerm('+', DataType('image'))
    m_object = ModeTerm('-', DataType('object'))
    p_object = ModeTerm('+', DataType('object'))
    s_color = ModeTerm('#', DataType('color'))
    s_shape = ModeTerm('#', DataType('shape'))

    # modeh_1 = ModeDeclaration('head', 'kp', p_image)

    modeb_list = [
        #ModeDeclaration('body', obj_num, lang.get_pred_by_name(
        #    'in'), [m_object, p_image]),
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'color'), [p_object, s_color]),
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'shape'), [p_object, s_shape]),
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'same_color_pair'), [p_object, p_object], ordered=False),
        ModeDeclaration('body', 2, lang.get_pred_by_name(
            'same_shape_pair'), [p_object, p_object], ordered=False),
        ModeDeclaration('b１ody', 1, lang.get_pred_by_name(
            'diff_color_pair'), [p_object, p_object], ordered=False),
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'diff_shape_pair'), [p_object, p_object], ordered=False),
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'closeby'), [p_object, p_object], ordered=False),
        ModeDeclaration('body', 1, lang.get_pred_by_name('online'), [
                        p_object, p_object, p_object, p_object, p_object], ordered=False),
        # ModeDeclaration('body', 2, lang.get_pred_by_name('diff_shape_pair'), [p_object, p_object]),
    ]
    return modeb_list

def get_mode_declarations_vilp(lang, dataset):
    p_colors = ModeTerm('+', DataType('colors'))
    p_color = ModeTerm('+', DataType('color'))
    # modeh_1 = ModeDeclaration('head', 'kp', p_image)
    if dataset=='member':
        modeb_list = [
            ModeDeclaration('body', 1, lang.get_pred_by_name(
            'member'), [p_color, p_colors])]
    elif dataset == 'delete':
        modeb_list = [
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'delete'), [p_color, p_colors, p_colors])]
    elif dataset == 'append':
        modeb_list = [
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'append'), [p_colors, p_colors, p_colors])]
    elif dataset == 'reverse':
        modeb_list = [
        ModeDeclaration('body', 1, lang.get_pred_by_name(
            'reverse'), [p_colors, p_colors, p_colors])
        #ModeDeclaration('body', 1, lang.get_pred_by_name(
        #    'append'), [p_colors, p_colors, p_colors]),
            ]
    elif dataset == 'sort':
        modeb_list = [
        ModeDeclaration('body', 1, lang.get_pred_by_name('perm'), [p_colors, p_colors]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('is_sorted'), [p_colors]),
        ModeDeclaration('body', 1, lang.get_pred_by_name('smaller'), [p_color, p_color]),
        ]
        #ModeDeclaration('body', 1, lang.get_pred_by_name(
        #    'append'), [p_colors, p_colors, p_colors]),
        #ModeDeclaration('body', 1, lang.get_pred_by_name(
        #    'reverse'), [p_colors, p_colors]),
        #ModeDeclaration('body', 1, lang.get_pred_by_name(
        #    'sort'), [p_colors, p_colors])
    return modeb_list

def get_mode_declarations(args, lang):
    if args.dataset_type == 'kandinsky':
        return get_mode_declarations_kandinsky(lang, args.num_objects)
    elif args.dataset_type == 'clevr-hans':
        return get_mode_declarations_clevr(lang, 10)
    elif args.dataset_type == 'vilp':
        return get_mode_declarations_vilp(lang, args.dataset)
    else:
        assert False, "Invalid data type."