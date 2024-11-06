import neumann
import neumann.fol.logic as logic
from neumann.fol.language import DataType, Language
from neumann.fol.logic import Const


def scene_graph_to_language(scene_graph, text, logic_generator, num_objects=2):
    """Extract a FOL language from a scene graph to parse rules later."""
    
    # Extract and sanitize object names from the scene graph
    objects = {str(obj).replace(" ", "") for obj in scene_graph.objects}
    datatype = DataType("type")
    
    # Define constants using the extracted object names
    constants = [Const(obj, datatype) for obj in objects]

    # Prepare constant response for predicates
    const_response = "Constants:\ntype:" + ",".join(objects)
    
    # Generate predicates using the logic generator based on the input text
    predicates, pred_response = logic_generator.generate_predicates(text)
    print(f"Predicate generator response:\n    {pred_response}")
    
    # Formulate the language using constants and predicates
    lang = Language(consts=list(constants), preds=list(predicates), funcs=[])
    return lang


def get_init_language_with_sgg(scene_graph, text, logic_generator):
    """Extract an initial FOL language from a predicted scene graph to parse rules later."""
    
    # Extract unique object and subject names from the scene graph
    objects = {rel["o_str"] for rel in scene_graph}
    subjects = {rel["s_str"] for rel in scene_graph}
    datatype = DataType("type")
    
    # Define constants using the extracted names
    constants = [Const(obj, datatype) for obj in objects | subjects]

    # Prepare constant response for predicates
    const_response = "Constants:\ntype:" + ",".join(objects)
    
    # Generate predicates using the logic generator based on the input text
    predicates, pred_response = logic_generator.generate_predicates(text, const_response)
    print(f"Predicate generator response:\n    {pred_response}")
    
    # Formulate the language using constants and predicates
    lang = Language(consts=list(constants), preds=list(predicates), funcs=[])
    return lang


def scene_graph_to_language_with_sgg(scene_graph):
    """Extract a complete FOL language from a scene graph for use with a semantic unifier."""
    
    # Extract unique objects, subjects, and relationships from the scene graph
    objects = {rel["o_str"] for rel in scene_graph}
    subjects = {rel["s_str"] for rel in scene_graph}
    relationships = {rel["p_str"] for rel in scene_graph}
    datatype = DataType("type")
    obj_datatype = DataType("object")
    
    # Define constants using the extracted names
    constants = [Const(obj, datatype) for obj in objects | subjects]

    # Define predicates for each relationship
    predicates = [
        logic.Predicate(rel.replace(" ", "_").lower(), 2, [obj_datatype, obj_datatype])
        for rel in relationships
    ]

    # Formulate the language using constants and predicates
    lang = Language(consts=list(constants), preds=list(predicates), funcs=[])
    return lang


def objdata_to_box(data):
    """Convert object data to a bounding box format."""
    x, y, w, h = data["x"], data["y"], data["w"], data["h"]
    return x, y, w, h
