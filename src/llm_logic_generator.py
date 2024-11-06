import json
import time
import lark
import openai
from lark import Lark
from neumann.fol.exp_parser import ExpTree
from neumann.fol.language import DataType, Language
from neumann.fol.logic import Atom, Clause, Const, NeuralPredicate, Predicate


flatten = lambda x: [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]


class LLMLogicGenerator:
    """
    A class to generate logic languages and rules from task descriptions in natural language
    using GPT and prepared textual prompts.
    """

    def __init__(self, api_key):
        self.constants_prompt_path = "prompt/gen_constants.txt"
        self.predicates_prompt_path = "prompt/gen_predicates.txt"
        self.rules_prompt_path = "prompt/gen_rules.txt"

        self.constants_prompt = self._load_prompt(self.constants_prompt_path)
        self.predicates_prompt = self._load_prompt(self.predicates_prompt_path)
        self.rules_prompt = self._load_prompt(self.rules_prompt_path)

        # Setup parser
        lark_path = "src/lark/exp.lark"
        with open(lark_path, encoding="utf-8") as grammar:
            grammar_content = grammar.read()
        self.lp_atom = Lark(grammar_content, start="atom")
        self.lp_clause = Lark(grammar_content, start="clause")

        # Setup OpenAI API
        openai.api_key = api_key
        openai.organization = None

    def _load_prompt(self, file_path):
        with open(file_path, "r") as file:
            return file.read()

    def _parse_response(self, response, parse_function):
        return flatten([parse_function(line) for line in response.split("\n") if line.strip()])

    def _parse_constants(self, line):
        try:
            dtype_name, const_names_str = line.replace(" ", "").split(":")
            dtype = DataType(dtype_name)
            const_names = const_names_str.split(",")
            return [Const(name, dtype) for name in const_names]
        except ValueError:
            return []

    def _parse_predicates(self, line):
        pred_names = line.replace(" ", "").split(",")
        return [
            NeuralPredicate(name, 2, [DataType("object"), DataType("object")])
            for name in pred_names
        ]

    def _parse_rules(self, line, language):
        if ":-" in line:
            tree = self.lp_clause.parse(line.replace(" ", ""))
            return ExpTree(language).transform(tree)
        return None

    def query_gpt(self, text):
        """Query GPT-3.5 with a textual prompt."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": text}],
            )
            return response.choices[0]["message"]["content"].strip()
        except (
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
            json.JSONDecodeError,
        ):
            time.sleep(3)
            print("OpenAI API error encountered. Retrying...")
            return self.query_gpt(text)

    def generate_constants(self, text):
        prompt = f"{self.constants_prompt}\n{text}\n Constants:"
        response = self.query_gpt(prompt)
        constants = self._parse_response(response, self._parse_constants)
        return constants, "Constants:\n" + response

    def generate_predicates(self, text):
        prompt = f"{self.predicates_prompt}\n{text}\n"
        response = self.query_gpt(prompt)
        predicates = self._parse_response(response, self._parse_predicates)

        required_predicates = [
            ("type", 2, [DataType("object"), DataType("type")]),
            ("target", 1, [DataType("object")]),
            ("cond1", 1, [DataType("object")]),
            ("cond2", 1, [DataType("object")]),
            ("cond3", 1, [DataType("object")]),
        ]

        for name, arity, datatypes in required_predicates:
            if not any(p.name == name for p in predicates):
                if arity == 2:
                    predicates.append(NeuralPredicate(name, arity, datatypes))
                else:
                    predicates.append(Predicate(name, arity, datatypes))

        return predicates, response

    def get_preds_string(self, preds):
        return ",".join(p.name for p in preds if p.name not in ["target", "type", "cond1", "cond2", "cond3"])

    def generate_rules(self, text, language):
        pred_response = self.get_preds_string(language.preds)
        self.pred_response = pred_response
        prompt = f"\n\n{self.rules_prompt}\n{text}\navailable predicates: {pred_response}\n"
        response = self.query_gpt(prompt)
        self.rule_response = response

        rules = self._parse_response(response, lambda line: self._parse_rules(line, language))
        return [rule for rule in rules if rule]

    def generate_logic(self, text):
        """Generate constants, predicates, and rules using LLMs."""
        constants, const_response = self.generate_constants(text)
        if not constants:
            raise ValueError("Error: No constants found in the prompt.")
        predicates, _ = self.generate_predicates(text)
        if not predicates:
            raise ValueError("Error: No predicates found in the prompt.")
        
        language = Language(consts=constants, preds=predicates, funcs=[])
        rules = self.generate_rules(text, language)
        return language, rules