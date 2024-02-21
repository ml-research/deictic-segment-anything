import json
import time

import lark
import openai
from lark import Lark

from neumann.fol.exp_parser import ExpTree
from neumann.fol.language import DataType, Language
from neumann.fol.logic import Atom, Clause, Const, NeuralPredicate, Predicate


class LLMLogicGenerator(object):
    """
    A class to generate logic languages and rules from task descriptions in natural language.
    We generate them by using GPT and prepared textual prompts (in the prompt folder).
    """

    def __init__(self, api_key):
        super(LLMLogicGenerator, self).__init__()
        self.constants_prompt_path = "prompt/gen_constants.txt"
        self.predicates_prompt_path = "prompt/gen_predicates.txt"
        self.rules_prompt_path = "prompt/gen_rules.txt"
        self.constants_prompt = self._load_constants_prompt()
        self.predicates_prompt = self._load_predicates_prompt()
        self.rules_prompt = self._load_rules_prompt()

        # setup parser
        lark_path = "src/lark/exp.lark"
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_atom = Lark(grammar.read(), start="atom")
        with open(lark_path, encoding="utf-8") as grammar:
            self.lp_clause = Lark(grammar.read(), start="clause")

        # setup openai API
        openai.organization = None
        openai.api_key = api_key

    def _load_constants_prompt(self):
        f = open(self.constants_prompt_path)
        prompt = f.read()
        f.close()
        return prompt

    def _load_predicates_prompt(self):
        f = open(self.predicates_prompt_path)
        prompt = f.read()
        f.close()
        return prompt

    def _load_rules_prompt(self):
        f = open(self.rules_prompt_path)
        prompt = f.read()
        f.close()
        return prompt

    def _parse_response_to_constants(self, response):
        constants = []
        lines = response.split("\n")
        for line in lines:
            # match_result = re.match(pattern, line)
            # if not match_result == None:
            try:
                line = line.replace("\n", "").replace(" ", "")
                dtype_name, const_names_str = line.split(":")
                dtype = DataType(dtype_name)
                const_names = const_names_str.split(",")
                constants.extend(
                    [Const(const_name, dtype) for const_name in const_names]
                )
            except ValueError:
                next
        return constants

    def _parse_response_to_predicates(self, response):
        predicates = []
        lines = response.split("\n")
        for line in lines:
            line = line.replace("\n", "").replace(" ", "")
            pred_names = line.split(",")
            for pred_name in pred_names:
                # we asuume LLM generates only object-object relations
                datatypes = [DataType("object"), DataType("object")]
                predicates.append(NeuralPredicate(pred_name, 2, datatypes))
        return predicates

    def _parse_response_to_rules(self, response, language):
        rules = []
        lines = response.split("\n")
        for line in lines:
            # todo: format checker?
            # todo: better checker
            if ":-" in line:
                line = line.replace(" ", "")
                tree = self.lp_clause.parse(line)
                rule = ExpTree(language).transform(tree)
                rules.append(rule)
        return rules

    def query_gpt(self, text):
        """Query to GPT3.5 with a textual prompt.

        Args:
            text (str): A textual prompt.

        Returns:
            str: A response of GPT3.5.
        """
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="gpt-4",
                messages=[
                    {"role": "user", "content": text},
                ],
            )
            answer = response.choices[0]["message"]["content"].strip()
            return answer
        except (
            openai.error.APIError,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
            json.decoder.JSONDecodeError,
        ):
            time.sleep(3)
            print("Found openai API error. Retrying...")
            return self.query_gpt(text)

    def generate_constants(self, text):
        prompt = self.constants_prompt + "\n" + text + "\n Constants:"
        response = self.query_gpt(prompt)
        return self._parse_response_to_constants(response), "Constants:\n" + response

    def generate_predicates(self, text, const_response):
        prompt = self.predicates_prompt + "\n" + text + "\n"
        response = self.query_gpt(prompt)
        predicates, pred_response = (
            self._parse_response_to_predicates(response),
            response,
        )
        # check if type/target predicates is generated
        if not "type" in [p.name for p in predicates]:
            obj_dt = DataType("object")
            type_dt = DataType("type")
            p_type = NeuralPredicate("type", 2, [obj_dt, type_dt])
            predicates.append(p_type)
        if not "target" in [p.name for p in predicates]:
            obj_dt = DataType("object")
            p_target = Predicate("target", 1, [obj_dt])
            predicates.append(p_target)
        if not "cond1" in [p.name for p in predicates]:
            obj_dt = DataType("object")
            p_cond = Predicate("cond1", 1, [obj_dt])
            predicates.append(p_cond)
        if not "cond2" in [p.name for p in predicates]:
            obj_dt = DataType("object")
            p_cond = Predicate("cond2", 1, [obj_dt])
            predicates.append(p_cond)
        if not "cond3" in [p.name for p in predicates]:
            obj_dt = DataType("object")
            p_cond = Predicate("cond3", 1, [obj_dt])
            predicates.append(p_cond)
        return predicates, pred_response

    def get_preds_string(self, preds):
        # generate "pred1,pred2,...,predn" as string from a list of predicates.
        result = ""
        for pred in preds:
            if not pred.name in ["target", "type", "cond1", "cond2", "cond3"]:
                result += pred.name
                result += ","
        return result[:-1]

    def generate_rules(self, text, language):
        """Generate FOL rules given deictic prompt and FOL language.

        Args:
            text (str): A deictic prompt.
            language (neumann.fol.language.Language): A FOL language.

        Returns:
            list[neumann.fol.logic.Clause]: A list of generated FOL rules.
        """
        # prompt = self.rules_prompt + "\n" + text + "\n Rules:"
        pred_response = self.get_preds_string(language.preds)
        self.pred_response = pred_response
        prompt = (
            " \n\n"
            + self.rules_prompt
            + "\n"
            + text
            + "\n"
            # + const_response
            # + "\n"
            + "available predicates: "
            + pred_response
            + "\n"
            # + "\n Rules:"
        )
        # print("Prompt: ", prompt)
        response = self.query_gpt(prompt)
        self.rule_response = response

        # print("==== Prompt; ")
        # print(prompt)
        # print("==== Pred response:")
        # print("    ", pred_response)
        # print("==== Rule response:")
        # print(response)
        try:
            rules = self._parse_response_to_rules(response, language)
            return rules
        except lark.exceptions.UnexpectedEOF:
            # no period in the response
            if response[-1] != ".":
                rules = rules = self._parse_response_to_rules(response + ".", language)
                return rules
            else:
                # there are new line
                rules = rules = self._parse_response_to_rules(
                    response.replace("\n", ""), language
                )
                return rules

    def generate_logic(self, text):
        """Generate constants, predicates, and rules using LLMs. Currently this function is not used."""
        constants, const_response = self.generate_constants(text)
        assert not len(constants) == 0, "Error: No constants found in the prompt."
        predicates, pred_response = self.generate_predicates(text, const_response)
        assert not len(predicates) == 0, "Error: No predicates found in the prompt."
        language = Language(consts=constants, preds=predicates, funcs=[])
        rules = self.generate_rules(text, const_response, pred_response, language)
        return language, rules
