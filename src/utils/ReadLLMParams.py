import argparse

from src.utils.Constants import Constants

LLMProcessorConstants = Constants.LLMProcessorConstants


class ReadLLMParams:
    @staticmethod
    def read_llm_params(args: argparse.ArgumentParser):
        args.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
        args.add_argument("--not_load_in_4bit", action="store_false", default=LLMProcessorConstants.LOAD_IN_4BIT,
                          help="True if the model should be loaded in 4-bit.")
        args.add_argument("--not_load_in_8bit", action="store_false", default=LLMProcessorConstants.LOAD_IN_8BIT,
                          help="True if the model should be loaded in 8-bit.")
        args.add_argument("--trust_remote_code", action="store_true", default=LLMProcessorConstants.TRUST_REMOTE_CODE,
                          help="True if the model should trust remote code.")
        args.add_argument("--not_return_token_type_ids", action="store_false",
                          default=LLMProcessorConstants.RETURN_TOKEN_TYPE_IDS,
                          help="True if the model should not return token type ids.")
        return args