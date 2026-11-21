"""Run the BFCL command line with our Granite 4.2 handler registered (BFCL's own source stays untouched).

  python bfcl_run.py generate --model granite42-xml-FC --test-category simple_python --backend vllm ...
  python bfcl_run.py evaluate --model granite42-xml-FC --test-category simple_python
"""
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

from bfcl_granite import Granite42Handler

MODEL_CONFIG_MAPPING["granite42-xml-FC"] = ModelConfig(
    model_name="ibm-granite/granite-4.2-3b",       # Hugging Face id; --local-model-path points at our merged models
    display_name="Granite 4.2 format (FC; our handler)",   # no comma in the name: BFCL writes unquoted CSV (B7)
    url="https://huggingface.co/ibm-granite/granite-4.2-3b",
    org="IBM",
    license="Apache-2.0",
    model_handler=Granite42Handler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,                         # our prompts turn "a.b" into "a_b"; the checker maps it back
)

if __name__ == "__main__":
    from bfcl_eval.__main__ import cli
    cli()
