#=========================================================================
# Quantize LLM to 4-bit and save TRUE quantized weights
#=========================================================================

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

#=========================================================================

def quantize_and_save_model(model_name, save_directory):
    # 4-bit quantization settings (recommended defaults)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",        # better than fp4 in most cases
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load model directly in 4-bit (no higher-precision weights kept)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False
    )

    # Attach config to ensure reload stays quantized
    model.config.quantization_config = bnb_config

    # Save quantized weights (safetensors recommended)
    model.save_pretrained(
        save_directory,
        safe_serialization=True
    )

    print(f"TRUE 4-bit quantized model saved to: {save_directory}")


# Example usage
if __name__ == "__main__":
    quantize_and_save_model(
        "path/to/bf16-model",
        "path/to/save/quantized-model"
    )
