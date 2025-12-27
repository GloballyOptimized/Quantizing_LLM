#=========================================================================
# Quantize LLM to 4 bit 
#=========================================================================

#Library Imports 

from transformers import AutoModelForCausalLM, BitsAndBytesConfig

#=========================================================================

def quantize_and_save_model(model_name, save_directory):
    # Set up the bitsandbytes config for int4 quantization
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    
    # Load the model in BF16 (or whatever its original format is) and quantize it
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config
    )

    # Save the quantized model
    model.save_pretrained(save_directory)
    print(f"Model saved to {save_directory} in int4 format")

# Example usage
if __name__ == "__main__":
    quantize_and_save_model("path/to/bf16-model", "path/to/save/quantized-model")

#===================================================================================
