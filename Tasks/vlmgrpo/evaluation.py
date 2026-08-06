#evaluation file 
def load_model():
    model,tokenizer=FastVisionModel.from_pretrained(
         model, tokenizer = FastVisionModel.from_pretrained(
        model_name=TrainConfig.OUTPUT_DIR,
        max_seq_length=ModelConfig.MAX_SEQ_LENGTH,
        dtype=ModelConfig.TORCH_DTYPE,
        load_in_4bit=ModelConfig.LOAD_IN_4BIT,
        device_map=ModelConfig.DEVICE_MAP,
    )
    model.eval()

    return model, tokenizer

    )