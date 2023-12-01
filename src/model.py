import torch
from transformers import (
    AutoConfig,
    BartTokenizer,
    BartForConditionalGeneration,
    BartConfig,
)
from transformers.models.bart.modeling_bart import shift_tokens_right


class BartWithControlConfig(BartConfig):
    """
    Extension of the BART Configuration Class to include parameters for control signal embedding.

    Attributes:
        control_signal_dim (int): Dimension of the control signal.
        control_embedding_dim (int): Dimension of the control embedding.
        epsilon (float): A small value for initializing random weights in the embedding transformation.
    """

    def __init__(
        self,
        control_signal_dim=8,
        control_embedding_dim=16,
        control_epsilon=0.01,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.control_signal_dim = control_signal_dim
        self.control_embedding_dim = control_embedding_dim
        self.control_epsilon = control_epsilon


class BartWithConcatControlEmbedding(BartForConditionalGeneration):
    """
    Modified BART Model Class that concatenates control embeddings with input embeddings.

    This class extends BartForConditionalGeneration by adding a control signal embedding,
    which is concatenated with the regular input embeddings before being fed into the encoder.

    Attributes:
        control_embedding (torch.nn.Linear): A linear layer for control embedding.
        embedding_transformation (torch.nn.Linear): A linear transformation layer to combine
                                                    control and input embeddings.
    """

    def __init__(self, config):
        super().__init__(config)
        self.control_embedding = torch.nn.Linear(
            config.control_signal_dim, config.d_model
        )
        self._init_control_embed(config.control_epsilon)

    def _init_control_embed(self, epsilon):
        """
        Initializes the embedding transformation layer.

        Args:
            input_dim (int): Dimension of the input embeddings.
            control_dim (int): Dimension of the control embeddings.
            epsilon (float): A small value for initializing random weights.
        """
        self.control_embedding.weight.data *= epsilon
        self.control_embedding.bias.data.zero_()

    def forward(self, input_ids, control_signal, **kwargs):
        """
        Defines the forward pass of the model with control signal integration.

        Args:
            input_ids (torch.Tensor): Input IDs for tokenized text.
            control_signal (torch.Tensor): Control signal tensor.
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass.
        """

        decoder_input_ids = shift_tokens_right(
            input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
        )
        inputs_embeds = self._embed_input(input_ids)
        decoder_inputs_embeds = self._embed_input(decoder_input_ids)

        control_embed = (
            self.control_embedding(control_signal)
            .unsqueeze(0)
            .expand(input_ids.size(0), -1, -1)
        )

        combined_embeddings = inputs_embeds + control_embed
        decoder_combined_embeddings = decoder_inputs_embeds + control_embed

        return super().forward(
            inputs_embeds=combined_embeddings,
            decoder_inputs_embeds=decoder_combined_embeddings,
            **kwargs
        )

    def _embed_input(self, x):
        if len(x.size()) == 2:
            # input is normal tokens
            return self.model.encoder.embed_tokens(x)
        # input is soft tokens
        return torch.matmul(x, self.model.encoder.embed_tokens.weight)


if __name__ == "__main__":
    # Instantiate the Model with Custom Config
    custom_config = BartWithControlConfig()
    bart_config = AutoConfig.from_pretrained("facebook/bart-large")
    custom_config.update(bart_config.to_dict())

    model = BartWithConcatControlEmbedding(custom_config)

    # Example usage
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    # Example control signal and input
    control_signal = torch.randn(custom_config.control_signal_dim)
    input_text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Forward pass
    output = model(input_ids=input_ids, control_signal=control_signal)
