import torch
from diffusers.models.attention import CrossAttention

def get_attention_scores(self, query, key, attention_mask=None):
    dtype = query.dtype
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    attention_scores = torch.baddbmm(
        torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
        query,
        key.transpose(-1, -2),
        beta=0,
        alpha=self.scale,
    )

    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    if self.upcast_softmax:
        attention_scores = attention_scores.float()
    
    attention_scores = attention_scores/1.25
    attention_probs = attention_scores.softmax(dim=-1)
    attention_probs = attention_probs.to(dtype)

    return attention_probs

class MyCrossAttnProcessor:
    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn.get_attention_scores = get_attention_scores.__get__(attn, type(attn))
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # new bookkeeping to save the attn probs
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):
    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        if 'attn2' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.set_processor(MyCrossAttnProcessor())
    return unet
