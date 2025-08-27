# Problem1 

> What Unicode character does chr(0) return?
-> NULL

> How does this character’s string representation (__repr__()) differ from its printed representation?
-> __repr__() returns haxadecimal escape sequence; its unambiguous representation; shows exactly what characters are
printed representation: null is invisible; 
The key insight is:
repr() gives you a developer-friendly, unambiguous representation that shows '\x00'
print() attempts to display the actual character, which for null is nothing visible


---

# Problem 2

> What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings
-> 更小的文件尺寸和更高效的存储， 兼容性较好

> decode_utf8_bytes_to_str_wrong
-> It tries to decode individual characters into UTF-8; However, some characters may need multiple UTF-8 (variable-length encoding)

---

Q: Consider GPT-2 XL, which has the following configuration:
vocab_size : 50,257
context_length : 1,024
num_layers : 48
d_model : 1,600
num_heads : 25
d_ff : 6,400
Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

A:
```
Parameters: 
input_param = vocab_size * d_model = 50,257 * 1,600
output_param = vocab_size * d_model = 50,257 * 1,600
position_embedding = context_length * d_model = 1,024 * 1,600
mha = (Q,K,A,Output) * d_model^2 = 4 * 1,600 ^ 2
mha_rmsnorm = d_model = 1,600
ffn = d_model * d_ff + d_ff * d_model = 2 * (d_ff * d_model) = 2 * (6,400 * 1,600)
ffn_rmsnorm = d_model = 1,600
transformer = num_layers * (mha_rmsnorm + mha + ffn_rmsnorm + ffn)
total_param = input_param + output_param + position_embedding + transformer
= 160,822,400 + 160,822,400 + 1,638,400 + 1,474,713,600
= 1,637,174,400
= 1.6 billion

Memory Usage:
single float: 4 bytes
= 1,637,174,400 * 4
= 6.56 * 10^9 bytes 
= 6 GB
```